#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
import re
import sys
import inspect
from pathlib import Path

import numpy as np

try:
    import torch
    import torchaudio
except Exception as e:
    raise SystemExit(f"[error] torch/torchaudio not available: {e}")

try:
    import soundfile as sf
except Exception as e:
    raise SystemExit(f"[error] soundfile not available: {e}")

# ---- Robust Turbo import (different installs expose it differently) ----
TurboClass = None
_import_errors = []

for mod, name in [
    ("chatterbox.tts_turbo", "ChatterboxTurboTTS"),  
    ("chatterbox.turbo_tts", "ChatterboxTurboTTS"),
    ("chatterbox.tts", "ChatterboxTurboTTS"),
    ("chatterbox", "ChatterboxTurboTTS"),
]:
    try:
        m = __import__(mod, fromlist=[name])
        TurboClass = getattr(m, name)
        break
    except Exception as e:
        _import_errors.append(f"{mod}.{name}: {e}")

if TurboClass is None:
    raise SystemExit(
        "[error] Could not import ChatterboxTurboTTS.\n"
        "Tried:\n  - " + "\n  - ".join(_import_errors) + "\n"
        "Fix: reinstall/update the chatterbox package used by this repo."
    )


SUPPORTED_TAGS = (
    "[laugh]", "[chuckle]", "[sigh]", "[gasp]", "[cough]",
    "[clear throat]", "[sniff]", "[groan]", "[shush]",
)
TAG_PATTERN = "(?:" + "|".join(re.escape(tag) for tag in SUPPORTED_TAGS) + ")"
TAG_ONLY_RE = re.compile(rf"^(?:\s*{TAG_PATTERN}\s*)+$")


def normalize_text(text: str) -> str:
    """Normalize horizontal spacing without destroying poetic structure."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t\f\v]+", " ", line).strip() for line in text.split("\n")]
    # Preserve stanza breaks, while reducing accidental runs of blank lines.
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _line_units(line: str) -> list[str]:
    """Split one line at sentence endings and retain trailing reaction tags."""
    pattern = re.compile(rf".*?[.!?。！？]+(?:\s*{TAG_PATTERN})*|.+$", re.IGNORECASE)
    units = [m.group(0).strip() for m in pattern.finditer(line) if m.group(0).strip()]

    # A tag-only fragment belongs to its neighboring spoken phrase.
    bonded: list[str] = []
    pending = ""
    for unit in units:
        if TAG_ONLY_RE.fullmatch(unit):
            if bonded:
                bonded[-1] += " " + unit
            else:
                pending = (pending + " " + unit).strip()
        else:
            bonded.append(((pending + " ") if pending else "") + unit)
            pending = ""
    if pending:
        if bonded:
            bonded[-1] += " " + pending
        else:
            bonded.append(pending)
    return bonded


def split_sentences(text: str, max_chars: int = 240) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []

    # Each unit remembers the line/stanza separator that preceded it.
    parts: list[tuple[str, str]] = []
    pending_sep = ""
    for line in text.split("\n"):
        if not line:
            pending_sep = "\n\n"
            continue
        units = _line_units(line)
        for index, unit in enumerate(units):
            sep = pending_sep if index == 0 else " "
            parts.append((sep, unit))
            pending_sep = "\n"

    # Never leave a standalone reaction tag at a chunk boundary.
    bonded_parts: list[tuple[str, str]] = []
    leading_tags = ""
    for sep, part in parts:
        if TAG_ONLY_RE.fullmatch(part):
            if bonded_parts:
                old_sep, old_part = bonded_parts[-1]
                bonded_parts[-1] = (old_sep, old_part + sep + part)
            else:
                leading_tags = (leading_tags + " " + part).strip()
        else:
            if leading_tags:
                part = leading_tags + " " + part
                leading_tags = ""
            bonded_parts.append((sep, part))
    parts = bonded_parts

    chunks: list[str] = []
    buf = ""
    for sep, p in parts:
        if not buf:
            buf = p
        elif len(buf) + len(sep) + len(p) <= max_chars:
            buf += sep + p
        else:
            chunks.append(buf)
            buf = p
    if buf:
        chunks.append(buf)
    return chunks


def crossfade(a: np.ndarray, b: np.ndarray, fade_samples: int) -> np.ndarray:
    fade_samples = int(max(0, min(fade_samples, len(a), len(b))))
    if fade_samples <= 0:
        return np.concatenate([a, b])

    # equal-power crossfade
    t = np.linspace(0, np.pi / 2, fade_samples, dtype=np.float32)
    fade_out = (np.cos(t) ** 2).astype(np.float32)
    fade_in = (np.sin(t) ** 2).astype(np.float32)

    a_tail = a[-fade_samples:]
    b_head = b[:fade_samples]
    mid = a_tail * fade_out + b_head * fade_in

    return np.concatenate([a[:-fade_samples], mid, b[fade_samples:]])


def call_generate(model, **kwargs):
    """
    Call model.generate(...) but only pass kwargs it actually accepts.
    This makes the script resilient to minor API differences.
    """
    sig = inspect.signature(model.generate)
    accepted = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if (k in accepted and v is not None)}
    return model.generate(**filtered)


def seed_generators(seed: int) -> None:
    """Seed every RNG Chatterbox/Torch may use before one chunk."""
    # NumPy requires a 32-bit unsigned seed. Python and Torch accept the
    # original integer, so preserve it there and only normalize for NumPy.
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        torch.mps.manual_seed(seed)


def filename_part(value: str, fallback: str, strip_extension: bool = False) -> str:
    """Return a readable Windows-safe filename component."""
    if value and strip_extension:
        value = re.split(r"[/\\]", value)[-1]
        value = value.rsplit(".", 1)[0] if "." in value else value
    value = value or fallback
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "-", value)
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"\s*-\s*", " - ", value).strip(" .-_")
    return value or fallback


def metadata_output_name(args: argparse.Namespace) -> str:
    """Build [text][reference][style][speed][seed].wav."""
    text_name = filename_part(args.text_file or "", "text", strip_extension=True)
    ref_name = filename_part(args.ref_wav or "", "no-ref", strip_extension=True)
    style_name = filename_part(args.style or "", "Custom")
    speed_name = filename_part(f"{args.speed_factor:g}", "1")
    seed_name = filename_part(
        "random" if args.seed is None else str(args.seed), "random"
    )
    return (
        f"[{text_name}][{ref_name}][{style_name}]"
        f"[{speed_name}][{seed_name}].wav"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default=None)
    ap.add_argument("--text_file", default=None)

    ap.add_argument("--ref_wav", default=None)

    # Turbo-ish knobs
    ap.add_argument("--exaggeration", type=float, default=None)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--cfg_weight", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--speed_factor", type=float, default=1.0)
    ap.add_argument("--language", default="en")
    ap.add_argument("--style", default="Custom")

    # chunking
    ap.add_argument("--split_text", action="store_true")
    ap.add_argument("--chunk_chars", type=int, default=200)
    ap.add_argument("--crossfade_ms", type=int, default=35)
    ap.add_argument("--lead_in_ms", type=int, default=150)

    # io
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--out_name", default=None)

    # device
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])

    args = ap.parse_args()

    # load text
    if args.text_file:
        p = Path(args.text_file)
        if not p.exists():
            raise SystemExit(f"[error] text_file not found: {p}")
        text = p.read_text(encoding="utf-8").strip()
    else:
        text = (args.text or "").strip()

    if not text:
        raise SystemExit("[error] empty text")

    if (not args.split_text) and (len(text) >= 260):
        args.split_text = True

    ref = None
    if args.ref_wav:
        rp = Path(args.ref_wav)
        if not rp.exists():
            raise SystemExit(f"[error] ref_wav not found: {rp}")
        ref = str(rp)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = args.out_name or metadata_output_name(args)
    out_path = out_dir / out_name
    print(f"[info] out_dir={out_dir}  out_name={out_name}")
    print("[info] supported Turbo tags: " + " ".join(SUPPORTED_TAGS))

    # load model
    device = args.device
    print(f"[info] loading Turbo model on {device}...")
    model = TurboClass.from_pretrained(device=device)
    print("[info] model loaded.")

    # choose chunks
    if args.split_text:
        chunks = split_sentences(text, max_chars=args.chunk_chars)
    else:
        chunks = [normalize_text(text)]

    stitched = None
    sr_final = None

    for i, ch in enumerate(chunks, start=1):
        print(f"[gen] chunk {i}/{len(chunks)} ({len(ch)} chars)")

        # Use one fixed seed for consistent narration across every chunk.
        if args.seed is None:
            seed_i = None
        else:
            seed_i = int(args.seed)

        if seed_i is not None:
            seed_generators(seed_i)
            print(f"[seed] chunk {i}/{len(chunks)} = {seed_i}")

        wav = call_generate(
            model,
            text=ch,
            audio_prompt_path=ref,
            exaggeration=args.exaggeration,
            temperature=args.temperature,
            cfg_weight=args.cfg_weight,
            seed=seed_i,
            speed_factor=args.speed_factor,
            language=args.language,
        )

        # handle return styles
        if isinstance(wav, (tuple, list)) and len(wav) >= 1:
            wav_tensor = wav[0]
            sr = wav[1] if len(wav) >= 2 else 24000
        else:
            wav_tensor = wav
            sr = 24000

        if isinstance(wav_tensor, torch.Tensor):
            wav_np = wav_tensor.detach().cpu().float().numpy()
        else:
            wav_np = np.asarray(wav_tensor, dtype=np.float32)

        # force mono
        if wav_np.ndim > 1:
            wav_np = wav_np.squeeze()
        if wav_np.ndim != 1:
            wav_np = wav_np.reshape(-1)

        if stitched is None:
            stitched = wav_np
            sr_final = sr
        else:
            if sr_final != sr:
                print(f"[warn] sample rate changed {sr_final} -> {sr}")
            fade_samples = int((args.crossfade_ms / 1000.0) * (sr_final or sr))
            stitched = crossfade(stitched, wav_np, fade_samples)

    if stitched is None:
        raise SystemExit("[error] no audio generated")

    lead = int((args.lead_in_ms / 1000.0) * float(sr_final or 24000))
    if lead > 0:
        stitched = np.concatenate(
            [np.zeros(lead, dtype=np.float32), stitched.astype(np.float32)]
        )

    sf.write(str(out_path), stitched, sr_final or 24000)

    # HARD VERIFY
    if (not out_path.exists()) or (out_path.stat().st_size < 1000):
        raise SystemExit(f"[error] write failed or empty file: {out_path}")

    print(f"[OK] saved: {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

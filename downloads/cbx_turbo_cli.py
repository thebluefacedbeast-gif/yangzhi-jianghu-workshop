#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
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


def split_sentences(text: str, max_chars: int = 240) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    # sentence-ish split including JP punctuation
    parts = re.split(r"(?<=[\.\!\?\。\！\？])\s+", text)
    parts = [p.strip() for p in parts if p.strip()]

    chunks: list[str] = []
    buf = ""
    for p in parts:
        if not buf:
            buf = p
        elif len(buf) + 1 + len(p) <= max_chars:
            buf += " " + p
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
    out_name = args.out_name or "cbx_turbo_out.wav"
    out_path = out_dir / out_name
    print(f"[info] out_dir={out_dir}  out_name={out_name}")

    # load model
    device = args.device
    print(f"[info] loading Turbo model on {device}...")
    model = TurboClass.from_pretrained(device=device)
    print("[info] model loaded.")

    # choose chunks
    if args.split_text:
        chunks = split_sentences(text, max_chars=args.chunk_chars)
    else:
        chunks = [text]

    stitched = None
    sr_final = None

    for i, ch in enumerate(chunks, start=1):
        print(f"[gen] chunk {i}/{len(chunks)} ({len(ch)} chars)")

        # base_seed + chunk_index logic
        if args.seed is None:
            seed_i = None
        else:
            seed_i = int(args.seed) + (i - 1)

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
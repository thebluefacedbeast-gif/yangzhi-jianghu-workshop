#!/usr/bin/env python3
"""Build one silent Heart Dao slideshow from a downloaded episode folder.

The episode folder must contain a WAV narration file, a TXT script, and folders
named "images" and "images 1". All source images are used at least once. The
older "images" set remains primary; "images 1" cards are secondary accents.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
W, H, FPS = 1920, 1080, 24
MAIN_HOLD, SECONDARY_HOLD, TITLE_HOLD, PREVIEW_HOLD = 13.0, 18.0, 12.0, 10.0
FADE = 0.65
UNWANTED_NOTE = "redone pack. drive selections restricted to the heart dao image folder only"


@dataclass(frozen=True)
class Asset:
    source: Path
    prepared: Path
    group: str
    role: str
    hold: float
    cleaned_footer: bool = False


def run(cmd: Sequence[str], capture: bool = False) -> str:
    print("+", " ".join(str(x) for x in cmd), flush=True)
    done = subprocess.run(
        list(map(str, cmd)), check=True, text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )
    return done.stdout.strip() if capture else ""


def natural_key(path: Path) -> list[object]:
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", path.name)]


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return re.sub(r"_+", "_", value).strip("_") or "Heart_Dao_Episode"


def find_one(root: Path, suffix: str) -> Path:
    items = sorted((p for p in root.rglob(f"*{suffix}") if p.is_file()), key=natural_key)
    if not items:
        raise FileNotFoundError(f"No {suffix} file found beneath {root}")
    return items[0]


def duration_of(path: Path) -> float:
    value = run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path)
    ], True)
    duration = float(value)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"Invalid WAV duration: {value!r}")
    return duration


def locate_image_dirs(root: Path) -> tuple[Path, Path]:
    dirs = [p for p in root.rglob("*") if p.is_dir()]
    main = next((p for p in dirs if p.name.strip().lower() == "images"), None)
    secondary = next((p for p in dirs if p.name.strip().lower() == "images 1"), None)
    if main is None or secondary is None:
        raise FileNotFoundError("Both 'images' and 'images 1' folders are required")
    return main, secondary


def collect_images(folder: Path) -> list[Path]:
    items = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    items.sort(key=natural_key)
    if not items:
        raise FileNotFoundError(f"No images found in {folder}")
    return items


def role_of(path: Path) -> str:
    name = path.stem.lower()
    if "preview" in name or "next_episode" in name or "next episode" in name:
        return "preview"
    if "title" in name or re.match(r"^0*1[_ -].*title", name):
        return "title"
    return "interior"


def fit_16_9(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    ratio = w / h
    target = W / H
    if ratio > target:
        new_w = int(round(h * target))
        x = max(0, (w - new_w) // 2)
        img = img[:, x:x + new_w]
    elif ratio < target:
        new_h = int(round(w / target))
        y = max(0, (h - new_h) // 2)
        img = img[y:y + new_h, :]
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_LANCZOS4)


def bottom_text_lines(img: np.ndarray) -> list[dict[str, float | int]]:
    h, w = img.shape[:2]
    y0 = int(h * 0.68)
    roi = img[y0:]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    masks = []
    for k in (9, 15, 21, 31):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        top = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        masks.append((top > 10).astype(np.uint8) * 255)
    masks.append(((hsv[:, :, 2] > 125) & (hsv[:, :, 1] < 145)).astype(np.uint8) * 255)
    mask = np.maximum.reduce(masks)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    comps = []
    for i in range(1, count):
        x, y, cw, ch, area = map(int, stats[i])
        if 2 <= area <= 2200 and 2 <= ch <= 45 and 1 <= cw <= 170:
            comps.append({"x": x, "y": y, "w": cw, "h": ch, "cy": y + ch / 2})
    lines: list[dict[str, object]] = []
    for comp in sorted(comps, key=lambda item: float(item["cy"])):
        for line in lines:
            if abs(float(comp["cy"]) - float(line["cy"])) <= 12:
                line_comps = line["components"]
                assert isinstance(line_comps, list)
                line_comps.append(comp)
                line["cy"] = sum(float(c["cy"]) for c in line_comps) / len(line_comps)
                break
        else:
            lines.append({"cy": float(comp["cy"]), "components": [comp]})
    found = []
    for line in lines:
        cs = line["components"]
        assert isinstance(cs, list)
        x1 = min(int(c["x"]) for c in cs)
        x2 = max(int(c["x"]) + int(c["w"]) for c in cs)
        ly1 = min(int(c["y"]) for c in cs)
        ly2 = max(int(c["y"]) + int(c["h"]) for c in cs)
        found.append({
            "x1": x1, "x2": x2, "y1": y0 + ly1, "y2": y0 + ly2,
            "span": x2 - x1, "count": len(cs), "cy": y0 + float(line["cy"])
        })
    return found


def remove_unwanted_footer(img: np.ndarray) -> tuple[np.ndarray, bool]:
    h, w = img.shape[:2]
    candidates = [
        line for line in bottom_text_lines(img)
        if float(line["cy"]) > h * 0.76
        and int(line["span"]) > w * 0.28
        and int(line["count"]) >= 25
        and int(line["x1"]) < w * 0.18
    ]
    if not candidates:
        return img, False
    line = max(candidates, key=lambda item: int(item["span"]) + 8 * int(item["count"]))
    crop_bottom = max(int(h * 0.76), int(line["y1"]) - max(8, int(h * 0.012)))
    crop_bottom = min(crop_bottom, h - 1)
    crop_w = min(w, int(round(crop_bottom * 16 / 9)))
    spare = max(0, w - crop_w)
    x0 = spare // 4
    cropped = img[:crop_bottom, x0:x0 + crop_w]
    return cv2.resize(cropped, (W, H), interpolation=cv2.INTER_LANCZOS4), True


def prepare(path: Path, group: str, folder: Path, index: int) -> Asset:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read {path}")
    cleaned = False
    if group == "secondary":
        img, cleaned = remove_unwanted_footer(img)
    if not cleaned:
        img = fit_16_9(img)
    role = role_of(path)
    hold = TITLE_HOLD if role == "title" else PREVIEW_HOLD if role == "preview" else (
        SECONDARY_HOLD if group == "secondary" else MAIN_HOLD
    )
    out = folder / f"{index:03d}_{group}_{slugify(path.stem)}.jpg"
    if not cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise IOError(f"Failed to write {out}")
    return Asset(path, out, group, role, hold, cleaned)


def cycle_main(items: list[Asset], count: int) -> list[Asset]:
    if not items:
        return []
    result: list[Asset] = []
    cycle = 0
    while len(result) < count:
        seq = items if cycle % 2 == 0 else list(reversed(items))
        offset = cycle % len(seq)
        result.extend(seq[offset:] + seq[:offset])
        cycle += 1
    return result[:count]


def interleave(main: list[Asset], secondary: list[Asset]) -> list[Asset]:
    if not secondary:
        return main
    if not main:
        return secondary
    slots: dict[int, list[Asset]] = {}
    for i, asset in enumerate(secondary):
        pos = round((i + 1) * len(main) / (len(secondary) + 1))
        slots.setdefault(pos, []).append(asset)
    result: list[Asset] = []
    for pos in range(len(main) + 1):
        result.extend(slots.get(pos, []))
        if pos < len(main):
            result.append(main[pos])
    return result


def make_timeline(assets: list[Asset], target: float) -> list[Asset]:
    mt = [a for a in assets if a.group == "main" and a.role == "title"]
    st = [a for a in assets if a.group == "secondary" and a.role == "title"]
    mp = [a for a in assets if a.group == "main" and a.role == "preview"]
    sp = [a for a in assets if a.group == "secondary" and a.role == "preview"]
    mi = [a for a in assets if a.group == "main" and a.role == "interior"]
    si = [a for a in assets if a.group == "secondary" and a.role == "interior"]
    titles, previews = mt + st, sp + mp
    fixed = sum(a.hold for a in titles + previews + si)
    main_count = max(len(mi), int(math.ceil(max(0.0, target - fixed) / MAIN_HOLD))) if mi else 0
    timeline = titles + interleave(cycle_main(mi, main_count), si) + previews
    if not timeline:
        raise ValueError("No timeline images")
    total = sum(a.hold for a in timeline)
    pool = mi or si or titles or previews
    insert_at = max(len(titles), len(timeline) - len(previews))
    i = 0
    while total < target + 1.0:
        asset = pool[i % len(pool)]
        timeline.insert(insert_at, asset)
        insert_at += 1
        total += asset.hold
        i += 1
    return timeline


def render_segment(asset: Asset, out: Path, variant: int) -> None:
    fade_start = max(0.0, asset.hold - FADE)
    zoom = "min(max(zoom,pzoom)+0.00013,1.045)" if variant % 2 == 0 else "if(eq(on,0),1.045,max(1.0,pzoom-0.00013))"
    vf = (
        "scale=2000:1125:force_original_aspect_ratio=increase,crop=2000:1125,"
        f"zoompan=z='{zoom}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s={W}x{H}:fps={FPS},"
        f"fade=t=in:st=0:d={FADE:.3f},fade=t=out:st={fade_start:.3f}:d={FADE:.3f},format=yuv420p"
    )
    run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-loop", "1",
        "-framerate", str(FPS), "-i", str(asset.prepared), "-t", f"{asset.hold:.3f}",
        "-vf", vf, "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "21",
        "-r", str(FPS), "-g", str(FPS * 2), "-keyint_min", str(FPS * 2),
        "-sc_threshold", "0", "-pix_fmt", "yuv420p", str(out)
    ])


def validate(path: Path, expected: float) -> dict[str, object]:
    probe = json.loads(run([
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)
    ], True))
    streams = probe.get("streams", [])
    video = [s for s in streams if s.get("codec_type") == "video"]
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    if len(video) != 1 or audio:
        raise ValueError("Output must contain one video stream and no audio stream")
    v = video[0]
    if int(v.get("width", 0)) != W or int(v.get("height", 0)) != H:
        raise ValueError("Unexpected output resolution")
    actual = float(probe["format"]["duration"])
    if abs(actual - expected) > 0.35:
        raise ValueError(f"Duration mismatch: {actual:.3f}s vs {expected:.3f}s")
    return {"duration_seconds": actual, "width": W, "height": H, "fps": v.get("avg_frame_rate"), "codec": v.get("codec_name"), "audio_streams": 0}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--episode-name", required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    root = args.source.resolve()
    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = slugify(args.episode_name)
    wav, txt = find_one(root, ".wav"), find_one(root, ".txt")
    target = duration_of(wav)
    text = txt.read_text(encoding="utf-8", errors="replace")
    word_count = len(re.findall(r"\b[\w'’-]+\b", text))
    main_dir, secondary_dir = locate_image_dirs(root)
    main_images, secondary_images = collect_images(main_dir), collect_images(secondary_dir)

    with tempfile.TemporaryDirectory(prefix="heart_dao_") as temp_name:
        temp = Path(temp_name)
        prepared, segments = temp / "prepared", temp / "segments"
        prepared.mkdir(); segments.mkdir()
        assets: list[Asset] = []
        for i, path in enumerate(main_images, 1):
            assets.append(prepare(path, "main", prepared, i))
        offset = len(assets)
        for i, path in enumerate(secondary_images, 1):
            assets.append(prepare(path, "secondary", prepared, offset + i))
        timeline = make_timeline(assets, target)
        segment_map: dict[Path, Path] = {}
        for i, asset in enumerate(assets):
            seg = segments / f"segment_{i:03d}.mp4"
            render_segment(asset, seg, i)
            segment_map[asset.prepared] = seg
        concat = temp / "timeline.ffconcat"
        with concat.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write("ffconcat version 1.0\n")
            for asset in timeline:
                escaped = str(segment_map[asset.prepared].resolve()).replace("'", "'\\''")
                handle.write(f"file '{escaped}'\n")
        mp4 = out_dir / f"{slug}_Silent_Slideshow_1080p.mp4"
        run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat",
            "-safe", "0", "-i", str(concat), "-t", f"{target:.6f}", "-an", "-r", str(FPS),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(mp4)
        ])
        output_info = validate(mp4, target)
        cleaned = [str(a.source.relative_to(root)) for a in assets if a.cleaned_footer]
        counts: dict[str, int] = {}
        for asset in timeline:
            key = str(asset.source.relative_to(root))
            counts[key] = counts.get(key, 0) + 1
        manifest = {
            "episode": args.episode_name,
            "source_wav": str(wav.relative_to(root)),
            "source_text": str(txt.relative_to(root)),
            "source_text_word_count": word_count,
            "wav_duration_seconds": target,
            "main_images_folder": str(main_dir.relative_to(root)),
            "secondary_images_folder": str(secondary_dir.relative_to(root)),
            "main_image_count": len(main_images),
            "secondary_image_count": len(secondary_images),
            "all_unique_images_used": len({a.source.resolve() for a in assets}) == len(main_images) + len(secondary_images),
            "timeline_entries": len(timeline),
            "cleaned_unwanted_footer_count": len(cleaned),
            "cleaned_unwanted_footer_images": cleaned,
            "removed_text": UNWANTED_NOTE,
            "ordering": "images = main; images 1 = secondary; main repeats only as needed",
            "transition": f"{FADE:.2f}s fade-through-black with subtle slow zoom",
            "output": output_info,
            "timeline_source_counts": counts,
        }
        manifest_path = out_dir / f"{slug}_Slideshow_Manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        zip_path = out_dir / f"{slug}_Silent_Slideshow.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.write(mp4, mp4.name)
            archive.write(manifest_path, manifest_path.name)
    print(json.dumps({"episode": args.episode_name, "mp4": str(mp4), "zip": str(zip_path), "duration": target, "main_images": len(main_images), "secondary_images": len(secondary_images)}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(f"Command failed with exit code {exc.returncode}", file=sys.stderr)
        if exc.stdout:
            print(exc.stdout, file=sys.stderr)
        if exc.stderr:
            print(exc.stderr, file=sys.stderr)
        raise

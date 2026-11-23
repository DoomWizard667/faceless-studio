from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from .agents.quality import (
    ControleRendu,
    ControleSynchro,
    ControleVisuels,
    ControleVoix,
)

VERTICAL_SIZE = (1080, 1920)
HORIZONTAL_SIZE = (1920, 1080)


def _stamp_to_seconds(stamp: str) -> float:
    hours, minutes, rest = stamp.split(":")
    seconds, _, millis = rest.partition(",")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000.0


def parse_srt(path: Path) -> list:
    if not path.exists():
        return []
    entries = []
    blocks = re.split(r"\n\s*\n", path.read_text(encoding="utf-8").strip())
    for block in blocks:
        lines = [line for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        timing = next((l for l in lines if "-->" in l), None)
        if not timing:
            continue
        left, _, right = timing.partition("-->")
        start = _stamp_to_seconds(left.strip())
        end = _stamp_to_seconds(right.strip().split(" ")[0])
        text = " ".join(lines[lines.index(timing) + 1 :])
        entries.append((start, end, text))
    return entries


def _extract_frames(video_path: Path, count: int, tmp_dir: Path) -> list:
    from moviepy import VideoFileClip

    tmp_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    with VideoFileClip(str(video_path)) as clip:
        duration = float(clip.duration)
        times = [duration * (i + 0.5) / count for i in range(count)]
        for i, t in enumerate(times):
            frame = clip.get_frame(t)
            image = Image.fromarray(frame)
            path = tmp_dir / f"frame_{i:02d}.png"
            image.save(path)
            frames.append(str(path))
    return frames


def _frame_caption_detected(frame_path: str) -> bool:
    arr = np.asarray(Image.open(frame_path).convert("L"), dtype=np.float32)
    height = arr.shape[0]
    caption_zone = arr[int(height * 0.58) : int(height * 0.84), :]
    return float(caption_zone.max()) > 200


def audit_video(video_path: Path, srt_dir: Path, label: str, frames_count: int = 8) -> dict:
    from moviepy import VideoFileClip

    result = {
        "fichier": video_path.name,
        "label": label,
        "issues": [],
        "verdict": "À REVOIR",
    }
    if not video_path.exists():
        result["issues"].append("fichier introuvable")
        return result

    with VideoFileClip(str(video_path)) as clip:
        duration = float(clip.duration)
        fps = float(clip.fps)
        size = (int(clip.size[0]), int(clip.size[1]))
        audio = clip.audio
        audio_duration = float(audio.duration) if audio else 0.0
    expected = VERTICAL_SIZE if label == "9x16" else HORIZONTAL_SIZE

    if size != expected:
        result["issues"].append(f"résolution {size[0]}x{size[1]} ≠ {expected[0]}x{expected[1]}")
    if fps < 23:
        result["issues"].append(f"fps {fps:.1f} < 24")
    if not audio or audio_duration < 1:
        result["issues"].append("piste audio absente ou muette")
    if duration > 0 and audio_duration and abs(duration - audio_duration) > 2.0:
        result["issues"].append(f"vidéo {duration:.1f}s ≠ audio {audio_duration:.1f}s")

    captions = parse_srt(srt_dir / "captions.srt")
    sentences = parse_srt(srt_dir / "sentences.srt")
    result["srt"] = {"captions": len(captions), "sentences": len(sentences)}

    def caption_expected(t: float) -> bool:
        return any(s - 0.25 <= t <= e + 0.25 for s, e, _ in (captions or sentences))

    tmp_dir = video_path.parent / "tmp_qc_frames"
    from moviepy import VideoFileClip as _VFC

    frames = []
    tmp_dir.mkdir(parents=True, exist_ok=True)
    with _VFC(str(video_path)) as clip:
        span = float(clip.duration)
        times = [span * (i + 0.5) / frames_count for i in range(frames_count)]
        for i, t in enumerate(times):
            frame = clip.get_frame(t)
            path = tmp_dir / f"frame_{i:02d}_{t:.2f}s.png"
            Image.fromarray(frame).save(path)
            frames.append((t, str(path)))

    dark = []
    no_caption = []
    for t, frame in frames:
        arr = np.asarray(Image.open(frame).convert("L"), dtype=np.float32)
        mean = float(arr.mean())
        std = float(arr.std())
        if mean < 6 or std < 6:
            dark.append(f"t={t:.1f}s")
        if caption_expected(t) and not _frame_caption_detected(frame):
            no_caption.append(f"t={t:.1f}s")
    result["frames"] = {
        "analysées": len(frames),
        "frames_noires": dark,
        "frames_sans_sous_titres": no_caption,
    }
    if len(dark) > 0:
        result["issues"].append(f"{len(dark)} frame(s) quasi noire(s)")
    if len(no_caption) > 0:
        result["issues"].append(f"sous-titres manquants alors qu'un mot est prononcé : {no_caption}")

    if not captions and not sentences:
        result["issues"].append("aucun fichier SRT présent")
    else:
        combined = captions or sentences
        monotonic = all(combined[i][0] >= combined[i - 1][0] - 0.05 for i in range(1, len(combined)))
        if not monotonic:
            result["issues"].append("timings SRT non monotones")
        gaps = []
        for i in range(1, len(sentences)):
            gap = sentences[i][0] - sentences[i - 1][1]
            if gap > 1.5:
                gaps.append(round(gap, 2))
        if gaps:
            result["issues"].append(f"trous inquiétants entre scènes (> 1.5s) : {gaps}")
        result["qa_interne"] = {
            "controle_synchro": ControleSynchro().check(
                {"segments": {"captions": captions, "caption_words": [], "sentences": sentences}, "text": ""}
            ).score,
            "controle_rendu": ControleRendu().check(
                {"outputs": {label: str(video_path)}, "audio_duration": audio_duration}
            ).score,
        }

    import shutil

    shutil.rmtree(tmp_dir, ignore_errors=True)
    score = 1.0 - 0.34 * len(result["issues"])
    result["score"] = round(max(score, 0.0), 2)
    result["verdict"] = "VALIDÉE" if result["score"] >= 0.7 else "À REVOIR"
    return result


def audit_dir(target: Path, frames_count: int = 8) -> dict:
    videos = sorted(target.glob("final_*.mp4"))
    report = {
        "cible": str(target),
        "videos": [],
    }
    for video in videos:
        label = "9x16" if "9x16" in video.name else "16x9"
        report["videos"].append(audit_video(video, target, label, frames_count))
    ok_all = all(v["verdict"] == "VALIDÉE" for v in report["videos"]) and report["videos"]
    report["verdict_global"] = "VALIDÉ" if ok_all else "À REVOIR"
    return report


def main_audit(target: str, frames: int = 8) -> int:
    import datetime

    report = audit_dir(Path(target), frames)
    report["genere_le"] = datetime.datetime.now().isoformat(timespec="seconds")
    out = Path(target) / "qc_report_demo.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n📋 Rapport → {out}")
    return 0 if report["verdict_global"] == "VALIDÉ" else 1

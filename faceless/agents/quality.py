from __future__ import annotations

import re
from pathlib import Path

from PIL import Image
import numpy as np

from .base import QualityAgent, QAResult
from ..config import VERTICAL, HORIZONTAL

CLICHES = [
    "un jour sombre et orageux",
    "il était une fois",
    "je ne pouvais pas croire mes yeux",
    "la réalité m'a frappé de plein fouet",
]


class ControleHistoire(QualityAgent):
    name = "QualitéHistoire"
    role = "vérifie hook, longueur, clichés, chute"
    threshold = 0.7

    def check(self, context: dict) -> QAResult:
        text = context["text"]
        sentences = re.split(r"(?<=[.!?…])\s+", text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        tokens = re.findall(r"\S+", text)
        word_count = len(tokens)
        issues = []
        score = 1.0

        hook = sentences[0] if sentences else ""
        hook_words = len(re.findall(r"\S+", hook))
        if hook_words > 16:
            issues.append(f"hook un peu long ({hook_words} mots, viser ≤ 16)")
            score -= 0.15
        if not hook.rstrip().endswith((".", "!", "?", "…")):
            issues.append("hook sans ponctuation finale")
            score -= 0.05

        if len(sentences) < 5:
            issues.append(f"trop court ({len(sentences)} phrases, viser ≥ 5)")
            score -= 0.25
        if word_count < 90:
            issues.append(f"histoire courte pour une vidéo ({word_count} mots, viser ≥ 120)")
            score -= 0.15
        if word_count > 230:
            issues.append(f"histoire longue pour un short ({word_count} mots, viser ≤ 220)")
            score -= 0.1

        lowered = text.lower()
        for cliche in CLICHES:
            if cliche in lowered:
                issues.append(f"cliché détecté : « {cliche} »")
                score -= 0.2

        scoring_weight = max(score, 0.0)
        details = {
            "phrases": len(sentences),
            "mots": word_count,
            "hook_mots": hook_words,
            "hook": hook,
        }
        return QAResult(ok=scoring_weight >= self.threshold, score=scoring_weight, issues=issues, details=details)


class ControleVoix(QualityAgent):
    name = "QualitéVoix"
    role = "vérifie débit, durée et cohérence audio"
    threshold = 0.6

    def check(self, context: dict) -> QAResult:
        issues = []
        audio_duration = context.get("audio_duration") or 0.0
        words = max(context.get("word_count", 1), 1)
        score = 1.0

        duration = context["segments"]["sentences"][-1][1] if context["segments"]["sentences"] else 0.0
        voice_path = Path(context["voice_path"])
        if not voice_path.exists() or voice_path.stat().st_size < 1024:
            issues.append("fichier audio absent ou trop petit")
            return QAResult(ok=False, score=0.0, issues=issues)

        wpm = words / max(duration, 1.0) * 60.0
        if not (110 <= wpm <= 210):
            issues.append(f"débit inhabituel ({wpm:.0f} mots/min, attendu 110-210)")
            score -= 0.3
        if duration < 12:
            issues.append(f"vidéo vraiment courte ({duration:.1f}s)")
            score -= 0.15
        trailing = audio_duration - duration if audio_duration else 0.0
        if trailing > 3.0:
            issues.append(f"{trailing:.1f}s d'audio sans sous-titres en fin de vidéo")
            score -= 0.1
        return QAResult(ok=score >= self.threshold, score=max(score, 0.0), issues=issues, details={"duree": duration, "wpm": wpm})


class ControleVisuels(QualityAgent):
    name = "QualitéVisuels"
    role = "vérifie luminosité, contraste, variété des scènes"
    threshold = 0.7

    def check(self, context: dict) -> QAResult:
        issues = []
        images = context.get("images", {})
        if not images:
            return QAResult(ok=False, score=0.0, issues=["aucune image"])
        means = []
        flat = 0
        total = 0
        for label, paths in images.items():
            for p in paths:
                total += 1
                arr = np.asarray(Image.open(p).convert("L"), dtype=np.float32)
                mean = float(arr.mean())
                means.append(mean)
                if mean < 6 or mean > 240:
                    issues.append(f"{Path(p).name} : image quasi noire/blanche (moy={mean:.0f})")
                    flat += 1
                elif float(arr.std()) < 6:
                    issues.append(f"{Path(p).name} : image trop plate (contraste faible)")
                    flat += 1
        ratio_ok = 1 - (flat / max(total, 1))
        score = ratio_ok * 0.8 + min(float(np.std(means)) / 15.0, 1.0) * 0.2
        details = {"scenes": total, "moyenne_luminosite": float(np.mean(means))}
        return QAResult(ok=score >= self.threshold, score=score, issues=issues, details=details)


class ControleSynchro(QualityAgent):
    name = "QualitéSynchro"
    role = "vérifie cohérence texte ↔ sous-titres ↔ audio"
    threshold = 0.7

    def check(self, context: dict) -> QAResult:
        issues = []
        segments = context["segments"]
        words = segments.get("caption_words") or []
        text = context["text"]
        text_tokens = [t.strip(".,!?…;:") for t in re.findall(r"\S+", text) if t.strip(".,!?…;:")]

        score = 1.0
        if words:
            if abs(len(words) - len(text_tokens)) > max(2, len(text_tokens) * 0.25):
                issues.append(f"désalignement sous-titres/texte ({len(words)} vs {len(text_tokens)} mots)")
                score -= 0.4
            timings_ok = all(words[i][0] >= words[i - 1][0] - 0.05 for i in range(1, len(words)))
            if not timings_ok:
                issues.append("timings non monotones")
                score -= 0.3
        else:
            issues.append("aucun timing mot à mot (sous-titres estimates)")
            score -= 0.3

        sentences = segments["sentences"]
        if sentences:
            starts = [s[0] for s in sentences]
            if not all(starts[i] >= starts[i - 1] for i in range(1, len(starts))):
                issues.append("scènes non monotones")
                score -= 0.3
        return QAResult(ok=score >= self.threshold, score=max(score, 0.0), issues=issues)


class ControleRendu(QualityAgent):
    name = "QualitéRendu"
    role = "vérifie durées, flux, résolution et poids des MP4"
    threshold = 0.7

    def check(self, context: dict) -> QAResult:
        from moviepy import VideoFileClip

        issues = []
        outputs = context.get("outputs", {})
        if not outputs:
            return QAResult(ok=False, score=0.0, issues=["aucun fichier rendu"])
        audio_dur = context.get("audio_duration") or 0.0
        score = 1.0
        for label, path in outputs.items():
            p = Path(path)
            if not p.exists() or p.stat().st_size < 200_000:
                issues.append(f"{p.name} : fichier manquant ou trop léger")
                score -= 0.5
                continue
            with VideoFileClip(str(p)) as clip:
                duration = float(clip.duration)
                fps = clip.fps
                w, h = clip.size
            expected = VERTICAL if label == "9x16" else HORIZONTAL
            if (w, h) != expected:
                issues.append(f"{p.name} : résolution {w}x{h} au lieu de {expected[0]}x{expected[1]}")
                score -= 0.2
            if fps < 23:
                issues.append(f"{p.name} : fps {fps} < 24")
                score -= 0.1
            if audio_dur and abs(duration - audio_dur) > 2.0:
                issues.append(f"{p.name} : durée {duration:.1f}s ≠ audio {audio_dur:.1f}s")
                score -= 0.2
        return QAResult(ok=score >= self.threshold, score=max(score, 0.0), issues=issues)

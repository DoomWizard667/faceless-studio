from __future__ import annotations

import re
from pathlib import Path

from ..config import VERTICAL, HORIZONTAL
from ..images import fit_image, render_placeholder
from ..render import render_video
from ..stories import generate_story
from ..tts import synthesize
from .base import Agent, AgentResult


class Scripteur(Agent):
    name = "Scripteur"
    role = "écrit l'histoire (hook, tension, chute)"

    def run(self, context: dict) -> AgentResult:
        seed = context.get("attempt", 0) + (context.get("seed") or 0)
        story = generate_story(
            context["topic"],
            genre=context.get("genre", "horreur"),
            use_llm=context.get("use_llm", False),
            seed=seed,
        )
        words = len(re.findall(r"\S+", story.text))
        sentences = [s.strip() for s in re.split(r"(?<=[.!?…])\s+", story.text) if s.strip()]
        return AgentResult(
            ok=True,
            artifacts={"story": story, "text": story.text, "word_count": words, "sentences": sentences},
            notes=[f"{len(sentences)} phrases, {words} mots"],
        )


class VoixOff(Agent):
    name = "VoixOff"
    role = "génère la voix off et les timings mot à mot"

    def run(self, context: dict) -> AgentResult:
        out_dir = Path(context["out_dir"])
        voice_path = Path(context.get("voice_path") or out_dir / "voice.mp3")
        text = context["text"]
        if voice_path.exists() and not context.get("force_tts"):
            from moviepy import AudioFileClip

            with AudioFileClip(str(voice_path)) as clip:
                duration = float(clip.duration)
            from ..tts import _estimate_segments

            segments = _estimate_segments(voice_path, text)
            note = f"voix existante réutilisée ({duration:.1f}s), timings estimés"
        else:
            segments = synthesize(text, voice_path, voice=context.get("voice_name"))
            duration = segments["sentences"][-1][1] if segments["sentences"] else 0.0
            note = f"voix générée ({duration:.1f}s), timings Edge-TTS"
        audio_duration = context.get("audio_duration") or duration
        return AgentResult(
            ok=True,
            artifacts={
                "voice_path": str(voice_path),
                "segments": segments,
                "audio_duration": audio_duration,
            },
            notes=[note],
        )


class Artiste(Agent):
    name = "Artiste"
    role = "crée les visuels de chaque scène (placeholders ou images fournies)"

    def run(self, context: dict) -> AgentResult:
        out_dir = Path(context["out_dir"])
        style = context.get("style", "dark_cinematic")
        sentences = context["segments"]["sentences"]
        images = {}
        label_map = {"v": "9x16", "h": "16x9"}
        labels = [label_map[c] for c in context["formats"] if c in label_map] or ["9x16"]
        for label in labels:
            target = VERTICAL if label == "9x16" else HORIZONTAL
            images_dir = Path("/nonexistent")
            custom = context.get("images_dir")
            if custom:
                images_dir = Path(custom)
            files = sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.jpg")) if images_dir.exists() else []
            scene_dir = out_dir / f"images_{label}"
            scene_dir.mkdir(parents=True, exist_ok=True)
            paths = []
            if files:
                paths = [fit_image(p, target, scene_dir / p.name) for p in files]
            else:
                for i, (_, _, sentence) in enumerate(sentences):
                    paths.append(
                        render_placeholder(
                            sentence, i + 1, len(sentences), target,
                            scene_dir / f"scene_{i + 1:02d}.png", style=style,
                        )
                    )
            images[label] = [str(p) for p in paths]
        return AgentResult(ok=True, artifacts={"images": images}, notes=[f"{len(sentences)} scènes × {len(images)} format(s)"])


class Monteur(Agent):
    name = "Monteur"
    role = "assemble la vidéo finale (Ken Burns, fondus, karaoké)"

    def run(self, context: dict) -> AgentResult:
        out_dir = Path(context["out_dir"])
        style = context.get("style", "dark_cinematic")
        outputs = {}
        for label, paths in context["images"].items():
            size = VERTICAL if label == "9x16" else HORIZONTAL
            output = out_dir / f"final_{label}.mp4"
            render_video(
                context["voice_path"],
                context["segments"],
                paths,
                size,
                output,
                captions_on=not context.get("no_captions"),
                style=style,
            )
            outputs[label] = str(output)
        return AgentResult(ok=True, artifacts={"outputs": outputs}, notes=[", ".join(f"{k} → {v}" for k, v in outputs.items())])

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from .config import (
    DEFAULT_STYLE,
    HORIZONTAL,
    MJ_STYLE_PRESETS,
    VERTICAL,
    load_env,
)
from .images import fit_image, render_placeholder
from .render import render_video
from .stories import GENRES, generate_story
from .tts import synthesize

TOPIC_SEEDS = [
    "Mon voisin laissait sa lumière allumée tous les soirs à 3h12. Un soir, j'ai compris pourquoi.",
    "L'agenda que j'ai trouvé dans la maison de ma grand-mère expliquait toutes ses absences.",
    "Je suis chauffeur de nuit. Ce que sécurité m'a montré sur la caméra m'a fait démissionner.",
    "Mon patron a viré tout le monde par mail. Il ignorait que j'avais un accès admin.",
    "On m'a offert une maison en héritage. Le notaire a refusé d'entrer avec moi.",
    "L'ascenseur de mon immeuble s'arrête au sous-sol. Il n'y a pas de sous-sol.",
    "J'ai acheté un téléphone d'occasion. L'ancien propriétaire m'envoie encore des messages.",
    "La baby-sitter a filmé la chambre de mon enfant. Je n'aurais jamais dû regarder.",
    "Ce colis est arrivé sans expéditeur. Il contenait des photos de mon salon.",
    "Le restaurant n'avait jamais de clients. Un soir, j'ai demandé pourquoi.",
]


def _slug(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9àâäéèêëîïôöùûüç’' ]+", " ", text)
    text = text.replace("’", " ").replace("'", " ")
    words = re.findall(r"[a-z0-9àâäéèêëîïôöùûüç]+", text)
    return "-".join(words[:6]) or "histoire"


def _sizes(formats: str) -> list:
    sizes = []
    if "v" in formats:
        sizes.append(("9x16", VERTICAL))
    if "h" in formats:
        sizes.append(("16x9", HORIZONTAL))
    return sizes or [("9x16", VERTICAL)]


def write_srt(segments: list, path: Path) -> Path:
    def stamp(t: float) -> str:
        hours = int(t // 3600)
        minutes = int(t % 3600 // 60)
        seconds = int(t % 60)
        millis = int((t - int(t)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"

    lines = []
    for i, (start, end, text) in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{stamp(start)} --> {stamp(end)}")
        lines.append(text)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def save_story(story, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "story.txt").write_text(story.text, encoding="utf-8")
    (out_dir / "story.json").write_text(story.to_json(), encoding="utf-8")
    prompts = "\n".join(story.mj_prompts())
    (out_dir / "midjourney_prompts.txt").write_text(prompts, encoding="utf-8")
    return out_dir


def _prepare_images(images_dir: Path, style_dir: Path, sentences, size, style: str) -> list:
    custom = sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.jpg")) if images_dir.exists() else []
    if custom:
        return [fit_image(p, size, style_dir / p.name) for p in custom]
    paths = []
    for i, (_, _, sentence) in enumerate(sentences):
        paths.append(
            render_placeholder(
                sentence, i + 1, len(sentences), size, style_dir / f"scene_{i + 1:02d}.png", style=style
            )
        )
    return paths


def cmd_ideas(args):
    for i, topic in enumerate(TOPIC_SEEDS[: args.count], start=1):
        print(f"{i:02d}. {topic}")


def cmd_story(args):
    story = generate_story(args.topic, genre=args.genre, use_llm=args.llm, seed=args.seed)
    out_dir = save_story(story, Path(args.outdir) / _slug(args.topic))
    print(f"Histoire enregistrée dans {out_dir}")
    print("\n--- Histoire ---\n")
    print(story.text)
    print("\n--- Prompts Midjourney ---")
    for prompt in story.mj_prompts():
        print(prompt)


def _write_mj_prompts(sentences, style: str, out_dir: Path) -> None:
    base = MJ_STYLE_PRESETS.get(style, MJ_STYLE_PRESETS["dark_cinematic"])
    lines = []
    for i, entry in enumerate(sentences):
        sentence = entry[-1] if isinstance(entry, tuple) else entry
        excerpt = sentence.strip().strip(".")
        if len(excerpt) > 90:
            excerpt = excerpt[:90].rsplit(" ", 1)[0]
        lines.append(f"{i + 1}. {base} -- {excerpt} --ar 9:16 --style raw --v 7")
    (out_dir / "midjourney_prompts.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def cmd_video(args):
    load_env()
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.file:
        text = Path(args.file).read_text(encoding="utf-8").strip()
    else:
        text = args.text
    voice_path = Path(args.voice) if args.voice else out_dir / "voice.mp3"
    if voice_path.exists() and not args.voice:
        voice_path = out_dir / "voice.mp3"
        print(f"Voix existante réutilisée : {voice_path}")
        segments = _estimate_from_audio(voice_path, text)
    elif voice_path.exists():
        print(f"Voix fournie : {voice_path}")
        segments = _estimate_from_audio(voice_path, text)
    else:
        print("Génération de la voix off...")
        segments = synthesize(text, voice_path, voice=args.voice_name)
    write_srt(segments["captions"] or segments["sentences"], out_dir / "captions.srt")
    write_srt(segments["sentences"], out_dir / "sentences.srt")
    _write_mj_prompts(segments["sentences"], args.style, out_dir)
    for label, size in _sizes(args.formats):
        target = VERTICAL if size[1] > size[0] else HORIZONTAL
        output = out_dir / f"final_{label}.mp4"
        print(f"Rendu {label} → {output}")
        render_video(
            voice_path,
            segments,
            _prepare_images(Path(args.images) if args.images else Path("/nonexistent"),
                            out_dir / f"images_{label}", segments["sentences"], target, args.style),
            target,
            output,
            captions_on=not args.no_captions,
            style=args.style,
            sfx_on=not args.no_sfx,
        )
    print(f"Terminé : {out_dir}")


def _estimate_from_audio(voice_path, text):
    from .tts import _estimate_segments

    return _estimate_segments(Path(voice_path), text)


def cmd_pipeline(args):
    load_env()
    story = generate_story(args.topic, genre=args.genre, use_llm=args.llm, seed=args.seed)
    out_dir = Path(args.outdir) / _slug(args.topic)
    save_story(story, out_dir)
    print(f"[1/4] Histoire prête → {out_dir / 'story.txt'}")
    voice_path = out_dir / "voice.mp3"
    print("[2/4] Voix off...")
    segments = synthesize(story.text, voice_path, voice=args.voice_name)
    write_srt(segments["captions"] or segments["sentences"], out_dir / "captions.srt")
    write_srt(segments["sentences"], out_dir / "sentences.srt")
    _write_mj_prompts(segments["sentences"], args.style, out_dir)
    print("[3/4] Images + rendu...")
    for label, size in _sizes(args.formats):
        target = VERTICAL if size[1] > size[0] else HORIZONTAL
        output = out_dir / f"final_{label}.mp4"
        print(f"  Rendu {label} → {output}")
        render_video(
            voice_path,
            segments,
            _prepare_images(
                Path(args.images) if args.images else Path("/nonexistent"),
                out_dir / f"images_{label}",
                segments["sentences"],
                target,
                args.style,
            ),
            target,
            output,
            captions_on=not args.no_captions,
            style=args.style,
            sfx_on=not args.no_sfx,
        )
    print(f"[4/4] Terminé : {out_dir}")


def cmd_agents(args):
    from .agents import Realisateur

    load_env()
    out_dir = Path(args.outdir) / _slug(args.topic)
    realisateur = Realisateur()
    print("🎬 Réalisateur : assemblement de l'équipe d'agents")
    print("   Créateurs : Scripteur · VoixOff · Artiste · Monteur")
    print("   Qualité  : Histoire · Voix · Visuels · Synchro · Rendu")
    ok = realisateur.produce(
        {
            "topic": args.topic,
            "genre": args.genre,
            "seed": args.seed,
            "attempt": 0,
            "use_llm": args.llm,
            "formats": args.formats,
            "style": args.style,
            "voice_name": args.voice_name,
            "images_dir": args.images,
            "no_captions": args.no_captions,
            "no_sfx": args.no_sfx,
        },
        out_dir,
    )
    if ok:
        print(f"\n✅ Production validée par le contrôle qualité : {out_dir}")
    else:
        print(f"\n❌ Production bloquée par le contrôle qualité — voir {out_dir / 'qc_report.json'}")
        raise SystemExit(1)


def cmd_qcvideo(args):
    from .videoqc import main_audit

    code = main_audit(args.dir, frames=args.frames)
    raise SystemExit(code)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="faceless",
        description="Studio faceless : histoires IA → vidéos TikTok/YouTube sans montrer votre visage.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ideas = sub.add_parser("ideas", help="Lister des idées de vidéos")
    ideas.add_argument("--count", type=int, default=10)
    ideas.set_defaults(func=cmd_ideas)

    story_p = sub.add_parser("story", help="Générer une histoire + prompts Midjourney")
    story_p.add_argument("--topic", required=True)
    story_p.add_argument("--genre", default="horreur", choices=GENRES)
    story_p.add_argument("--llm", action="store_true", help="Utiliser un LLM (clé API requise)")
    story_p.add_argument("--seed", type=int, default=None)
    story_p.add_argument("--outdir", default="out")
    story_p.set_defaults(func=cmd_story)

    video_p = sub.add_parser("video", help="Texte existant → vidéo (TTS + images + montage)")
    video_p.add_argument("--text", default=None)
    video_p.add_argument("--file", default=None, help="Fichier .txt du texte")
    video_p.add_argument("--voice", default=None, help="MP3 existant à réutiliser")
    video_p.add_argument("--voice-name", default=None, help="Voix Edge-TTS (ex: fr-FR-VivienneMultilingualNeural)")
    video_p.add_argument("--images", default=None, help="Dossier d'images (01.png, 02.png…) sinon placeholders")
    video_p.add_argument("--style", default=DEFAULT_STYLE)
    video_p.add_argument("--formats", default="v", help="v=9:16, h=16:9, vh=les deux")
    video_p.add_argument("--no-captions", action="store_true")
    video_p.add_argument("--no-sfx", action="store_true", help="Désactiver le design sonore")
    video_p.add_argument("--outdir", default="out")
    video_p.set_defaults(func=cmd_video)

    pipe = sub.add_parser("pipeline", help="De l'idée au MP4 : histoire + voix + images + montage")
    pipe.add_argument("--topic", required=True)
    pipe.add_argument("--genre", default="horreur", choices=GENRES)
    pipe.add_argument("--llm", action="store_true")
    pipe.add_argument("--seed", type=int, default=None)
    pipe.add_argument("--voice-name", default=None)
    pipe.add_argument("--images", default=None)
    pipe.add_argument("--style", default=DEFAULT_STYLE)
    pipe.add_argument("--formats", default="vh")
    pipe.add_argument("--no-captions", action="store_true")
    pipe.add_argument("--no-sfx", action="store_true", help="Désactiver le design sonore")
    pipe.add_argument("--outdir", default="out")
    pipe.set_defaults(func=cmd_pipeline)

    agents_p = sub.add_parser("agents", help="Production par équipe d'agents avec contrôles qualité")
    agents_p.add_argument("--topic", required=True)
    agents_p.add_argument("--genre", default="horreur", choices=GENRES)
    agents_p.add_argument("--llm", action="store_true")
    agents_p.add_argument("--seed", type=int, default=None)
    agents_p.add_argument("--voice-name", default=None)
    agents_p.add_argument("--images", default=None)
    agents_p.add_argument("--style", default=DEFAULT_STYLE)
    agents_p.add_argument("--formats", default="vh")
    agents_p.add_argument("--no-captions", action="store_true")
    agents_p.add_argument("--no-sfx", action="store_true", help="Désactiver le design sonore")
    agents_p.add_argument("--outdir", default="out")
    agents_p.set_defaults(func=cmd_agents)

    qc_p = sub.add_parser("qcvideo", help="Auditer la qualité de vidéos déjà rendues (frames, SRT, audio)")
    qc_p.add_argument("--dir", required=True, help="Dossier contenant les MP4 (ex: examples/demo)")
    qc_p.add_argument("--frames", type=int, default=8)
    qc_p.set_defaults(func=cmd_qcvideo)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FONT_PATH = PROJECT_ROOT / "assets" / "fonts" / "Montserrat-Bold.ttf"
ASSETS_DIR = PROJECT_ROOT / "assets"

VERTICAL = (1080, 1920)
HORIZONTAL = (1920, 1080)

DEFAULT_VOICE = "fr-FR-HenriNeural"
DEFAULT_STYLE = "dark_cinematic"

WORD_GROUP_SIZE = 4

MJ_STYLE_PRESETS = {
    "dark_cinematic": (
        "dark moody cinematic still, single warm light source, deep shadows, "
        "film grain, muted desaturated palette, mysterious atmosphere, "
        "35mm photography, shallow depth of field"
    ),
    "office_revanche": (
        "cinematic office scene at dusk, teal and orange grading, tension in the air, "
        "photorealistic, dramatic rim lighting, 35mm"
    ),
    "liminal_night": (
        "night photography, empty corridor, fluorescent flicker, liminal space, "
        "uncanny atmosphere, cinematic still"
    ),
}


def load_env() -> None:
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)

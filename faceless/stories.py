from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass

from .config import MJ_STYLE_PRESETS, env
from .llm import llm_chat

GENRES = ["horreur", "revanche", "mystere", "malaise"]

LLM_SYSTEM_PROMPT = (
    "Tu es un auteur d'histoires courtes virales pour TikTok et YouTube Shorts. "
    "Tu écris en français, à la première personne, ton crépusculaire et nerveux. "
    "Structure imposée : une phrase d'accroche choc immédiate, une tension croissante "
    "avec des détails concrets, une chute surprenante ou inquiétante en dernière phrase. "
    "Interdits : clichés, longues descriptions, morale finale. "
    "Phrases courtes. 140 à 200 mots. Réponds uniquement avec l'histoire, sans titre ni commentaire."
)

HOOK_TEMPLATES = [
    "Ce qui s'est passé {quand} m'a empêché de dormir pour toujours.",
    "Je n'aurais jamais dû {action}.",
    "Personne ne me croit quand je raconte ce que j'ai vu {quand}.",
    "Il y a une seule règle {lieu} : ne jamais {action}.",
    "Le pire n'est pas ce qui s'est passé {quand}. C'est ce qui s'est passé après.",
]

BEAT_TEMPLATES = {
    "horreur": [
        "Tout a commencé {quand}, {lieu}. Rien ne laissait présager ce qui allait suivre.",
        "Au début, j'ai cru à une erreur. Puis les détails se sont mis à s'accumuler.",
        "Chaque nuit, la même chose se reproduisait, à quelques minutes près.",
        "Le soir où j'ai décidé de vérifier par moi-même, la maison était silencieuse.",
        "J'ai trouvé quelque chose qui n'aurait jamais dû exister. Et ça m'avait laissé un mot.",
        "Depuis ce jour, la porte reste fermée. Et je ne pose plus jamais de questions.",
    ],
    "revanche": [
        "Mon patron m'a humilié devant toute l'équipe. Il pensait que je me tairais.",
        "Il ignorait une chose : j'étais là depuis le premier jour. Je connaissais tout.",
        "J'ai rassemblé les preuves une par une, sans rien lui montrer.",
        "Le jour de la réunion, j'ai attendu mon moment. Puis j'ai déposé le dossier sur la table.",
        "Le silence dans la salle durait encore. Et son visage disait déjà tout.",
        "Aujourd'hui, c'est lui qui regarde les offres d'emploi. Moi, je dors enfin tranquille.",
    ],
    "mystere": [
        "Ce colis est arrivé sans expéditeur. Destination : mon adresse exacte.",
        "À l'intérieur, une photo de ma maison. Prise depuis l'intérieur.",
        "J'ai vérifié les caméras. Rien. Aucune trace, aucun mouvement.",
        "La seconde enveloppe est arrivée une semaine plus tard. Sans un mot.",
        "Il n'y avait qu'une phrase à l'intérieur. Et elle datait d'hier.",
        "La police dit d'ignorer les lettres. Mais la troisième photo, c'était ma chambre.",
    ],
    "malaise": [
        "Le nouvel employé était irréprochable. Trop irréprochable, en fait.",
        "Il connaissait les habitudes de tout le monde. Même celles qu'on ne raconte pas.",
        "Un jour, il a cité mes mots exacts d'une conversation privée.",
        "J'ai commencé à garder mes distances. Il a remarqué. Comme toujours.",
        "Son bureau a été vidé un matin. Le badge de la porte disait qu'il n'avait jamais existé.",
        "Mais hier soir, quelqu'un a glissé un mot dans ma boîte. Une signature que je connais trop bien.",
    ],
}

ENDINGS = [
    "Et je n'ai plus jamais reparlé de cette histoire. Jusqu'à aujourd'hui.",
    "Ce que j'ai compris ensuite a tout changé. Tout.",
]

WHEN = [
    "il y a exactement un mois",
    "l'automne dernier",
    "la semaine de mon anniversaire",
    "la nuit du 13",
    "le jour où j'ai emménagé",
]
PLACE = [
    "dans mon immeuble",
    "au fond de la campagne",
    "à trois rues de chez moi",
    "dans la maison familiale",
    "au sous-sol du bureau",
]
ACTION = [
    "ouvrir cette porte",
    "regarder sous le lit",
    "répondre à ce message",
    "descendre au sous-sol",
    "débrancher cette caméra",
]


@dataclass
class Story:
    topic: str
    genre: str
    title: str
    sentences: list

    @property
    def hook(self) -> str:
        return self.sentences[0] if self.sentences else ""

    @property
    def text(self) -> str:
        return " ".join(self.sentences)

    def mj_prompts(self, style: str = "dark_cinematic") -> list:
        base = MJ_STYLE_PRESETS.get(style, MJ_STYLE_PRESETS["dark_cinematic"])
        prompts = []
        for i, sentence in enumerate(self.sentences):
            excerpt = sentence.strip().strip(".")
            if len(excerpt) > 90:
                excerpt = excerpt[:90].rsplit(" ", 1)[0]
            prompts.append(f"{i + 1}. {base} -- {excerpt} --ar 9:16 --style raw --v 7")
        return prompts

    def to_json(self) -> str:
        return json.dumps(
            {
                "topic": self.topic,
                "genre": self.genre,
                "title": self.title,
                "sentences": self.sentences,
                "text": self.text,
                "midjourney_prompts": self.mj_prompts(),
            },
            ensure_ascii=False,
            indent=2,
        )


def _sentences_from_text(text: str) -> list:
    parts = re.split(r"(?<=[.!?…])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _offline_story(topic: str, genre: str, rng: random.Random) -> Story:
    when = rng.choice(WHEN)
    place = rng.choice(PLACE)
    action = rng.choice(ACTION)
    hook = rng.choice(HOOK_TEMPLATES).format(quand=when, action=action, lieu=place)
    beats = list(BEAT_TEMPLATES[genre])
    rng.shuffle(beats)
    mid = [b.format(quand=when, lieu=place) for b in beats[:4]]
    ending = rng.choice(ENDINGS)
    topic_label = topic if len(topic) <= 60 else topic[:57] + "..."
    sentences = [hook] + mid + [ending]
    return Story(topic=topic, genre=genre, title=topic_label, sentences=sentences)


def generate_story(topic: str, genre: str = "horreur", use_llm: bool = False, seed: int | None = None) -> Story:
    genre = genre if genre in GENRES else "horreur"
    rng = random.Random(seed)
    api_key = env("LLM_API_KEY")
    if use_llm and api_key:
        response = llm_chat(
            [
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": f"Sujet de l'histoire : {topic}"},
            ],
            base_url=env("LLM_BASE_URL", "https://api.mistral.ai/v1"),
            api_key=api_key,
            model=env("LLM_MODEL", "mistral-small-latest"),
        )
        sentences = _sentences_from_text(response)
        if len(sentences) >= 3:
            return Story(topic=topic, genre=genre, title=topic[:60], sentences=sentences)
    return _offline_story(topic, genre, rng)

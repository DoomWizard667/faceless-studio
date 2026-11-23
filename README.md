# Faceless Studio

Pipeline complet pour créer une chaîne TikTok / YouTube Shorts "faceless" d'histoires
vampirisantes, sans montrer votre visage ni utiliser votre voix.

De l'idée au MP4 : **histoire → voix off → images → sous-titres → vidéo**.

## Installation

```bash
pip install -r requirements.txt
```

Tout fonctionne **gratuitement** (voix Edge-TTS, images générées localement, ffmpeg inclus via pip).
Options payantes optionnelles : voir `.env.example`.

## Utilisation

### 1. Trouver des idées

```bash
python -m faceless.cli ideas --count 10
```

### 2. Générer une histoire + prompts Midjourney

```bash
python -m faceless.cli story --topic "Le voisin qui allumait sa lumière à 3h12" --genre horreur
```

Genres : `horreur`, `revanche`, `mystere`, `malaise`.
Avec `.env` configuré (`LLM_API_KEY`), ajoutez `--llm` pour des histoires uniques via API.

### 3. Pipeline complet (histoire → 2 MP4)

```bash
python -m faceless.cli pipeline --topic "L'agenda de ma grand-mère" --formats vh
```

Sorties dans `out/<slug>/` :

| Fichier | Description |
|---|---|
| `story.txt` / `story.json` | Histoire (texte brut + données structurées) |
| `midjourney_prompts.txt` | Un prompt par scène, style cohérent, `--ar 9:16` |
| `voice.mp3` | Voix off |
| `captions.srt` / `sentences.srt` | Sous-titres importables ailleurs |
| `images_9x16/` `images_16x9/` | Visuels de chaque scène |
| `final_9x16.mp4` | TikTok / Shorts / Reels |
| `final_16x9.mp4` | YouTube classique |

### Utiliser vos propres images Midjourney

1. Générez l'histoire (`story`) puis les images dans Midjourney avec `midjourney_prompts.txt`
2. Placez-les dans un dossier nommées `01.png`, `02.png`…
3. Lancez :

```bash
python -m faceless.cli video --file out/<slug>/story.txt --images mon_dossier --formats vh
```

Les images sont automatiquement recadrées (cover) aux deux formats.

## Rendu visuel inclus

Chaque vidéo est composée automatiquement : décors procéduraux (dégradés cinéma, halos lumineux,
silhouettes, portes, couloirs, skyline, bokeh, grain film, vignette), effet Ken Burns avec pan
latéral alterné, fondus enchaînés entre scènes, sous-titres karaoké avec mot actif surligné
(style CapCut) et barre de progression.

## Personnalisation

- **Voix** : `--voice-name fr-FR-VivienneMultilingualNeural` (liste complète : `edge-tts --list-voices`), ou ElevenLabs via `.env`
- **Style visuel** : `--style dark_cinematic | office_revanche | liminal_night | ember`
- **Sous-titres** : désactivables avec `--no-captions`
- **Steps suivants conseillés** : remplacer les placeholders par des images Midjourney (voir ci-dessus) pour un rendu pro

## Workflow recommandé (chaîne)

1. 3-5 vidéos/semaine, hooks variés, testez 2-3 voix avant de trancher
2. Publiez le même MP4 9:16 sur TikTok ET YouTube Shorts (double audience)
3. Utilisez le MP4 16:9 pour compilations YouTube longues (= monétisation)
4. Ne changez pas de style visuel : la reconnaissance est votre marque

## Avertissements

- Voix Edge-TTS : usage personnel OK, vérifiez les conditions pour un usage commercial massif
- Contenu 100 % IA sans retouche : ajoutez toujours une relecture et une retouche personnelle
- Les plateformes exigent de déclarer le contenu généré (synthetic media) lors de la publication

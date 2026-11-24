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

### 4. Mode équipe d'agents (recommandé)

```bash
python -m faceless.cli agents --topic "L'agenda de ma grand-mère" --formats vh
```

Chaque vidéo est produite par une équipe spécialisée, supervisée par un `Réalisateur` :

| Agent | Rôle |
|---|---|
| **Scripteur** | écrit l'histoire (hook, tension, chute) — régénérée automatiquement si refusée |
| **VoixOff** | voix off + timings mot à mot |
| **Artiste** | visuels de chaque scène (placeholders procéduraux ou vos images Midjourney) |
| **Monteur** | assemblage final (Ken Burns, fondus, karaoké, barre de progression) |

| Contrôle qualité | Sécurité vérifiée |
|---|---|
| **QualitéHistoire** | hook ≤ 16 mots, ≥ 120 mots, pas de clichés, structure |
| **QualitéVoix** | débit 110-210 mots/min, durée cohérente |
| **QualitéVisuels** | aucune scène vide/invisible, contraste suffisant |
| **QualitéSynchro** | timings monotones, texte ↔ sous-titres alignés |
| **QualitéRendu** | durée = audio, résolution exacte, fps 24, poids minimal |

Un échec déclenche une reprise (l'histoire est régénérée avec une nouvelle graine, jusqu'à 3 essais).
Le rapport complet est écrit dans `out/<slug>/qc_report.json`.

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

## Rendu visuel et sonore inclus

Chaque vidéo est composée automatiquement par un **moteur de décors procéduraux avancé** :
scènes en perspective (sols avec lignes de fuite, murs à moulures, couloirs à portes latérales,
escaliers avec rambarde, maisons à fumée de cheminée, skylines sur 3 plans de profondeur),
**atmosphère volumétrique** (brouillard bruité multi-bandes, rayons de lumière, halos, voiles
de brume au sol), **étalonnage cinéma** (split-toning ombres/lumière, courbe en S, bloom sur les
hautes lumières, aberration chromatique, grain, vignette), **motifs choisis selon le contenu de
chaque phrase** (porte, escaliers, fenêtre, maison, téléphone, couloir, skyline, yeux,
silhouettes…) et **personnages construits** : silhouettes à épaules, bras et jambes avec double
rim light d'intensité contrastée, ombre portée au sol, paires face-à-face, yeux détaillés
(iris, reflet, obscurité ambiante), smartphone avec bulles de messages et pouce — le karaoké
reste le seul bloc de texte à l'écran.

Du mouvement partout pour accrocher l'œil : effet Ken Burns renforcé (zoom 1,24×, pan latéral
alterné **combiné à une dérive verticale sinusoïdale** et une respiration d'échelle),
**transitions variées** (fondu, zoom punch, slide), double couche de **particules flottantes**
(fines + gros points flous), punch d'échelle d'ouverture, barre de progression, et sous-titres
karaoké avec mot actif surligné, **pop d'échelle** et flottement vertical (style CapCut).

Le tout avec un **design sonore synthétisé** : ambiance sombre continue, whoosh à chaque
transition de scène, boom d'impact à l'ouverture et à la chute (désactivable avec `--no-sfx`),
et une **musique de fond libre de droits (CC0)** sélectionnée automatiquement selon le sujet
et mixée sous la voix avec fondu entrée/sortie : deux pistes fournies dans `assets/music/`
(crédits dans [`assets/music/CREDITS.md`](assets/music/CREDITS.md)) — déposez-y vos propres
fichiers .mp3/.ogg/.wav pour les utiliser ; `--music none` désactive, `--music fichier.mp3`
impose une piste (choix automatique par défaut via le hash du sujet).

### 5. Auditer des vidéos déjà rendues

```bash
python -m faceless.cli qcvideo --dir examples/demo
```

Un auditeur analyse chaque MP4 : durée vs audio, résolution, fps, frames échantillonnées
(luminosité, présence des sous-titres aux instants où un mot est prononcé), cohérence des SRT
(monotonie, trous entre scènes). Le verdict et le rapport détaillé sont écrits dans
`qc_report_demo.json` à côté des vidéos.

## Personnalisation

- **Voix** : `--voice-name fr-FR-VivienneMultilingualNeural` (liste complète : `edge-tts --list-voices`), ou ElevenLabs via `.env`
- **Style visuel** : `--style dark_cinematic | office_revanche | liminal_night | ember`
- **Sous-titres** : désactivables avec `--no-captions`
- **Musique** : `--music auto | none | fichier.mp3` (pistes CC0 incluses dans `assets/music/`)
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

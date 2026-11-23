from __future__ import annotations

import json
from pathlib import Path

from .base import QualityAgent, Agent, AgentResult, QAResult
from .creators import Artiste, Monteur, Scripteur, VoixOff
from .quality import ControleHistoire, ControleRendu, ControleSynchro, ControleVoix, ControleVisuels


class Realisateur:
    """Orchestre l'équipe : chaque étape passe par ses contrôles qualité."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.history = []

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _stage(self, stage_name: str, agent: Agent, qas: list, context: dict, retry: int = 1) -> bool:
        attempts = retry + 1
        for attempt in range(1, attempts + 1):
            self._log(f"\n▶ [{stage_name}] {agent.name} — {agent.role} (tentative {attempt}/{attempts})")
            result = agent.run(context)
            if not result.notes:
                self._log("  … terminé")
            for note in result.notes:
                self._log(f"  {note}")
            context.update(result.artifacts)
            qa_results = []
            for qa in qas:
                qa_result = qa.check(context)
                qa_results.append(qa_result)
                status = "✅" if qa_result.ok else "❌"
                self._log(f"  {status} {qa.name} — score {qa_result.score:.2f}")
                for issue in qa_result.issues:
                    self._log(f"    ⚠ {issue}")
            all_ok = all(r.ok for r in qa_results)
            self.history.append(
                {
                    "stage": stage_name,
                    "agent": agent.name,
                    "attempt": attempt,
                    "ok": all_ok,
                    "qa": [{"name": qa.name, "ok": r.ok, "score": r.score, "issues": r.issues, "details": r.details} for qa, r in zip(qas, qa_results)],
                }
            )
            if all_ok:
                return True
            if result.notes:
                self._log("  … reprise demandée par le contrôle qualité")
        self._log(f"\n✖ [{stage_name}] échec après {attempts} tentatives")
        return False

    def produce(self, config: dict, out_dir: Path) -> bool:
        out_dir.mkdir(parents=True, exist_ok=True)
        context = dict(config)
        context["out_dir"] = str(out_dir)

        ok = self._stage(
            "Écriture",
            Scripteur(),
            [ControleHistoire()],
            context,
            retry=2,
        )
        if not ok:
            self._write_report(out_dir)
            return False
        story = context["story"]
        (out_dir / "story.txt").write_text(story.text, encoding="utf-8")
        (out_dir / "story.json").write_text(story.to_json(), encoding="utf-8")
        self._log(f"\n📝 Histoire retenue :\n   « {story.hook} »")

        ok = self._stage("Voix off", VoixOff(), [ControleVoix()], context)
        if not ok:
            self._write_report(out_dir)
            return False

        ok = self._stage("Visuels", Artiste(), [ControleVisuels()], context)
        if not ok:
            self._write_report(out_dir)
            return False

        ok = self._stage(
            "Montage",
            Monteur(),
            [ControleRendu(), ControleSynchro()],
            context,
        )
        self._write_report(out_dir)
        return ok

    def _write_report(self, out_dir: Path) -> None:
        report = {
            "agents": [
                {"nom": "Scripteur", "tache": "histoire"},
                {"nom": "VoixOff", "tache": "voix off + timings"},
                {"nom": "Artiste", "tache": "visuels de scènes"},
                {"nom": "Monteur", "tache": "assemblage MP4"},
                {"nom": "Contrôles qualité", "tache": "histoire, voix, visuels, synchro, rendu"},
            ],
            "etapes": self.history,
        }
        (out_dir / "qc_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._log(f"\n📋 Rapport qualité → {out_dir / 'qc_report.json'}")

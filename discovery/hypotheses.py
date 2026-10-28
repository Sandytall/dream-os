"""Structured hypothesis management (Phase 9D, section 7).

A hypothesis here is NOT a free-text explanation from a language model treated as a
finding. It is a falsifiable record about a recurring, measured prediction failure,
carrying every field section 7 requires: a precise statement, the motivating
observations, supporting and contradicting evidence (each pointing at recorded
numbers), an explicit predicted consequence, a discriminating experiment, the
observations needed to evaluate it, the expected cost, and -- once run -- the result.

Experiments carry, per hypothesis, the outcome each hypothesis predicts. Two
hypotheses are *distinguishable* iff some experiment predicts different outcomes for
them; if no available experiment separates a pair, they are marked indistinguishable
rather than one being declared the cause.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Experiment:
    name: str
    description: str
    observations_needed: list[str]
    predicted_outcome: dict[str, str]        # hypothesis_id -> expected outcome under it
    expected_cost: str
    status: str = "designed"                 # designed | run | invalid
    result: Optional[dict] = None


@dataclass
class Hypothesis:
    id: str
    statement: str
    motivating_observations: list[str]
    supporting_evidence: list[str]
    contradicting_evidence: list[str]
    predicted_consequence: str
    discriminating_experiments: list[str]    # Experiment.name values
    expected_cost: str
    status: str = "open"                     # open | supported | contradicted | indistinguishable | unresolved
    result: Optional[str] = None


@dataclass
class HypothesisSet:
    failure: str                             # the recurring failure under study
    evidence_source: str                     # report/file the evidence traces to
    hypotheses: list[Hypothesis] = field(default_factory=list)
    experiments: list[Experiment] = field(default_factory=list)

    # -- integrity --------------------------------------------------------- #
    def experiment(self, name: str) -> Experiment:
        for e in self.experiments:
            if e.name == name:
                return e
        raise KeyError(name)

    def validate(self) -> list[str]:
        """Return a list of problems; empty list == well-formed (Gate C requirement)."""
        problems = []
        names = {e.name for e in self.experiments}
        for h in self.hypotheses:
            if not h.statement.strip():
                problems.append(f"{h.id}: empty statement")
            if not h.motivating_observations:
                problems.append(f"{h.id}: no motivating observations")
            if not h.predicted_consequence.strip():
                problems.append(f"{h.id}: no predicted consequence")
            if not h.discriminating_experiments:
                problems.append(f"{h.id}: no discriminating experiment")
            for xn in h.discriminating_experiments:
                if xn not in names:
                    problems.append(f"{h.id}: references unknown experiment '{xn}'")
            if not (h.supporting_evidence or h.contradicting_evidence):
                problems.append(f"{h.id}: no documented evidence")
        for e in self.experiments:
            missing = [h.id for h in self.hypotheses if h.id not in e.predicted_outcome]
            if missing:
                problems.append(f"experiment '{e.name}': no predicted outcome for {missing}")
        return problems

    # -- distinguishability ------------------------------------------------ #
    def distinguishability(self) -> dict:
        """For each hypothesis pair, list experiments that predict different outcomes.
        A pair with an empty list is (currently) indistinguishable."""
        ids = [h.id for h in self.hypotheses]
        out = {}
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                sep = [e.name for e in self.experiments
                       if a in e.predicted_outcome and b in e.predicted_outcome
                       and e.predicted_outcome[a] != e.predicted_outcome[b]]
                out[f"{a}|{b}"] = sep
        return out

    def indistinguishable_pairs(self) -> list[str]:
        return [k for k, v in self.distinguishability().items() if not v]

    def to_dict(self) -> dict:
        return {
            "failure": self.failure,
            "evidence_source": self.evidence_source,
            "hypotheses": [asdict(h) for h in self.hypotheses],
            "experiments": [asdict(e) for e in self.experiments],
            "distinguishability": self.distinguishability(),
            "indistinguishable_pairs": self.indistinguishable_pairs(),
            "well_formed": not self.validate(),
            "validation_problems": self.validate(),
        }

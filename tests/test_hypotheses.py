"""Tests for the Phase 9D hypothesis-management component.

Run: env -u PYTHONPATH ./.venv/bin/python tests/test_hypotheses.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from discovery.hypotheses import Hypothesis, Experiment, HypothesisSet   # noqa: E402


def _toy():
    exp = Experiment(
        name="add_abs_velocity_feature",
        description="retrain with an absolute object-velocity input",
        observations_needed=["free-slide bias with vs without feature"],
        predicted_outcome={"H_friction": "bias unchanged", "H_velocity": "bias shrinks"},
        expected_cost="one retrain")
    hs = HypothesisSet(failure="under-travel", evidence_source="test")
    hs.experiments.append(exp)
    hs.hypotheses += [
        Hypothesis(id="H_friction", statement="learned friction too high",
                   motivating_observations=["under-travel"], supporting_evidence=["e1"],
                   contradicting_evidence=[], predicted_consequence="over-damped speed",
                   discriminating_experiments=["add_abs_velocity_feature"], expected_cost="cheap"),
        Hypothesis(id="H_velocity", statement="velocity under-represented",
                   motivating_observations=["velp is relative"], supporting_evidence=["e2"],
                   contradicting_evidence=[], predicted_consequence="bias shrinks with abs vel",
                   discriminating_experiments=["add_abs_velocity_feature"], expected_cost="cheap"),
    ]
    return hs


def test_wellformed_and_validation():
    hs = _toy()
    assert hs.validate() == [], hs.validate()
    assert hs.to_dict()["well_formed"] is True


def test_distinguishability():
    hs = _toy()
    d = hs.distinguishability()
    assert d["H_friction|H_velocity"] == ["add_abs_velocity_feature"]
    assert hs.indistinguishable_pairs() == []


def test_detects_malformed():
    hs = _toy()
    hs.hypotheses.append(Hypothesis(
        id="H_bad", statement="", motivating_observations=[], supporting_evidence=[],
        contradicting_evidence=[], predicted_consequence="",
        discriminating_experiments=["does_not_exist"], expected_cost="?"))
    probs = hs.validate()
    assert any("H_bad" in p for p in probs)
    assert hs.to_dict()["well_formed"] is False


def test_indistinguishable_flagged():
    hs = _toy()
    # an experiment that predicts the SAME outcome for both -> does not separate them
    hs.experiments[0].predicted_outcome = {"H_friction": "same", "H_velocity": "same"}
    assert hs.indistinguishable_pairs() == ["H_friction|H_velocity"]


if __name__ == "__main__":
    test_wellformed_and_validation()
    test_distinguishability()
    test_detects_malformed()
    test_indistinguishable_flagged()
    print("ok")

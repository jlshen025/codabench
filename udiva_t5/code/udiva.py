"""UDIVA-HHOI Track 5 (causal event grounding) — data loading.

Per effect: a textual description, five candidate-cause options (A-E), the effect
onset/offset timestamps, and (development set only) the ground-truth cause label set.
"""
import json
from dataclasses import dataclass, field

BASE = "<datasets>/UDIVA-HHOI/development/annotated_sessions"
CAUSAL_REF = f"{BASE}/starting_kit/causal/reference.json"


@dataclass
class Effect:
    effect_id: str
    session: str
    t_b: float
    t_e: float
    description: str
    options: dict          # {"A": text, ...}
    gt_labels: set         # {"C"} or {"A","B"} for multi-cause (empty for test)
    gt_windows: list = field(default_factory=list)


def load_causal(ref_path=CAUSAL_REF):
    """Development set with ground truth: effect_id -> Effect."""
    d = json.load(open(ref_path))["causal"]
    out = {}
    for sid, effs in d.items():
        for eid, e in effs.items():
            eff = e["effect"]
            causes = e.get("causes", [])
            out[eid] = Effect(
                effect_id=eid, session=sid,
                t_b=eff["t_b"], t_e=eff["t_e"], description=eff["description"],
                options=dict(e["options"]),
                gt_labels={c["option"] for c in causes},
                gt_windows=[(c["t_b"], c["t_e"]) for c in causes],
            )
    return out


def load_mcqs(path):
    """Test MCQ file {"SEGMENT": {"<sid>.mp4": {"<eid>": {"effect": ..., "options": ...}}}}:
    effect_id -> Effect (no ground truth)."""
    d = json.load(open(path))
    root = d.get("SEGMENT") or d.get("causal") or d
    out = {}
    for sk, effs in root.items():
        sid = sk[:-4] if sk.endswith(".mp4") else sk
        for eid, e in effs.items():
            eff = e["effect"]
            out[eid] = Effect(effect_id=eid, session=sid, t_b=eff["t_b"], t_e=eff["t_e"],
                              description=eff["description"], options=dict(e["options"]),
                              gt_labels=set())
    return out

"""Hybrid route-aware RM rewards for v5.64 routed VLM recheck.

This version keeps the v5.62 scope-heavy behavior, then adds binary guards only
where the route says the model is likely to over-trust a fake claim:
- REAL vs FULL_SYNTHETIC targeted disputes;
- REAL vs TAMPERED full rechecks;
- fake-scope conflicts where both reports may be fake-biased but GT is REAL.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Dict, List

from swift.rewards import ORM, orms


def _safe_file(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


PLUGIN_DIR = Path(__file__).resolve().parent
RELEASE_ROOT = Path(os.environ.get("SOFAKE_R1_RELEASE", PLUGIN_DIR.parent))
BASE_PLUGIN_CANDIDATES = [
    PLUGIN_DIR / "judge_v561_grpo_route_aware_rm_plugin.py",
    RELEASE_ROOT / "plugins/judge_v561_grpo_route_aware_rm_plugin.py",
]
BASE_PLUGIN = next((p for p in BASE_PLUGIN_CANDIDATES if _safe_file(p)), None)
if BASE_PLUGIN is None:
    raise RuntimeError(
        "Cannot find base reward plugin. Tried: "
        + ", ".join(str(p) for p in BASE_PLUGIN_CANDIDATES)
    )
spec = importlib.util.spec_from_file_location("judge_v561_base_for_v564", BASE_PLUGIN)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load base reward plugin: {BASE_PLUGIN}")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


LABELS = base.LABELS
ROUTES = base.ROUTES


def _route_detail(extra: Dict, sol: Dict) -> str:
    return str(extra.get("route_detail") or sol.get("route_detail") or "").lower()


class JudgeV564HybridRouteBinaryScopeRMDecisionReward(ORM):
    """Reward exact decisions with scope-heavy terms and route-local binary guards."""

    def __call__(self, completions, solution=None, extra_info=None, **kwargs) -> List[float]:
        rewards = []
        for idx, content in enumerate(completions):
            extra = base._get_dict_item(extra_info, idx)
            sol = base._load_dict(base._get_item(solution, idx))
            gt = base._truth(base._get_item(solution, idx), extra)
            parsed = base._parse_completion(content)
            pred = parsed["final_answer"]
            if gt not in LABELS or pred not in LABELS:
                rewards.append(-2.5)
                continue

            score = 0.0
            gt_auth = base._auth(gt)
            pred_auth = base._auth(pred)
            gt_scope = base._truth_scope(gt)
            pred_scope = base._truth_scope(pred)
            detail = _route_detail(extra, sol)
            is_scope_conflict = "fake_scope_conflict" in detail or "scope_conflict" in detail
            is_real_vs_tampered = "real_vs_tampered" in detail or "binary_conflict_real_vs_tampered" in detail
            is_real_vs_full = "real_vs_full_synthetic" in detail or "real_vs_full" in detail

            exact = pred == gt
            binary_ok = pred_auth == gt_auth

            consistent = (
                parsed["authenticity"] == pred_auth
                and parsed["fake_scope"] == pred_scope
                and ((pred == "TAMPERED") == (parsed["bbox"] is not None))
            )
            score += 0.08 if consistent else -0.25

            # Base correctness. Binary matters, but exact and scope still decide.
            score += 0.45 if binary_ok else -0.75
            score += 1.05 if exact else -0.20

            # Route-local binary guards. These are deliberately not global.
            if is_real_vs_full:
                if pred not in {"REAL", "FULL_SYNTHETIC"}:
                    score -= 0.60
                if gt == "REAL":
                    score += 0.45 if pred == "REAL" else -0.55
                elif gt == "FULL_SYNTHETIC":
                    score += 0.40 if pred == "FULL_SYNTHETIC" else -0.45
            if is_real_vs_tampered:
                if pred not in {"REAL", "TAMPERED"}:
                    score -= 0.35
                if exact:
                    score += 0.25
                elif gt == "REAL" and pred == "TAMPERED":
                    score -= 0.60
                elif gt == "TAMPERED" and pred == "REAL":
                    score -= 0.60

            # Scope-heavy objective from v5.62, with a REAL rescue path for
            # fake-scope conflicts where both reports may be wrong together.
            if gt in {"FULL_SYNTHETIC", "TAMPERED"}:
                if pred == gt:
                    score += 0.80
                elif pred_auth == gt_auth:
                    score -= 0.70
                else:
                    score -= 0.40
                if is_scope_conflict:
                    if exact:
                        score += 0.45
                    elif pred_auth == gt_auth:
                        score -= 0.65
                    else:
                        score -= 0.50
            else:
                if pred == "REAL":
                    score += 0.35 if is_scope_conflict else 0.15
                else:
                    score -= 0.45
                    if is_scope_conflict:
                        score -= 0.35

            route = base._route_expected(extra, sol)
            if route:
                score += 0.04 if parsed["route_action"] == route else -0.08

            sources = base._source_answers(extra, sol)
            if exact:
                # Correctly rejecting both reports is one of the actual DM skills.
                score += 0.22 if gt in sources else 0.65
            else:
                if gt in sources and pred in sources:
                    score -= 0.30
                elif gt in sources and pred not in sources:
                    score -= 0.45
                elif gt not in sources and pred in sources:
                    score -= 0.55
                else:
                    score -= 0.15

            pred_box = parsed["bbox"]
            if gt == "TAMPERED":
                if pred != "TAMPERED":
                    score -= 0.50
                    if pred == "FULL_SYNTHETIC":
                        score -= 0.35
                    if pred == "REAL":
                        score -= 0.35
                elif pred_box is None:
                    score -= 0.75
                else:
                    iou = base._mask_iou(pred_box, base._gt_mask(extra, sol))
                    if iou <= 0:
                        iou = base._box_iou(pred_box, base._gt_bbox(extra, sol))
                    score += 0.90 * iou
                    if iou >= 0.50:
                        score += 0.25
                    elif iou <= 0.05:
                        score -= 0.15
            else:
                if pred == "TAMPERED" or pred_box is not None:
                    score -= 0.40
                    if gt == "FULL_SYNTHETIC":
                        score -= 0.25

            provider = base._norm_label(extra.get("evidence_provider_answer") or extra.get("provider_answer"))
            if provider == "TAMPERED" and gt != "TAMPERED" and pred == "TAMPERED":
                score -= 0.35
            if gt == "FULL_SYNTHETIC" and pred == "TAMPERED":
                score -= 0.50
            if gt == "TAMPERED" and pred == "FULL_SYNTHETIC":
                score -= 0.65
            if gt in {"FULL_SYNTHETIC", "TAMPERED"} and pred == "REAL":
                score -= 0.25
            if gt == "REAL" and pred != "REAL":
                score -= 0.20

            rewards.append(float(max(-4.0, min(4.5, score))))
        return rewards


class JudgeV564HybridRouteBinaryScopeRMFormatReward(base.JudgeV561RouteAwareRMFormatReward):
    pass


orms["judge_v564_hybrid_route_binary_scope_rm_decision"] = JudgeV564HybridRouteBinaryScopeRMDecisionReward
orms["judge_v564_hybrid_route_binary_scope_rm_format"] = JudgeV564HybridRouteBinaryScopeRMFormatReward

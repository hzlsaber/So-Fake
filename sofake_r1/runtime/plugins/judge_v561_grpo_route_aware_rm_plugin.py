"""Route-aware RM rewards for v5.61 GRPO hard-case recheck.

The reward is intentionally source-neutral. It does not prefer observer or
provider by identity; it rewards the final answer, real/fake correctness,
fake-scope correctness, localization IoU for TAMPERED, and the arbitration
behavior implied by whether the ground truth appears among the two reports.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from PIL import Image

from swift.rewards import ORM, orms


LABELS = {"REAL", "FULL_SYNTHETIC", "TAMPERED"}
ROUTES = {"TARGETED_RECHECK", "FULL_RECHECK"}
SCOPES = {"N/A", "FULL_SYNTHETIC", "TAMPERED"}

LABEL_RE = re.compile(r"\b(REAL|FULL_SYNTHETIC|TAMPERED)\b", re.I)
BBOX_RE = re.compile(
    r"<\|box_start\|>\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)\s*,\s*"
    r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)\s*<\|box_end\|>",
    re.I,
)


def _load_dict(obj) -> Dict:
    if isinstance(obj, dict):
        return obj
    if not isinstance(obj, str):
        return {}
    try:
        parsed = json.loads(obj)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _get_item(obj, idx: int):
    if isinstance(obj, list):
        return obj[idx] if idx < len(obj) else None
    return obj


def _get_dict_item(obj, idx: int) -> Dict:
    return _load_dict(_get_item(obj, idx))


def _tag_text(text: str, tag: str) -> str:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text or "", re.I | re.S)
    return m.group(1).strip() if m else ""


def _field(text: str, name: str) -> Optional[str]:
    m = re.search(rf"^{re.escape(name)}:\s*([A-Z_/]+)\s*$", text or "", re.I | re.M)
    return m.group(1).upper() if m else None


def _norm_label(value) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = value.upper()
    return value if value in LABELS else None


def _parse_bbox(text: str) -> Optional[Tuple[int, int, int, int]]:
    m = BBOX_RE.search(text or "")
    if not m:
        return None
    x1, y1, x2, y2 = [int(v) for v in m.groups()]
    x1, x2 = sorted((max(0, min(224, x1)), max(0, min(224, x2))))
    y1, y2 = sorted((max(0, min(224, y1)), max(0, min(224, y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _parse_completion(text: str) -> Dict:
    text = text or ""
    auth_block = _tag_text(text, "authenticity_decision")
    scope_block = _tag_text(text, "scope_decision")
    route_block = _tag_text(text, "routing_decision")
    final_assessment = _tag_text(text, "final_assessment")
    final_answer_text = _tag_text(text, "final_answer")
    answer_match = LABEL_RE.search(final_answer_text)
    answer = answer_match.group(1).upper() if answer_match else None
    auth = _field(auth_block, "authenticity")
    scope = _field(scope_block, "fake_scope")
    route = _field(route_block, "route_action")
    return {
        "has_auth": bool(auth_block),
        "has_scope": bool(scope_block),
        "has_route": bool(route_block),
        "has_assessment": bool(final_assessment),
        "has_final": bool(final_answer_text),
        "authenticity": auth if auth in {"REAL", "FAKE"} else None,
        "fake_scope": scope if scope in SCOPES else None,
        "route_action": route if route in ROUTES else None,
        "final_answer": answer if answer in LABELS else None,
        "bbox": _parse_bbox(final_answer_text),
    }


def _truth(solution, extra: Dict) -> Optional[str]:
    data = _load_dict(solution)
    answer = data.get("final_answer") or data.get("label_name") or extra.get("gt_label_name")
    return _norm_label(answer)


def _truth_scope(gt: Optional[str]) -> Optional[str]:
    if gt == "REAL":
        return "N/A"
    if gt == "FULL_SYNTHETIC":
        return "FULL_SYNTHETIC"
    if gt == "TAMPERED":
        return "TAMPERED"
    return None


def _auth(label: Optional[str]) -> Optional[str]:
    if label == "REAL":
        return "REAL"
    if label in {"FULL_SYNTHETIC", "TAMPERED"}:
        return "FAKE"
    return None


def _box_from_obj(obj) -> Optional[Tuple[int, int, int, int]]:
    if isinstance(obj, str):
        return _parse_bbox(obj)
    if isinstance(obj, (list, tuple)) and len(obj) == 4:
        try:
            x1, y1, x2, y2 = [int(v) for v in obj]
        except Exception:
            return None
        x1, x2 = sorted((max(0, min(224, x1)), max(0, min(224, x2))))
        y1, y2 = sorted((max(0, min(224, y1)), max(0, min(224, y2))))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2
    return None


def _box_iou(a: Optional[Tuple[int, int, int, int]], b) -> float:
    b = _box_from_obj(b)
    if not a or not b:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@lru_cache(maxsize=8192)
def _mask_prefix(mask_path: str):
    if not mask_path or not os.path.exists(mask_path):
        return None
    try:
        img = Image.open(mask_path).convert("RGBA").resize((224, 224), Image.Resampling.NEAREST)
    except Exception:
        return None
    data = img.getdata()
    prefix = [[0] * 225 for _ in range(225)]
    idx = 0
    for y in range(224):
        row_sum = 0
        for x in range(224):
            r, g, b, _ = data[idx]
            idx += 1
            val = 1 if max(r, g, b) > 0 else 0
            row_sum += val
            prefix[y + 1][x + 1] = prefix[y][x + 1] + row_sum
    area = prefix[224][224]
    return prefix, area


def _rect_sum(prefix, box: Tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    return prefix[y2][x2] - prefix[y1][x2] - prefix[y2][x1] + prefix[y1][x1]


def _mask_iou(box: Optional[Tuple[int, int, int, int]], mask_path: Optional[str]) -> float:
    if not box or not mask_path:
        return 0.0
    cached = _mask_prefix(mask_path)
    if not cached:
        return 0.0
    prefix, mask_area = cached
    if mask_area <= 0:
        return 0.0
    box_area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    if box_area <= 0:
        return 0.0
    inter = _rect_sum(prefix, box)
    union = box_area + mask_area - inter
    return inter / union if union > 0 else 0.0


def _source_answers(extra: Dict, solution: Dict) -> set[str]:
    obs = _norm_label(extra.get("observer_answer") or solution.get("observer_answer"))
    prov = _norm_label(extra.get("evidence_provider_answer") or extra.get("provider_answer") or solution.get("evidence_provider_answer"))
    return {x for x in (obs, prov) if x in LABELS}


def _route_expected(extra: Dict, solution: Dict) -> Optional[str]:
    route = extra.get("route_action") or solution.get("route_action")
    return route if route in ROUTES else None


def _gt_bbox(extra: Dict, solution: Dict):
    return solution.get("gt_bbox") if "gt_bbox" in solution else extra.get("gt_bbox")


def _gt_mask(extra: Dict, solution: Dict):
    return solution.get("gt_mask_path") if solution.get("gt_mask_path") else extra.get("gt_mask_path")


class JudgeV561RouteAwareRMDecisionReward(ORM):
    """Route-aware final-answer reward with arbitration and IoU terms."""

    def __call__(self, completions, solution=None, extra_info=None, **kwargs) -> List[float]:
        rewards = []
        for idx, content in enumerate(completions):
            extra = _get_dict_item(extra_info, idx)
            sol = _load_dict(_get_item(solution, idx))
            gt = _truth(_get_item(solution, idx), extra)
            parsed = _parse_completion(content)
            pred = parsed["final_answer"]
            if gt not in LABELS or pred not in LABELS:
                rewards.append(-1.5)
                continue

            score = 0.0
            gt_auth = _auth(gt)
            pred_auth = _auth(pred)
            gt_scope = _truth_scope(gt)
            pred_scope = _truth_scope(pred)

            # Minimal parse and self-consistency.
            score += 0.05
            consistent = (
                parsed["authenticity"] == pred_auth
                and parsed["fake_scope"] == pred_scope
                and ((pred == "TAMPERED") == (parsed["bbox"] is not None))
            )
            score += 0.10 if consistent else -0.20

            # Main correctness terms.
            score += 0.70 if pred_auth == gt_auth else -0.50
            score += 0.80 if pred == gt else 0.0
            if gt in {"FULL_SYNTHETIC", "TAMPERED"}:
                if pred == gt:
                    score += 0.30
                elif pred_auth == gt_auth:
                    score += 0.00
                else:
                    score -= 0.20

            # Route field should stay compatible with the prompt.
            route = _route_expected(extra, sol)
            if route:
                score += 0.05 if parsed["route_action"] == route else -0.05

            # Arbitration: selecting the correct provided side is good; correctly
            # rejecting both report claims is deliberately more valuable.
            sources = _source_answers(extra, sol)
            if pred == gt:
                if gt in sources:
                    score += 0.25
                else:
                    score += 0.60
            else:
                if gt in sources and pred in sources:
                    score -= 0.25
                elif gt in sources and pred not in sources:
                    score -= 0.40
                elif gt not in sources and pred in sources:
                    score -= 0.35
                else:
                    score -= 0.10

            # Localization. IoU matters for TAMPERED; bbox is harmful elsewhere.
            pred_box = parsed["bbox"]
            if gt == "TAMPERED":
                if pred != "TAMPERED":
                    pass
                elif pred_box is None:
                    score -= 0.30
                else:
                    iou = _mask_iou(pred_box, _gt_mask(extra, sol))
                    if iou <= 0:
                        iou = _box_iou(pred_box, _gt_bbox(extra, sol))
                    score += 0.50 * iou
                    if iou >= 0.50:
                        score += 0.10
            else:
                if pred == "TAMPERED" or pred_box is not None:
                    score -= 0.20

            # Common failure penalties observed in OOD/external recheck.
            provider = _norm_label(extra.get("evidence_provider_answer") or extra.get("provider_answer"))
            if provider == "TAMPERED" and gt != "TAMPERED" and pred == "TAMPERED":
                score -= 0.25
            if gt == "FULL_SYNTHETIC" and pred == "TAMPERED":
                score -= 0.30
            if gt == "TAMPERED" and pred == "FULL_SYNTHETIC":
                score -= 0.30

            rewards.append(float(max(-2.0, min(3.5, score))))
        return rewards


class JudgeV561RouteAwareRMFormatReward(ORM):
    """Small format reward; semantic correctness lives in the RM reward."""

    def __call__(self, completions, extra_info=None, **kwargs) -> List[float]:
        rewards = []
        for content in completions:
            parsed = _parse_completion(content)
            score = 0.0
            score += 0.20 if parsed["has_auth"] else -0.10
            score += 0.20 if parsed["has_scope"] else -0.10
            score += 0.20 if parsed["has_route"] else -0.10
            score += 0.20 if parsed["has_assessment"] else -0.10
            score += 0.20 if parsed["has_final"] else -0.10
            score += 0.05 if parsed["authenticity"] in {"REAL", "FAKE"} else -0.05
            score += 0.05 if parsed["fake_scope"] in SCOPES else -0.05
            score += 0.05 if parsed["route_action"] in ROUTES else -0.05
            rewards.append(float(max(-1.0, min(1.0, score))))
        return rewards


orms["judge_v561_route_aware_rm_decision"] = JudgeV561RouteAwareRMDecisionReward
orms["judge_v561_route_aware_rm_format"] = JudgeV561RouteAwareRMFormatReward

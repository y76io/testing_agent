from typing import Dict, List, Tuple, Any


def _intersects(a: List[str], b: List[str]) -> bool:
    return bool(set(a) & set(b))


def compute_items_stats(items: List[Dict[str, Any]], requires_reference_field: str | None = None) -> Dict[str, Any]:
    count = len(items)
    all_have_ref = True
    if requires_reference_field:
        for it in items:
            if requires_reference_field not in it:
                all_have_ref = False
                break
    return {"count": count, "all_have_ref_field": all_have_ref if requires_reference_field else False}


def dataset_supports_metric(dataset_profile: Dict[str, Any], items_stats: Dict[str, Any], applicability: Dict[str, Any] | None) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    req = (applicability or {}).get("requires", {})
    dis = (applicability or {}).get("disallows", {})

    if req.get("modality") and not _intersects(dataset_profile.get("modalities", []), req["modality"]):
        reasons.append("modality mismatch")
    if req.get("languages_any") and not _intersects(dataset_profile.get("languages", []), req["languages_any"]):
        reasons.append("language mismatch")
    if req.get("dataset_tags_all"):
        tags = set(dataset_profile.get("tags", []))
        if not set(req["dataset_tags_all"]).issubset(tags):
            reasons.append("missing required dataset tags")
    if req.get("needs_ground_truth") and not dataset_profile.get("has_ground_truth", False):
        reasons.append("ground_truth required")
    ref_field = req.get("requires_reference_field")
    if ref_field and not items_stats.get("all_have_ref_field", False):
        reasons.append(f"dataset items missing field '{ref_field}'")
    if req.get("min_items") and items_stats.get("count", 0) < req["min_items"]:
        reasons.append(f"min_items {req['min_items']} not met")

    dis_any = set((dis.get("dataset_tags_any") or []))
    if dis_any & set(dataset_profile.get("tags", [])):
        reasons.append("dataset contains disallowed tags")

    return (len(reasons) == 0, reasons)


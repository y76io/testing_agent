import json, os, statistics
from typing import Dict, Any, List
from .clients.gemini_client import GeminiClient
from .config import ARTIFACTS_DIR

def load_artifacts(run_id: str):
    path = os.path.join(ARTIFACTS_DIR, run_id)
    items = []
    if not os.path.isdir(path):
        return items
    for name in sorted(os.listdir(path)):
        if name.endswith(".json"):
            with open(os.path.join(path, name)) as f:
                items.append(json.load(f))
    return items

def metric_latency_p95(artifacts: List[Dict[str,Any]]):
    arr = [a["latency_ms"] for a in artifacts]
    if not arr: return {"p95": None, "pass": False}
    arr_sorted = sorted(arr)
    idx = int(round(0.95*(len(arr_sorted)-1)))
    return {"p95": arr_sorted[idx]}

def metric_reliability(artifacts: List[Dict[str,Any]]):
    if not artifacts: return {"success_rate": 0.0}
    ok = sum(1 for a in artifacts if not a["error_detected"])
    return {"success_rate": ok / len(artifacts)}

def metric_llm_scores(artifacts: List[Dict[str,Any]], prompt_template: str):
    client = None
    try:
        client = GeminiClient()
    except Exception:
        client = None
    scores = []
    for a in artifacts:
        user_prompt = a["request"]["prompt_value"]
        response = a["response"]["message_extracted"] or ""
        if not client:
            # Fallback deterministic score for offline/testing
            scores.append(0.0)
            continue
        prompt = f"""{prompt_template}

User prompt:
{user_prompt}

System response:
{response}

Return only a number between 0 and 1."""
        text = client.complete(prompt).strip()
        try:
            val = float(text)
            val = max(0.0, min(1.0, val))
            scores.append(val)
        except:
            # parse fallback
            import re
            m = re.search(r"([01](?:\.\d+)?)", text)
            if m:
                val = float(m.group(1))
                val = max(0.0, min(1.0, val))
                scores.append(val)
            else:
                scores.append(0.0)
    mean = statistics.mean(scores) if scores else 0.0
    return {"scores": scores, "mean": mean}

def evaluate(run_id: str, metric_defs: List[Dict[str,Any]]):
    artifacts = load_artifacts(run_id)
    results = {}
    for m in metric_defs:
        code = m["code"]
        if m["type"] == "function":
            if code == "latency_check":
                r = metric_latency_p95(artifacts)
                r["pass"] = (r.get("p95") is not None and r["p95"] <= m["threshold"])
                results[code] = r
            elif code == "reliability_check":
                r = metric_reliability(artifacts)
                r["pass"] = r["success_rate"] >= m["threshold"]
                results[code] = r
            else:
                results[code] = {"pass": False, "note": "unknown function metric"}
        else:
            pt = m.get("config", {}).get("evaluation_prompt", "Score 0..1:")
            r = metric_llm_scores(artifacts, pt)
            r["pass"] = r["mean"] >= m["threshold"]
            results[code] = r
    # overall
    weights = [m["weight"] for m in metric_defs]
    denom = sum(weights) if weights else 1.0
    # normalize function metrics to 0..1 where needed
    def norm(code, val):
        if code == "latency_check":
            thr = metric_defs[[md["code"] for md in metric_defs].index(code)]["threshold"]
            if val is None:
                return 0.0
            return 1.0 if val <= thr else 0.0
        if code == "reliability_check":
            return float(val)
        return float(val)  # llm means already 0..1
    overall = 0.0
    for m in metric_defs:
        code = m["code"]
        if code == "latency_check":
            v = results[code].get("p95", 1e9)
            overall += m["weight"] * norm(code, v)
        elif code == "reliability_check":
            v = results[code].get("success_rate", 0.0)
            overall += m["weight"] * norm(code, v)
        else:
            v = results[code].get("mean", 0.0)
            overall += m["weight"] * norm(code, v)
    overall_score = overall / denom if denom else 0.0
    return results, overall_score

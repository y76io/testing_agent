import json, time, httpx, os
from typing import Dict, Any, List
from .utils.jsonpath_utils import jsonpath_get, jsonpath_set
from .config import ARTIFACTS_DIR

def build_request(headers, body, prompt_paths, prompt_text):
    body_copy = json.loads(json.dumps(body))
    placed = False
    for p in prompt_paths:
        if jsonpath_set(body_copy, p, prompt_text):
            placed = True
            break
    if not placed and isinstance(body_copy, dict):
        body_copy["input"] = prompt_text  # fallback
    return headers, body_copy

def execute_dataset(run_id: str, system: Dict[str,Any], mapping: Dict[str,Any], dataset: List[Dict[str,Any]]):
    out_dir = os.path.join(ARTIFACTS_DIR, run_id)
    os.makedirs(out_dir, exist_ok=True)
    client = httpx.Client(timeout=15.0)
    results = []
    for idx, item in enumerate(dataset, start=1):
        prompt = item.get("prompt","")
        headers, req_body = build_request(system["headers"], system["body"], mapping["prompt_paths"], prompt)
        t0 = time.time()
        error = False
        resp_body = None
        status = None
        try:
            if system["method"].upper() == "GET":
                r = client.get(system["endpoint"], headers=headers, params=req_body)
            else:
                r = client.post(system["endpoint"], headers=headers, json=req_body)
            status = r.status_code
            resp_body = r.json() if "application/json" in r.headers.get("content-type","") else {"text": r.text}
        except Exception as e:
            error = True
            resp_body = {"error": str(e)}
            status = 0
        latency_ms = int((time.time() - t0)*1000)

        # error rules
        rules = mapping["error_rules"]
        if not error:
            if any(lo <= status <= hi for lo,hi in rules.get("status_code_ranges", [])):
                error = True
            for path in rules.get("json_field_paths", []):
                if jsonpath_get(resp_body, path) is not None:
                    error = True
            import re
            body_text = json.dumps(resp_body).lower()
            for rgx in rules.get("regexes", []):
                if re.search(rgx, body_text):
                    error = True

        # extract message
        message = None
        if not error:
            for rp in mapping["response_paths"]:
                val = jsonpath_get(resp_body, rp)
                if val is not None:
                    message = val
                    break
            if message is None:
                # fallback common keys
                message = resp_body.get("output") or resp_body.get("message") or resp_body.get("text")

        artifact = {
            "idx": idx,
            "request": {
                "headers": headers,
                "body": req_body,
                "prompt_path_used": mapping["prompt_paths"][0] if mapping["prompt_paths"] else None,
                "prompt_value": prompt
            },
            "response": {
                "status": status,
                "headers": dict(r.headers) if not error else {},
                "body": resp_body,
                "response_path_used": mapping["response_paths"][0] if mapping["response_paths"] else None,
                "message_extracted": message
            },
            "latency_ms": latency_ms,
            "error_detected": error
        }
        with open(os.path.join(out_dir, f"{idx:06}.json"), "w") as f:
            json.dump(artifact, f, indent=2)
        results.append(artifact)
    client.close()
    return results
import json, time, httpx, os
from typing import Dict, Any, List
from .utils.jsonpath_utils import jsonpath_get, jsonpath_set
from .config import ARTIFACTS_DIR
import uuid

def _replace_input_placeholders(obj, placeholder: str, value: str):
    if isinstance(obj, dict):
        return {k: _replace_input_placeholders(v, placeholder, value) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_replace_input_placeholders(v, placeholder, value) for v in obj]
    if isinstance(obj, str) and obj == placeholder:
        return value
    return obj


def build_request(headers, body, prompt_paths, prompt_text, *, placeholder: str = "${input}", session_id_field: str | None = None, api_key_name: str | None = None, api_key_value: str | None = None, session_id_value: str | None = None):
    body_copy = json.loads(json.dumps(body))
    # Replace placeholder occurrences
    body_copy = _replace_input_placeholders(body_copy, placeholder, prompt_text)
    # Add session id if requested
    if session_id_field and isinstance(body_copy, dict):
        body_copy.setdefault(session_id_field, session_id_value or str(uuid.uuid4()))
    # Build headers, adding API key if provided
    headers_copy = dict(headers or {})
    if api_key_name and api_key_value:
        headers_copy[api_key_name] = api_key_value
    # Also support legacy JSONPath prompt_paths as fallback
    if prompt_paths:
        placed = False
        tmp = json.loads(json.dumps(body_copy))
        for p in prompt_paths:
            if jsonpath_set(tmp, p, prompt_text):
                placed = True
                break
        if placed:
            body_copy = tmp
    return headers_copy, body_copy

def execute_dataset(run_id: str, system: Dict[str,Any], mapping: Dict[str,Any], dataset: List[Dict[str,Any]]):
    out_dir = os.path.join(ARTIFACTS_DIR, run_id)
    os.makedirs(out_dir, exist_ok=True)
    client = httpx.Client(timeout=15.0)
    results = []
    for idx, item in enumerate(dataset, start=1):
        session_uuid = str(uuid.uuid4()) if mapping.get("session_id_field") else None
        conversation = item.get("conversation")
        if conversation:
            conv_log = []
            last_latency = 0
            error = False
            last_status = None
            last_resp_body = None
            last_headers = {}
            for turn in conversation:
                if not isinstance(turn, dict) or turn.get("role") != "user":
                    # Only send user messages to the system
                    continue
                prompt = turn.get("content", "")
                headers, req_body = build_request(
                    system["headers"],
                    system["body"],
                    mapping.get("prompt_paths", []),
                    prompt,
                    placeholder=mapping.get("input_placeholder", "${input}"),
                    session_id_field=mapping.get("session_id_field"),
                    api_key_name=mapping.get("api_key_name"),
                    api_key_value=mapping.get("api_key_value"),
                    session_id_value=session_uuid,
                )
                t0 = time.time()
                resp_body = None
                status = None
                try:
                    if system["method"].upper() == "GET":
                        r = client.get(system["endpoint"], headers=headers, params=req_body)
                    else:
                        r = client.post(system["endpoint"], headers=headers, json=req_body)
                    status = r.status_code
                    resp_body = r.json() if "application/json" in r.headers.get("content-type","") else {"text": r.text}
                    last_headers = dict(r.headers)
                except Exception as e:
                    error = True
                    resp_body = {"error": str(e)}
                    status = 0
                last_latency = int((time.time() - t0)*1000)
                last_status = status

                # error rules
                rules = mapping["error_rules"]
                if status and 200 <= int(status) < 400:
                    err_flag = False
                else:
                    err_flag = False
                    if any(lo <= status <= hi for lo,hi in rules.get("status_code_ranges", [])):
                        err_flag = True
                    if not err_flag:
                        import re
                        body_text = json.dumps(resp_body).lower()
                        for path in rules.get("json_field_paths", []):
                            if jsonpath_get(resp_body, path) is not None:
                                err_flag = True; break
                        if not err_flag:
                            for rgx in rules.get("regexes", []):
                                if re.search(rgx, body_text):
                                    err_flag = True; break
                error = error or err_flag

                # extract message
                message = None
                if not error:
                    extractor_code = mapping.get("message_extractor")
                    if extractor_code:
                        ns = {}
                        try:
                            exec(extractor_code, {"__builtins__": {}}, ns)
                            if "extract_message" in ns and callable(ns["extract_message"]):
                                message = ns["extract_message"](resp_body)
                        except Exception:
                            message = None
                    if message in (None, ""):
                        for rp in mapping.get("response_paths", []) or []:
                            val = jsonpath_get(resp_body, rp)
                            if val is not None:
                                message = val
                                break
                        if message in (None, ""):
                            message = resp_body.get("output") or resp_body.get("message") or resp_body.get("text")

                conv_log.append({"role": "user", "content": prompt})
                conv_log.append({"role": "assistant", "content": message or ""})
                last_resp_body = resp_body

            artifact = {
                "idx": idx,
                "conversation": conv_log,
                "response": {
                    "status": last_status,
                    "headers": last_headers if not error else {},
                    "body": last_resp_body,
                    "message_extracted": conv_log[-1]["content"] if conv_log else ""
                },
                "latency_ms": last_latency,
                "error_detected": error
            }
            with open(os.path.join(out_dir, f"{idx:06}.json"), "w") as f:
                json.dump(artifact, f, indent=2)
            results.append(artifact)
        else:
            prompt = item.get("prompt","")
            headers, req_body = build_request(
                system["headers"],
                system["body"],
                mapping.get("prompt_paths", []),
                prompt,
                placeholder=mapping.get("input_placeholder", "${input}"),
                session_id_field=mapping.get("session_id_field"),
                api_key_name=mapping.get("api_key_name"),
                api_key_value=mapping.get("api_key_value"),
                session_id_value=session_uuid,
            )
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
            if status and 200 <= int(status) < 400:
                err_flag = False
            else:
                err_flag = False
                if any(lo <= status <= hi for lo,hi in rules.get("status_code_ranges", [])):
                    err_flag = True
                if not err_flag:
                    import re
                    body_text = json.dumps(resp_body).lower()
                    for path in rules.get("json_field_paths", []):
                        if jsonpath_get(resp_body, path) is not None:
                            err_flag = True; break
                    if not err_flag:
                        for rgx in rules.get("regexes", []):
                            if re.search(rgx, body_text):
                                err_flag = True; break

            error = error or err_flag

            # extract message
            message = None
            if not error:
                # Preferred: use saved extractor function
                extractor_code = mapping.get("message_extractor")
                if extractor_code:
                    ns = {}
                    try:
                        exec(extractor_code, {"__builtins__": {}}, ns)
                        if "extract_message" in ns and callable(ns["extract_message"]):
                            message = ns["extract_message"](resp_body)
                    except Exception:
                        message = None
                # Fallback: use response_paths or common keys
                if message in (None, ""):
                    for rp in mapping.get("response_paths", []) or []:
                        val = jsonpath_get(resp_body, rp)
                        if val is not None:
                            message = val
                            break
                    if message in (None, ""):
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

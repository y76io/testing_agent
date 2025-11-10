import json, re, time
from typing import Dict, List, Any
from .clients.gemini_client import GeminiClient

COMMON_PROMPT_KEYS = [
    "prompt","input","query","message","user_input",
    # chat formats
    "messages","inputs","question","text"
]
COMMON_RESPONSE_KEYS = [
    "output","answer","message","content","data","result",
    "choices","completion"
]

def heuristic_paths(headers: Dict[str,Any], body: Dict[str,Any]):
    prompt_paths, response_paths = [], []
    # Heuristic for prompt in body
    def scan(obj, path="$"):
        if isinstance(obj, dict):
            for k,v in obj.items():
                newp = f"{path}.{k}" if path != "$" else f"$.{k}"
                if k.lower() in COMMON_PROMPT_KEYS:
                    prompt_paths.append(newp)
                if k.lower() in COMMON_RESPONSE_KEYS:
                    response_paths.append(newp)
                scan(v, newp)
        elif isinstance(obj, list):
            for i,v in enumerate(obj):
                newp = f"{path}[{i}]"
                scan(v, newp)
    scan(body, "$")
    # also check headers for prompt-like keys
    for k in headers.keys():
        if k.lower() in COMMON_PROMPT_KEYS:
            prompt_paths.append(f"$.{k}")
    return list(dict.fromkeys(prompt_paths)), list(dict.fromkeys(response_paths))

def llm_refine(headers: Dict[str,Any], body: Dict[str,Any]) -> Dict[str,List[str]]:
    prompt = f"""You are a JSONPath assistant. Given the following HTTP request samples,
identify likely JSONPath(s) to place a user chat prompt, and likely JSONPath(s) in typical responses to extract the returned message.
If 'messages' exists, prefer the last user message content path. Return two JSON arrays keys: prompt_paths, response_paths.

headers JSON:
{json.dumps(headers, indent=2)}
body JSON:
{json.dumps(body, indent=2)}

Only output JSON with keys: prompt_paths, response_paths.
"""
    try:
        client = GeminiClient()
        text = client.complete(prompt).strip()
        try:
            data = json.loads(text)
            return {
                "prompt_paths": data.get("prompt_paths", []),
                "response_paths": data.get("response_paths", []),
            }
        except Exception:
            return {"prompt_paths": [], "response_paths": []}
    except Exception:
        # If Gemini is unavailable (e.g., no API key), fallback gracefully
        return {"prompt_paths": [], "response_paths": []}

def detect_error_rules() -> Dict[str,Any]:
    return {
        "status_code_ranges": [[400,599]],
        "json_field_paths": ["$.error", "$.message"],
        "regexes": ["(?i)error", "(?i)invalid api key", "(?i)not authorized"],
        "llm_error_check": True,
        "advice": [
            "Verify Authorization header/token.",
            "Ensure request schema matches mapping prompt path."
        ]
    }

def detect_mapping(headers: Dict[str,Any], body: Dict[str,Any]) -> Dict[str,Any]:
    h_prompts, h_resps = heuristic_paths(headers, body)
    llm = llm_refine(headers, body)
    prompt_paths = list(dict.fromkeys(llm.get("prompt_paths", []) + h_prompts))
    response_paths = list(dict.fromkeys(llm.get("response_paths", []) + h_resps))
    rules = detect_error_rules()
    return {
        "prompt_paths": prompt_paths or ["$.input"],
        "response_paths": response_paths or ["$.output"],
        "error_rules": rules
    }


def llm_generate_extractor(resp_body: Dict[str, Any]) -> str:
    """Ask LLM to produce a Python function `def extract_message(resp): ...` returning a string.
    The function must assume `resp` is a Python dict (already JSON-decoded) and must not import modules.
    """
    try:
        client = GeminiClient()
    except Exception:
        client = None

    if not client:
        # Fallback heuristic extractor function
        return (
            "def extract_message(resp):\n"
            "    # heuristic fallback\n"
            "    for k in ['message','output','text','answer','content']:\n"
            "        if isinstance(resp, dict) and k in resp: return resp[k]\n"
            "    # nested common shapes\n"
            "    try:\n"
            "        choices = resp.get('choices')\n"
            "        if isinstance(choices, list) and choices:\n"
            "            c = choices[0]\n"
            "            if isinstance(c, dict):\n"
            "                return c.get('message') or c.get('text') or ''\n"
            "    except Exception:\n"
            "        pass\n"
            "    return ''\n"
        )

    prompt = f"""
You are an API response parsing assistant. Given the following JSON response (already parsed as a Python dict), write a minimal Python function that extracts the assistant's textual reply.

Requirements:
- Function name must be: extract_message
- Signature must be: def extract_message(resp):
- Assume resp is a Python dict.
- Do not import any modules.
- Return a string ('' if not found).

Response JSON (pretty):
{json.dumps(resp_body, indent=2)}

Return only the function code, no explanations.
"""
    code = client.complete(prompt).strip()
    # Basic guardrails: ensure function name exists
    if "def extract_message(" not in code:
        return (
            "def extract_message(resp):\n"
            "    return ''\n"
        )
    return code


def llm_analyze_error(status: int, resp_body: Dict[str, Any], req_body: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Ask LLM to determine if the response indicates an error and offer advice.
    Returns: {"is_error": bool, "reasons": [str], "advice": [str]}
    """
    # Heuristic baseline
    reasons = []
    is_err = False
    if status and status >= 400:
        is_err = True
        reasons.append(f"HTTP status {status}")
    body_text = json.dumps(resp_body).lower()
    for rgx in ["error", "invalid api key", "not authorized", "missing", "bad request", "forbidden", "unauthorized"]:
        if rgx in body_text:
            is_err = True
            if rgx != "error":
                reasons.append(rgx)
    baseline = {"is_error": is_err, "reasons": reasons, "advice": []}

    # Try LLM refinement
    try:
        client = GeminiClient()
    except Exception:
        client = None

    if not client:
        if is_err and not baseline["advice"]:
            baseline["advice"] = [
                "Verify API key header name/value.",
                "Confirm request body schema (placeholder field).",
                "Check authentication/authorization and allowed endpoint.",
            ]
        return baseline

    prompt = f"""
You analyze API responses. Decide if the response indicates an error vs a normal successful reply.
Return strict JSON with keys: is_error (true/false), reasons (array of strings), advice (array of strings).

Consider the HTTP status and body; if an error, provide practical advice on fixing the request (e.g., header name/value, missing fields, input format).

HTTP status: {status}
Request JSON (if available):
{json.dumps(req_body or {}, indent=2)}

Response JSON:
{json.dumps(resp_body, indent=2)}

Only output JSON.
"""
    try:
        text = client.complete(prompt).strip()
        data = json.loads(text)
        return {
            "is_error": bool(data.get("is_error", is_err)),
            "reasons": list(data.get("reasons", reasons) or []),
            "advice": list(data.get("advice", []) or []),
        }
    except Exception:
        return baseline

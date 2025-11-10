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

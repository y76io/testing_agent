import os, json, datetime
from fastapi import FastAPI, HTTPException, UploadFile, File, Body, Request, Form
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from .db import Base, engine, SessionLocal
from .models import System, Mapping, Run, Metric, Standard, Artifact, MetricResult, Dataset, DatasetItem, StandardMetric
from .schemas import (
    SUTIn, MappingDetectIn, MappingOut, PlanIn, RunIn, EvaluateIn, RunResultOut,
    DatasetIn, DatasetItemIn, StandardMetricIn, MetricUpdateIn,
)
from .mapping_service import detect_mapping, llm_generate_extractor, llm_analyze_error
from .executor import execute_dataset
from .evaluator import evaluate
from .reporting import export_csv, export_pdf
from .config import ARTIFACTS_DIR, REPORTS_DIR
from .utils.applicability import dataset_supports_metric, compute_items_stats
import json

app = FastAPI(title="Testing Agent (No LangFlow)")
templates = Jinja2Templates(directory="backend/templates")

# Initialize DB
Base.metadata.create_all(bind=engine)

# Best-effort additive schema adjustments for SQLite dev DB
def _ensure_columns_sqlite():
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            # helper to check and add column if missing
            def ensure(table: str, col: str, ddl: str):
                rows = conn.execute(text(f"PRAGMA table_info({table})")).fetchall()
                cols = {r[1] for r in rows}
                if col not in cols:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}"))
            ensure("standards", "version", "VARCHAR")
            ensure("metrics", "unit", "VARCHAR")
            ensure("metrics", "aggregation", "VARCHAR")
            ensure("metrics", "applicability", "TEXT")
            ensure("runs", "mapping_snapshot", "TEXT")
            ensure("runs", "selected_metric_codes", "TEXT")
            ensure("runs", "evaluation_id", "VARCHAR")
            ensure("mappings", "input_placeholder", "VARCHAR")
            ensure("mappings", "message_extractor", "TEXT")
            ensure("mappings", "session_id_field", "VARCHAR")
            ensure("mappings", "api_key_name", "VARCHAR")
            ensure("mappings", "api_key_value", "VARCHAR")
    except Exception:
        # silent in case of non-SQLite/permissions
        pass

_ensure_columns_sqlite()

def now_iso():
    return datetime.datetime.utcnow().isoformat()

def _load_standard_summary():
    import pathlib
    p = pathlib.Path("data/ISO_24001.json")
    if not p.exists():
        return {"name": "ISO 24001", "groups": []}
    data = json.loads(p.read_text())
    name = data.get("standard", {}).get("name", "ISO 24001")
    clauses = data.get("clauses", [])
    groups = {}
    for c in clauses:
        for s in c.get("subclauses", []) or []:
            em = s.get("evaluation_mode", "Unknown")
            ac = s.get("access", "Unknown")
            key = (em, ac)
            groups.setdefault(key, []).append({
                "id": s.get("id"),
                "name": s.get("name"),
                "methods": s.get("methods", [])
            })
    out = []
    for (em, ac), items in groups.items():
        out.append({"evaluation_mode": em, "access": ac, "items": items})
    return {"name": name, "groups": out}

def _load_metrics_list():
    import pathlib
    p = pathlib.Path("data/metrics.json")
    if not p.exists():
        return []
    data = json.loads(p.read_text())
    return [{"code": m.get("code"), "desc": m.get("description", ""), "type": m.get("type")} for m in data.get("metrics", [])]

@app.get("/")
def root(request: Request):
    return templates.TemplateResponse("landing.html", {"request": request})

@app.get("/healthz")
def healthz():
    return {"status": "ok", "time": now_iso()}

@app.get("/info")
def info():
    return {
        "ok": True,
        "service": "Testing Agent",
        "version": "mvp",
        "ui": "/",
        "docs": "/docs",
        "health": "/healthz",
        "endpoints": {
            "systems": ["/sut", "/sut/{system_id}"],
            "mapping": ["/mapping/detect"],
            "planning": ["/plan"],
            "execution": ["/run"],
            "evaluation": ["/evaluate", "/run/{run_id}/results", "/runs"],
            "reports": ["/reports/{run_id}/csv", "/reports/{run_id}/pdf"],
            "datasets": ["/datasets", "/datasets/items"],
            "standards_metrics": ["/standards/metrics", "/metrics/update"],
        },
        "notes": "Use the UI at '/' for a streamlined flow or see /docs for the full API.",
    }


# ---- Minimal UI ----

@app.get("/ui", response_class=HTMLResponse)
def ui_index(request: Request):
    return root(request)


@app.get("/ui/evals", response_class=HTMLResponse)
def ui_evals(request: Request):
    # Placeholder: list of full evaluations (non-API/manual). Will be populated later.
    db = SessionLocal()
    try:
        full_runs = db.query(Run).filter((Run.mode == "standard")).order_by(Run.id.desc()).all()
        return templates.TemplateResponse("evals_list.html", {"request": request, "full_runs": full_runs})
    finally:
        db.close()


def _load_standard_points():
    import pathlib
    p = pathlib.Path("data/ISO_24001.json")
    if not p.exists():
        return {"name": "ISO 24001", "points": []}
    data = json.loads(p.read_text())
    name = data.get("standard", {}).get("name", "ISO 24001")
    # Build evaluation link index id -> {path,title}
    link_index = {}
    for lk in data.get("evaluation_links_index", []) or []:
        link_index[lk.get("id")] = {"path": lk.get("path"), "title": lk.get("title")}
    pts = []
    def to_list(val):
        if val is None:
            return []
        if isinstance(val, list):
            return [x.strip() for x in val if x]
        # split on '+' if present
        return [x.strip() for x in str(val).split('+') if x.strip()]

    for c in data.get("clauses", []) or []:
        for s in c.get("subclauses", []) or []:
            modes = to_list(s.get("evaluation_mode"))
            if not modes:
                modes = [m for m in (s.get("evaluation_modes") or [])]
            accs = to_list(s.get("access"))
            eval_links = []
            for lid in (s.get("evaluation_links") or []):
                meta = link_index.get(lid)
                if meta:
                    eval_links.append({"id": lid, **meta})
            for em in modes:
                for ac in accs:
                    pts.append({
                        "id": s.get("id"),
                        "name": s.get("name"),
                        "evaluation_mode": em,
                        "access": ac,
                        "links": eval_links,
                    })
    return {"name": name, "points": pts}


@app.get("/ui/start", response_class=HTMLResponse)
def ui_start_eval(request: Request):
    standard = _load_standard_points()
    import uuid
    eval_id = f"eval-{uuid.uuid4()}"
    return templates.TemplateResponse("start_eval.html", {"request": request, "standard": standard, "eval_id": eval_id})


@app.post("/ui/start", response_class=HTMLResponse)
def ui_start_eval_submit(request: Request,
                         standard_code: str = Form("ISO_24001"),
                         manual_metrics: str | None = Form(None),
                         metric: list[str] | None = Form(None),
                         eval_id: str = Form(...)):
    std = _load_standard_points()
    all_pts = std["points"]
    if manual_metrics:
        selected_ids = set(metric or [])
        selected = [p for p in all_pts if p["id"] in selected_ids]
    else:
        selected = list(all_pts)
    # group by evaluation_mode then access
    groups_by_mode = {}
    for p in selected:
        em = p.get("evaluation_mode", "Unknown")
        ac = p.get("access", "Unknown")
        groups_by_mode.setdefault(em, {})
        groups_by_mode[em].setdefault(ac, [])
        groups_by_mode[em][ac].append(p)
    # Prepare structure for template
    grouped = []
    for em, acc_map in groups_by_mode.items():
        grouped.append({
            "evaluation_mode": em,
            "access_groups": [{"access": ac, "items": items} for ac, items in acc_map.items()]
        })
    return templates.TemplateResponse("plan_sections.html", {"request": request, "standard_name": std["name"], "groups": grouped, "eval_id": eval_id})


@app.get("/ui/api", response_class=HTMLResponse)
def ui_api_tests(request: Request):
    db = SessionLocal()
    try:
        eval_id = request.query_params.get('eval_id')
        item_id = request.query_params.get('item_id')
        metric_id = request.query_params.get('metric_id')
        q = db.query(Run).filter((Run.mode == "manual"))
        if eval_id:
            q = q.filter(Run.evaluation_id == eval_id)
        if item_id:
            # Filter runs by system_id suffix match? For now, show all API runs within eval
            pass
        api_runs = q.order_by(Run.id.desc()).all()
        return templates.TemplateResponse("api_list.html", {"request": request, "api_runs": api_runs, "eval_id": eval_id, "item_id": item_id, "metric_id": metric_id})
    finally:
        db.close()


@app.get("/ui/new", response_class=HTMLResponse)
def ui_new_form(request: Request):
    eval_id = request.query_params.get('eval_id')
    item_id = request.query_params.get('item_id')
    metric_id = request.query_params.get('metric_id')
    return templates.TemplateResponse("new_eval.html", {"request": request, "eval_id": eval_id, "item_id": item_id, "metric_id": metric_id})


@app.post("/ui/new", response_class=HTMLResponse)
def ui_new_submit(request: Request,
                  system_id: str = Form(...),
                  name: str = Form(""),
                  endpoint: str = Form(...),
                  method: str = Form("POST"),
                  headers_json: str = Form("{}"),
                  body_json: str = Form("{}"),
                  test_prompt: str = Form("Hello"),
                  add_api_key: str | None = Form(None),
                  api_key_name: str = Form(""),
                  api_key_value: str = Form(""),
                  add_session_id: str | None = Form(None),
                  session_id_field: str = Form("session_id")):
    # Parse JSON
    try:
        headers = json.loads(headers_json or "{}")
        body = json.loads(body_json or "{}")
    except Exception as e:
        return templates.TemplateResponse("new_eval.html", {"request": request, "error": f"Invalid JSON: {e}"})

    # SUT upsert
    _sut = SUTIn(system_id=system_id, name=name or system_id, endpoint=endpoint, method=method or "POST", headers=headers, body=body)
    create_or_update_sut(_sut)

    # Build test headers/body exactly as provided, plus optional API key and session id
    headers_final = dict(headers)
    if add_api_key:
        if not api_key_name or not api_key_value:
            return templates.TemplateResponse("new_eval.html", {"request": request, "error": "API key name and value are required when enabled."})
        headers_final[api_key_name] = api_key_value
    body_final = json.loads(json.dumps(body))
    if add_session_id:
        if isinstance(body_final, dict) and session_id_field:
            import uuid as _uuid
            body_final[session_id_field] = str(_uuid.uuid4())
    # Create a Mapping row now to persist extractor + options
    # Also attempt heuristic/LLM mapping detection to fill error rules and potential paths
    mdet = MappingDetectIn(system_id=system_id, headers=headers, body=body)
    md = detect_mapping(headers, body)
    mapping_row = Mapping(
        system_id=system_id,
        prompt_paths=json.dumps(md.get("prompt_paths", [])),
        response_paths=json.dumps(md.get("response_paths", [])),
        error_rules=json.dumps(md.get("error_rules", {})),
        created_at=now_iso(),
        input_placeholder="${input}",
        session_id_field=session_id_field if add_session_id else None,
        api_key_name=api_key_name if add_api_key else None,
        api_key_value=api_key_value if add_api_key else None,
    )
    db = SessionLocal()
    try:
        db.add(mapping_row)
        db.commit()
    finally:
        db.close()

    # Plan inline run (single test prompt)
    run_id = f"ui-{datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"
    plan_payload = PlanIn(
        run_id=run_id,
        system_id=system_id,
        mode="manual",
        standard_code=None,
        metric_codes=[],
        dataset={"type": "inline", "lines": [{"prompt": test_prompt}]}
    )
    create_plan(plan_payload)
    # Attach evaluation_id to this run if provided
    eval_id = request.query_params.get('eval_id')
    if eval_id:
        db2 = SessionLocal()
        try:
            r = db2.query(Run).filter_by(run_id=run_id).first()
            if r:
                r.evaluation_id = eval_id
                db2.commit()
        finally:
            db2.close()

    # Execute
    # Execute with one-off headers/body modifications
    # Temporarily override the system headers/body for this request by writing to DB, or pass via executor path.
    # For simplicity here, we update the System headers/body just for execution and then revert.
    db = SessionLocal()
    try:
        sys = db.query(System).filter_by(system_id=system_id).first()
        original_headers = sys.headers
        original_body = sys.body
        sys.headers = json.dumps(headers_final)
        sys.body = json.dumps(body_final)
        db.commit()
        run_execute(RunIn(run_id=run_id))
        # Restore originals
        sys.headers = original_headers
        sys.body = original_body
        db.commit()
    finally:
        db.close()

    # Load artifacts for display
    from .evaluator import load_artifacts
    artifacts = load_artifacts(run_id)
    # Analyze response for errors, and if not error, generate/save extractor and compute preview
    analysis = {"is_error": False, "reasons": [], "advice": []}
    extractor_code = None
    extracted_preview = ""
    if artifacts:
        first = artifacts[0]
        analysis = llm_analyze_error(first.get("response", {}).get("status", 0), first.get("response", {}).get("body", {}), first.get("request", {}).get("body", {}))
        if not analysis.get("is_error"):
            extractor_code = llm_generate_extractor(first.get("response", {}).get("body", {}))
            # Save extractor on the latest mapping for the system
            db = SessionLocal()
            try:
                mapping_latest = db.query(Mapping).filter_by(system_id=system_id).order_by(Mapping.id.desc()).first()
                if mapping_latest:
                    mapping_latest.message_extractor = extractor_code
                    db.commit()
            finally:
                db.close()
            # Compute extraction preview
            if extractor_code:
                ns = {}
                try:
                    exec(extractor_code, {"__builtins__": {}}, ns)
                    if "extract_message" in ns and callable(ns["extract_message"]):
                        extracted_preview = ns["extract_message"](first.get("response", {}).get("body", {})) or ""
                except Exception:
                    extracted_preview = ""

    item_id = request.query_params.get('item_id')
    metric_id = request.query_params.get('metric_id')
    return templates.TemplateResponse("new_eval_result.html", {
        "request": request,
        "system_id": system_id,
        "run_id": run_id,
        "mapping": {
            "prompt_paths": md.get("prompt_paths", []),
            "response_paths": md.get("response_paths", []),
            "error_rules": md.get("error_rules", {}),
            "input_placeholder": "${input}",
            "api_key_name": api_key_name if add_api_key else None,
            "session_id_field": session_id_field if add_session_id else None,
            "message_extractor": extractor_code,
        },
        "artifacts": artifacts,
        "analysis": analysis,
        "extracted_preview": extracted_preview,
        # previous values to enable re-run on the same page
        "prev": {
            "system_id": system_id,
            "name": name,
            "endpoint": endpoint,
            "method": method,
            "headers_json": json.dumps(headers, indent=2),
            "body_json": json.dumps(body, indent=2),
            "test_prompt": test_prompt,
            "add_api_key": bool(add_api_key),
            "api_key_name": api_key_name,
            "api_key_value": api_key_value,
            "add_session_id": bool(add_session_id),
            "session_id_field": session_id_field,
        },
        "eval_id": eval_id,
        "item_id": item_id,
        "metric_id": metric_id,
    })


def _route_for_mode_access(mode: str, access: str) -> str:
    mode = (mode or "").lower()
    access = (access or "").lower()
    if "system-level" in mode and "black-box" in access:
        return "/ui/api"
    if "evidence-based" in mode:
        return "/ui/evidence"
    if "organizational-judgment" in mode or "people" in mode:
        return "/ui/org"
    return "/ui/evidence"  # default placeholder


@app.get("/ui/eval")
def ui_eval_router(request: Request, eval_id: str, item_id: str, mode: str, access: str, metric_id: str | None = None):
    base = _route_for_mode_access(mode, access)
    # Preserve all params
    q = f"eval_id={eval_id}&item_id={item_id}&mode={mode}&access={access}"
    if metric_id:
        q += f"&metric_id={metric_id}"
    return RedirectResponse(url=f"{base}?{q}")


@app.get("/ui/evidence", response_class=HTMLResponse)
def ui_evidence(request: Request):
    eval_id = request.query_params.get('eval_id')
    item_id = request.query_params.get('item_id')
    metric_id = request.query_params.get('metric_id')
    mode = request.query_params.get('mode')
    access = request.query_params.get('access')
    return templates.TemplateResponse("evidence_placeholder.html", {"request": request, "eval_id": eval_id, "item_id": item_id, "metric_id": metric_id, "mode": mode, "access": access})


@app.get("/ui/org", response_class=HTMLResponse)
def ui_org(request: Request):
    eval_id = request.query_params.get('eval_id')
    item_id = request.query_params.get('item_id')
    metric_id = request.query_params.get('metric_id')
    mode = request.query_params.get('mode')
    access = request.query_params.get('access')
    return templates.TemplateResponse("org_placeholder.html", {"request": request, "eval_id": eval_id, "item_id": item_id, "metric_id": metric_id, "mode": mode, "access": access})


def _resolve_metric_path(metric_id: str):
    std = _load_standard_points()
    for p in std["points"]:
        for ln in p.get("links", []) or []:
            if ln.get("id") == metric_id:
                # Build absolute path under data/
                import pathlib
                base = pathlib.Path("data")
                path = ln.get("path")
                full = base / path
                return str(full)
    return None


@app.get("/ui/metric_runner", response_class=HTMLResponse)
def ui_metric_runner(request: Request):
    eval_id = request.query_params.get('eval_id')
    item_id = request.query_params.get('item_id')
    metric_id = request.query_params.get('metric_id')
    system_id = request.query_params.get('system_id') or ''
    # Load metric name
    metric_name = None
    mpath = _resolve_metric_path(metric_id) if metric_id else None
    if mpath:
        try:
            spec = json.loads(open(mpath).read())
            metric_name = spec.get('meta', {}).get('display_name')
        except Exception:
            metric_name = None
    return templates.TemplateResponse("metric_runner.html", {"request": request, "eval_id": eval_id, "item_id": item_id, "metric_id": metric_id, "system_id": system_id, "metric_name": metric_name})


@app.post("/ui/metric_runner", response_class=HTMLResponse)
def ui_metric_runner_post(request: Request,
                          eval_id: str = Form(...),
                          item_id: str = Form(...),
                          metric_id: str = Form(...),
                          system_id: str = Form(...),
                          run_mode: str = Form("regular"),
                          custom_reps: int = Form(3)):
    # Load metric spec
    mpath = _resolve_metric_path(metric_id)
    if not mpath:
        raise HTTPException(400, "Unknown metric")
    try:
        spec = json.loads(open(mpath).read())
    except Exception as e:
        raise HTTPException(400, f"Failed to load metric: {e}")

    # Build dataset of conversations
    items = []
    default_calls = spec.get('defaults', {}).get('calls_per_item') or None
    for it in spec.get('items', []) or []:
        reps = it.get('calls_per_item', default_calls) or 1
        if run_mode == 'test':
            reps = 1
        elif run_mode == 'custom':
            reps = max(1, int(custom_reps or 1))
        for _ in range(reps):
            for convo in (it.get('messages') or []):
                # Convert to conversation list of dicts
                conv = []
                for msg in convo:
                    conv.append({"role": msg.get("role"), "content": msg.get("content")})
                items.append({"conversation": conv})

    # Execute as one run
    run_id = f"metric-{datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"
    # Ensure mapping exists
    db = SessionLocal()
    try:
        mapping = db.query(Mapping).filter_by(system_id=system_id).order_by(Mapping.id.desc()).first()
        if not mapping:
            raise HTTPException(400, "No mapping found for system")
        sys = db.query(System).filter_by(system_id=system_id).first()
    finally:
        db.close()
    system = {
        "endpoint": sys.endpoint,
        "method": sys.method,
        "headers": json.loads(sys.headers or "{}"),
        "body": json.loads(sys.body or "{}")
    }
    mapping_obj = {
        "prompt_paths": json.loads(mapping.prompt_paths or "[]"),
        "response_paths": json.loads(mapping.response_paths or "[]"),
        "error_rules": json.loads(mapping.error_rules or "{}"),
        "input_placeholder": mapping.input_placeholder or "${input}",
        "session_id_field": mapping.session_id_field,
        "api_key_name": mapping.api_key_name,
        "api_key_value": mapping.api_key_value,
        "message_extractor": mapping.message_extractor,
    }

    # Save dataset transiently and execute
    execute_dataset(run_id, system, mapping_obj, items)

    # Evaluate per conversation
    from .evaluator import load_artifacts
    artifacts = load_artifacts(run_id)
    per_conv = []
    method = spec.get('scoring', {}).get('method')
    mean = 0.0
    failed = []
    if method == 'latency_ms':
        budget = spec.get('scoring', {}).get('budget_ms', 4000)
        scores = []
        for a in artifacts:
            lat = a.get('latency_ms') or 0
            # 1.0 at <= budget; linear down to 0 at 3x budget
            if lat <= budget:
                s = 1.0
            elif lat >= 3*budget:
                s = 0.0
            else:
                s = max(0.0, 1.0 - (lat - budget)/(2*budget))
            scores.append(s)
            ok = lat <= budget
            per_conv.append({"idx": a.get('idx'), "score": s, "pass": ok})
            if not ok:
                failed.append(a.get('idx'))
        mean = sum(scores)/len(scores) if scores else 0.0
    else:
        # LLM judge on conversation transcript
        from .clients.gemini_client import GeminiClient
        rubric = spec.get('scoring', {}).get('rubric', 'Score 0..1 based on quality.')
        try:
            client = GeminiClient()
        except Exception:
            client = None
        scores = []
        for a in artifacts:
            convo = a.get('conversation') or []
            transcript = "\n".join([f"{m.get('role')}: {m.get('content')}" for m in convo])
            prompt = f"{rubric}\n\nConversation transcript:\n{transcript}\n\nReturn only a number between 0 and 1."
            val = 0.0
            if client:
                try:
                    txt = client.complete(prompt).strip()
                    val = float(txt)
                except Exception:
                    val = 0.0
            scores.append(val)
            ok = val >= 0.8
            per_conv.append({"idx": a.get('idx'), "score": val, "pass": ok})
            if not ok:
                failed.append(a.get('idx'))
        mean = sum(scores)/len(scores) if scores else 0.0

    summary = {"count": len(per_conv), "mean": mean, "failed_count": len(failed)}
    # Persist MetricResult
    db = SessionLocal()
    try:
        mr = MetricResult(run_id=run_id, metric_code=metric_id, result=json.dumps({"per_conversation": per_conv, "summary": summary}), passed=1 if not failed else 0)
        db.add(mr)
        # also tag run with evaluation_id
        r = db.query(Run).filter_by(run_id=run_id).first()
        if r:
            r.evaluation_id = eval_id
            r.status = "evaluated"
            r.overall_score = mean
        db.commit()
    finally:
        db.close()

    return templates.TemplateResponse("metric_run_result.html", {"request": request, "run_id": run_id, "metric_id": metric_id, "per_conversation": per_conv, "summary": summary})


@app.post("/ui/api/run_dataset", response_class=HTMLResponse)
def ui_run_dataset(request: Request,
                   system_id: str = Form(...),
                   eval_id: str = Form(...),
                   item_id: str = Form(...),
                   metric_id: str = Form(None)):
    # Try to load dataset for this item_id from data/datasets/{item_id}.jsonl
    import pathlib
    path = pathlib.Path(f"data/datasets/{item_id}.jsonl")
    # If a metric_id is provided or built-in dataset not found, try resolve via ISO_24001.json links
    metric_dataset = None
    if not path.exists():
        # resolve metric json path
        std = _load_standard_points()
        # find any point for this item_id (they all share same links)
        p0 = next((p for p in std["points"] if p["id"] == item_id), None)
        if p0 and p0.get("links"):
            # choose the first link if metric_id not provided
            mid = metric_id or p0["links"][0]["id"]
            link = next((l for l in p0["links"] if l["id"] == mid), None)
            if link:
                metric_path = pathlib.Path("data") / "iso-iec-42001_x_25059" / link["path"].split("/")[-2] / link["path"].split("/")[-1] if "/" in link["path"] else pathlib.Path("data") / link["path"]
                if not metric_path.exists():
                    metric_path = pathlib.Path("data") / link["path"]
                if metric_path.exists():
                    try:
                        metric_dataset = json.loads(metric_path.read_text())
                    except Exception:
                        metric_dataset = None
        if not metric_dataset:
            return templates.TemplateResponse("api_list.html", {"request": request, "api_runs": [], "eval_id": eval_id, "error": f"Dataset not found for item {item_id} and metric {metric_id or ''}"})
    # Load into Dataset tables
    from .schemas import DatasetIn
    dsid = metric_id or item_id
    profile = {
        "dataset_id": dsid,
        "name": f"Dataset for {dsid}",
        "description": f"Auto-loaded dataset for {item_id}",
        "source": "local",
        "license": "",
        "profile": {"modalities": ["text"], "languages": ["en"], "domains": [], "tags": [item_id], "has_ground_truth": False, "item_schema": {"prompt":"string"}},
    }
    upsert_dataset(DatasetIn(**profile))
    # Load items
    idx = 0
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                idx += 1
                upsert_dataset_item(DatasetItemIn(dataset_id=dsid, item_index=idx, payload=json.loads(line), tags=[]))
    else:
        # Build from metric JSON messages
        for it in (metric_dataset.get("items") or []):
            # for each messages set, use last user message as prompt
            for convo in (it.get("messages") or []):
                user_msgs = [m.get("content", "") for m in convo if m.get("role") == "user"]
                prompt = user_msgs[-1] if user_msgs else (convo[-1].get("content") if convo else "")
                idx += 1
                upsert_dataset_item(DatasetItemIn(dataset_id=dsid, item_index=idx, payload={"prompt": prompt}, tags=[item_id]))
    # Create a run for this dataset and execute
    run_id = f"evalds-{datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"
    plan_payload = PlanIn(
        run_id=run_id,
        system_id=system_id,
        mode="manual",
        standard_code=None,
        metric_codes=[],
        dataset={"type": "dataset_id", "value": dsid}
    )
    create_plan(plan_payload)
    # attach eval_id
    db = SessionLocal()
    try:
        r = db.query(Run).filter_by(run_id=run_id).first()
        if r:
            r.evaluation_id = eval_id
            db.commit()
    finally:
        db.close()
    run_execute(RunIn(run_id=run_id))
    # Redirect to API list page
    return templates.TemplateResponse("api_list.html", {"request": request, "eval_id": eval_id, "api_runs": [
        {"run_id": run_id, "system_id": system_id, "status": "executed", "started_at": now_iso()}
    ]})


@app.get("/ui/run/{run_id}", response_class=HTMLResponse)
def ui_run_detail(request: Request, run_id: str):
    db = SessionLocal()
    try:
        r = db.query(Run).filter_by(run_id=run_id).first()
        from .evaluator import load_artifacts
        artifacts = load_artifacts(run_id)
        return templates.TemplateResponse("run_detail.html", {"request": request, "run": r, "artifacts": artifacts})
    finally:
        db.close()


@app.get("/ui/configure", response_class=HTMLResponse)
def ui_configure(request: Request, system_id: str | None = None):
    # Placeholder page for selecting standards/metrics after confirming API
    return templates.TemplateResponse("configure.html", {"request": request, "system_id": system_id})

@app.post("/sut")
def create_or_update_sut(payload: SUTIn):
    db = SessionLocal()
    try:
        sys = db.query(System).filter_by(system_id=payload.system_id).first()
        data = dict(
            system_id=payload.system_id,
            name=payload.name or payload.system_id,
            endpoint=payload.endpoint,
            method=payload.method.upper(),
            headers=json.dumps(payload.headers),
            body=json.dumps(payload.body),
            expected_output=json.dumps(payload.expected_output or {}),
            created_at=now_iso(),
            updated_at=now_iso()
        )
        if sys:
            for k,v in data.items():
                setattr(sys, k, v)
        else:
            sys = System(**data)
            db.add(sys)
        db.commit()
        return {"ok": True, "system_id": sys.system_id}
    finally:
        db.close()

@app.get("/sut/{system_id}")
def get_sut(system_id: str):
    db = SessionLocal()
    try:
        sys = db.query(System).filter_by(system_id=system_id).first()
        if not sys:
            raise HTTPException(404, "Not found")
        return {
            "system_id": sys.system_id,
            "name": sys.name,
            "endpoint": sys.endpoint,
            "method": sys.method,
            "headers": json.loads(sys.headers or "{}"),
            "body": json.loads(sys.body or "{}"),
            "expected_output": json.loads(sys.expected_output or "{}")
        }
    finally:
        db.close()

@app.post("/mapping/detect", response_model=MappingOut)
def mapping_detect(payload: MappingDetectIn):
    db = SessionLocal()
    try:
        m = detect_mapping(payload.headers, payload.body)
        mapping = Mapping(
            system_id=payload.system_id,
            prompt_paths=json.dumps(m["prompt_paths"]),
            response_paths=json.dumps(m["response_paths"]),
            error_rules=json.dumps(m["error_rules"]),
            created_at=now_iso()
        )
        db.add(mapping); db.commit()
        return {
            "system_id": payload.system_id,
            "prompt_paths": m["prompt_paths"],
            "response_paths": m["response_paths"],
            "error_rules": m["error_rules"]
        }
    finally:
        db.close()

@app.post("/plan")
def create_plan(payload: PlanIn):
    db = SessionLocal()
    try:
        # Validate dataset/metric applicability if dataset_ref is a registered dataset
        dataset_obj = payload.dataset or {}
        incompatible: list[dict] = []
        profile = None
        items_stats = {"count": 0}
        if dataset_obj.get("type") == "dataset_id":
            dsid = dataset_obj.get("value")
            ds = db.query(Dataset).filter_by(dataset_id=dsid).first()
            if not ds:
                raise HTTPException(400, f"Unknown dataset_id: {dsid}")
            try:
                profile = json.loads(ds.profile or "{}")
            except Exception:
                profile = {}
            # compute items_stats
            # We need applicability to check requires_reference_field; compute per metric below
            items = db.query(DatasetItem).filter_by(dataset_id=dsid).all()
            all_items = []
            for it in items:
                try:
                    all_items.append(json.loads(it.payload or "{}"))
                except Exception:
                    all_items.append({})
        # For each metric, load applicability (DB or file) and validate
        for code in payload.metric_codes:
            applicability = None
            m = db.query(Metric).filter_by(code=code).first()
            if m and (m.applicability and m.applicability.strip()):
                try:
                    applicability = json.loads(m.applicability)
                except Exception:
                    applicability = None
            else:
                # fallback file
                import pathlib
                metrics_path = pathlib.Path("data/metrics.json")
                if metrics_path.exists():
                    data = json.loads(metrics_path.read_text())
                    mm = next((x for x in data.get("metrics", []) if x.get("code") == code), None)
                    applicability = (mm or {}).get("applicability")
            if dataset_obj.get("type") == "dataset_id" and profile is not None:
                ref_field = (applicability or {}).get("requires", {}).get("requires_reference_field")
                items_stats = compute_items_stats(all_items, ref_field)
                ok, reasons = dataset_supports_metric(profile, items_stats, applicability or {})
                if not ok:
                    incompatible.append({"metric_code": code, "reasons": reasons})
        if incompatible:
            raise HTTPException(status_code=400, detail={"message": "Incompatible metrics for dataset", "incompatibilities": incompatible})

        # snapshot latest mapping
        mapping = db.query(Mapping).filter_by(system_id=payload.system_id).order_by(Mapping.id.desc()).first()
        mapping_snapshot = None
        if mapping:
            mapping_snapshot = json.dumps({
                "prompt_paths": json.loads(mapping.prompt_paths or "[]"),
                "response_paths": json.loads(mapping.response_paths or "[]"),
                "error_rules": json.loads(mapping.error_rules or "{}"),
            })
        run = Run(
            run_id=payload.run_id,
            system_id=payload.system_id,
            mode=payload.mode,
            standard_code=payload.standard_code,
            dataset_ref=json.dumps(payload.dataset),
            selected_metric_codes=json.dumps(payload.metric_codes),
            mapping_snapshot=mapping_snapshot,
            started_at=now_iso(),
            status="planned",
        )
        db.add(run)
        db.commit()
        return {"ok": True, "run_id": payload.run_id}
    finally:
        db.close()

def _load_dataset(ds):
    dtype = ds.get("type")
    if dtype == "single":
        return [{"prompt": ds.get("value","")}]
    if dtype == "inline":
        return ds.get("lines", [])
    if dtype == "dataset_id":
        db = SessionLocal()
        try:
            items = db.query(DatasetItem).filter_by(dataset_id=ds.get("value")).order_by(DatasetItem.item_index.asc()).all()
            out = []
            for it in items:
                try:
                    out.append(json.loads(it.payload or "{}"))
                except Exception:
                    out.append({})
            return out
        finally:
            db.close()
    if dtype == "jsonl":
        path = ds.get("path")
        items = []
        with open(path) as f:
            for line in f:
                line=line.strip()
                if not line: continue
                items.append(json.loads(line))
        return items
    return []

@app.post("/run")
def run_execute(payload: RunIn):
    db = SessionLocal()
    try:
        run = db.query(Run).filter_by(run_id=payload.run_id).first()
        if not run:
            raise HTTPException(404, "Run not found")
        sys = db.query(System).filter_by(system_id=run.system_id).first()
        mapping = db.query(Mapping).filter_by(system_id=run.system_id).order_by(Mapping.id.desc()).first()
        if not mapping:
            raise HTTPException(400, "No mapping found for system")
        dataset = json.loads(run.dataset_ref or "{}")
        ds = _load_dataset(dataset)
        system = {
            "endpoint": sys.endpoint,
            "method": sys.method,
            "headers": json.loads(sys.headers or "{}"),
            "body": json.loads(sys.body or "{}")
        }
        mapping_obj = {
            "prompt_paths": json.loads(mapping.prompt_paths or "[]"),
            "response_paths": json.loads(mapping.response_paths or "[]"),
            "error_rules": json.loads(mapping.error_rules or "{}"),
            "input_placeholder": mapping.input_placeholder or "${input}",
            "session_id_field": mapping.session_id_field,
            "api_key_name": mapping.api_key_name,
            "api_key_value": mapping.api_key_value,
            "message_extractor": mapping.message_extractor,
        }
        artifacts = execute_dataset(run.run_id, system, mapping_obj, ds)
        # snapshot mapping if not set
        if not run.mapping_snapshot:
            run.mapping_snapshot = json.dumps(mapping_obj)
        run.status = "executed"
        db.commit()
        return {"ok": True, "count": len(artifacts)}
    finally:
        db.close()

@app.post("/evaluate", response_model=RunResultOut)
def run_evaluate(payload: EvaluateIn):
    db = SessionLocal()
    try:
        run = db.query(Run).filter_by(run_id=payload.run_id).first()
        if not run:
            raise HTTPException(404, "Run not found")
        # load metric defs from DB or fallback to JSON files
        metric_defs = []
        for code in payload.metric_codes:
            m = db.query(Metric).filter_by(code=code).first()
            if not m:
                # try file
                import pathlib, json
                metrics_path = pathlib.Path("data/metrics.json")
                data = json.loads(metrics_path.read_text())
                mm = next((x for x in data["metrics"] if x["code"]==code), None)
                if not mm:
                    raise HTTPException(400, f"Unknown metric {code}")
                metric_defs.append(mm)
            else:
                metric_defs.append({
                    "code": m.code,
                    "description": m.description,
                    "type": m.type,
                    "weight": m.weight,
                    "threshold": m.threshold,
                    "config": json.loads(m.config or "{}")
                })
        metric_results, overall = evaluate(run.run_id, metric_defs)
        # persist
        for code, res in metric_results.items():
            mr = MetricResult(run_id=run.run_id, metric_code=code, result=json.dumps(res), passed=1 if res.get("pass") else 0)
            db.add(mr)
        run.overall_score = overall
        run.status = "evaluated"
        db.commit()
        return {"run_id": run.run_id, "overall_score": overall, "metric_pass_fail": {k: v.get("pass", False) for k,v in metric_results.items()}}
    finally:
        db.close()


# New endpoints: datasets, dataset items, standard-metric links, metric updates

@app.post("/datasets")
def upsert_dataset(payload: DatasetIn):
    db = SessionLocal()
    try:
        ds = db.query(Dataset).filter_by(dataset_id=payload.dataset_id).first()
        data = dict(
            dataset_id=payload.dataset_id,
            name=payload.name,
            description=payload.description or "",
            source=payload.source or "",
            license=payload.license or "",
            profile=json.dumps(payload.profile.dict()),
            created_at=now_iso(),
        )
        if ds:
            for k, v in data.items():
                setattr(ds, k, v)
        else:
            ds = Dataset(**data)
            db.add(ds)
        db.commit()
        return {"ok": True, "dataset_id": payload.dataset_id}
    finally:
        db.close()


@app.post("/datasets/items")
def upsert_dataset_item(payload: DatasetItemIn):
    db = SessionLocal()
    try:
        row = (
            db.query(DatasetItem)
            .filter_by(dataset_id=payload.dataset_id, item_index=payload.item_index)
            .first()
        )
        data = dict(
            dataset_id=payload.dataset_id,
            item_index=payload.item_index,
            payload=json.dumps(payload.payload),
            tags=json.dumps(payload.tags or []),
        )
        if row:
            for k, v in data.items():
                setattr(row, k, v)
        else:
            row = DatasetItem(**data)
            db.add(row)
        db.commit()
        return {"ok": True, "dataset_id": payload.dataset_id, "item_index": payload.item_index}
    finally:
        db.close()


@app.post("/standards/metrics")
def link_standard_metric(payload: StandardMetricIn):
    db = SessionLocal()
    try:
        row = (
            db.query(StandardMetric)
            .filter_by(standard_code=payload.standard_code, metric_code=payload.metric_code)
            .first()
        )
        if row:
            row.required = 1 if payload.required else 0
        else:
            row = StandardMetric(
                standard_code=payload.standard_code,
                metric_code=payload.metric_code,
                required=1 if payload.required else 0,
            )
            db.add(row)
        db.commit()
        return {"ok": True}
    finally:
        db.close()


@app.post("/metrics/update")
def update_metric(payload: MetricUpdateIn):
    db = SessionLocal()
    try:
        m = db.query(Metric).filter_by(code=payload.code).first()
        if not m:
            raise HTTPException(404, "Metric not found")
        if payload.unit is not None:
            m.unit = payload.unit
        if payload.aggregation is not None:
            m.aggregation = payload.aggregation
        if payload.applicability is not None:
            m.applicability = json.dumps(payload.applicability)
        db.commit()
        return {"ok": True}
    finally:
        db.close()

@app.get("/runs")
def list_runs():
    db = SessionLocal()
    try:
        rows = db.query(Run).all()
        return [{"run_id": r.run_id, "system_id": r.system_id, "status": r.status, "overall_score": r.overall_score} for r in rows]
    finally:
        db.close()

@app.get("/run/{run_id}/results")
def get_results(run_id: str):
    db = SessionLocal()
    try:
        r = db.query(Run).filter_by(run_id=run_id).first()
        if not r:
            raise HTTPException(404, "Run not found")
        mrs = db.query(MetricResult).filter_by(run_id=run_id).all()
        out = {mr.metric_code: json.loads(mr.result) for mr in mrs}
        return {"run_id": run_id, "overall_score": r.overall_score, "metrics": out}
    finally:
        db.close()

@app.get("/reports/{run_id}/{rtype}")
def get_report(run_id: str, rtype: str):
    db = SessionLocal()
    try:
        r = db.query(Run).filter_by(run_id=run_id).first()
        if not r:
            raise HTTPException(404, "Run not found")
        mrs = db.query(MetricResult).filter_by(run_id=run_id).all()
        out = {mr.metric_code: json.loads(mr.result) for mr in mrs}
        if rtype == "csv":
            path = export_csv(run_id, out, r.overall_score or 0.0)
        elif rtype == "pdf":
            path = export_pdf(run_id, out, r.overall_score or 0.0)
        else:
            raise HTTPException(400, "rtype must be csv|pdf")
        return FileResponse(path, filename=os.path.basename(path))
    finally:
        db.close()

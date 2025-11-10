import os, json, datetime
from fastapi import FastAPI, HTTPException, UploadFile, File, Body, Request, Form
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from .db import Base, engine, SessionLocal
from .models import System, Mapping, Run, Metric, Standard, Artifact, MetricResult, Dataset, DatasetItem, StandardMetric
from .schemas import (
    SUTIn, MappingDetectIn, MappingOut, PlanIn, RunIn, EvaluateIn, RunResultOut,
    DatasetIn, DatasetItemIn, StandardMetricIn, MetricUpdateIn,
)
from .mapping_service import detect_mapping, llm_generate_extractor
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

@app.get("/")
def root(request: Request):
    # Render the UI index for the root path
    db = SessionLocal()
    try:
        runs = db.query(Run).order_by(Run.id.desc()).limit(50).all()
        return templates.TemplateResponse("index.html", {"request": request, "runs": runs})
    finally:
        db.close()

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
    # Redirect UI index to root for a single entry point
    return root(request)


@app.get("/ui/new", response_class=HTMLResponse)
def ui_new_form(request: Request):
    return templates.TemplateResponse("new_eval.html", {"request": request})


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
    # Generate and save extractor function based on the first artifact response
    extractor_code = None
    if artifacts:
        extractor_code = llm_generate_extractor(artifacts[0].get("response", {}).get("body", {}))
        # Save extractor on the latest mapping for the system
        db = SessionLocal()
        try:
            mapping_latest = db.query(Mapping).filter_by(system_id=system_id).order_by(Mapping.id.desc()).first()
            if mapping_latest:
                mapping_latest.message_extractor = extractor_code
                db.commit()
        finally:
            db.close()

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
    })


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

"""STATGUARD FastAPI backend.

Endpoints (all JSON, camelCase, matching the React frontend's types):
  POST /api/upload            multipart CSV  -> analysis summary
  POST /api/analyze           re-run with new settings
  GET  /api/dashboard         current summary
  GET  /api/threats           filtered anomalies  (?severity=&category=&search=)
  GET  /api/investigate/{id}  one threat + baselines + recommendations
  GET|POST /api/evaluation    precision/recall/F1 vs Label column
  GET|POST /api/ensemble      per-flow detector votes
"""
from __future__ import annotations

import os
import threading
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

import engine as E

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "30"))
MAX_DATASETS = 3

app = FastAPI(title="STATGUARD API", version="1.0.0")

origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False,
                   allow_methods=["*"], allow_headers=["*"])

_lock = threading.Lock()
_datasets: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
_current: dict[str, str | None] = {"id": None}

RECOMMENDATIONS = {
    "DDoS Attack": ["Enable upstream rate limiting / CDN scrubbing for the targeted service.",
                    "Block or throttle the highest-volume source ranges at the edge firewall.",
                    "Scale out the affected service and watch connection-table utilisation."],
    "DoS Attack": ["Apply per-source connection and request limits on the targeted service.",
                   "Check server resource usage (CPU, sockets) for exhaustion."],
    "Port Scanning": ["Identify the scanning source and block it at the perimeter.",
                      "Review which ports answered and close anything unnecessary.",
                      "Add alerting for hosts touching many distinct ports."],
    "Brute Force Attack": ["Enforce account lockout / rate limiting on the authentication service.",
                           "Require MFA and key-based SSH; review recent failed logins."],
    "Suspicious Data Exfiltration": ["Inspect the destination and the volume of outbound data.",
                                     "Check the source host for compromise and review egress rules."],
    "Network Flooding": ["Throttle the noisy source and inspect for misconfigured or compromised devices."],
    "Unusual Network Access": ["Verify with the asset owner whether this access pattern is expected."],
}


def _store(name: str, parsed: dict, settings: E.Settings) -> dict[str, Any]:
    ds_id = E.new_dataset_id()
    res = E.analyze(parsed["df"], settings, ds_id, name, parsed["preprocessing"], parsed["hasLabels"])
    entry = {"id": ds_id, "name": name, "parsed": parsed, "res": res}
    with _lock:
        _datasets[ds_id] = entry
        while len(_datasets) > MAX_DATASETS:
            _datasets.popitem(last=False)
        _current["id"] = ds_id
    return entry


def _get(dataset_id: str | None = None) -> dict[str, Any]:
    with _lock:
        key = dataset_id if dataset_id in _datasets else _current["id"]
        entry = _datasets.get(key) if key else None
    if entry is None:
        raise HTTPException(status_code=404, detail="No dataset loaded. Upload a CSV first.")
    return entry


def _empty_summary() -> dict[str, Any]:
    return {"datasetName": "No Active Dataset", "analyzedAt": "", "totalRecords": 0, "normalRecords": 0,
            "anomalousRecords": 0, "anomalyRate": 0, "criticalCount": 0, "highCount": 0,
            "mediumCount": 0, "lowCount": 0, "networkRiskScore": 0,
            "categoryCounts": {c: 0 for c in E.CATEGORIES}, "protocolCounts": {}, "records": [],
            "anomalies": [], "featureBaselines": {},
            "preprocessing": {"totalRawRows": 0, "validRows": 0, "duplicatesRemoved": 0,
                              "missingValuesHandled": 0, "featuresNormalized": [], "columnMapping": {},
                              "parsingErrors": []}}


@app.get("/")
def root() -> dict[str, Any]:
    return {"service": "STATGUARD API", "status": "ok", "docs": "/docs"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> dict[str, Any]:
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large (limit {MAX_UPLOAD_MB} MB).")
    filename = file.filename or "dataset.csv"
    if not filename.lower().endswith((".csv", ".txt")):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")
    try:
        parsed = await run_in_threadpool(E.parse_csv, data, filename)
        entry = await run_in_threadpool(_store, filename, parsed, E.Settings())
    except E.DatasetError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    pre = parsed["preprocessing"]
    return {"datasetId": entry["id"], "datasetName": filename, "filename": filename,
            "totalRows": pre["totalRawRows"], "validRows": pre["validRows"],
            "message": f"Analysed {pre['validRows']:,} flows from {filename}.",
            "uploadedAt": datetime.now(timezone.utc).isoformat(), "preprocessing": pre,
            "summary": entry["res"]["summary"]}


@app.post("/api/analyze")
async def analyze(request: Request) -> dict[str, Any]:
    body = await request.json()
    entry = _get(body.get("dataset_id"))
    settings = E.Settings.from_payload(body.get("settings"))
    res = await run_in_threadpool(E.analyze, entry["parsed"]["df"], settings, entry["id"], entry["name"],
                                  entry["parsed"]["preprocessing"], entry["parsed"]["hasLabels"])
    with _lock:
        entry["res"] = res
    return {"message": "Statistical analysis completed.", **res["summary"]}


@app.get("/api/dashboard")
def dashboard() -> dict[str, Any]:
    with _lock:
        key = _current["id"]
        entry = _datasets.get(key) if key else None
    if entry is None:
        return {"hasDataset": False, "summary": _empty_summary()}
    return {"hasDataset": True, "summary": entry["res"]["summary"]}


@app.get("/api/threats")
def threats(severity: str | None = None, category: str | None = None, search: str | None = None) -> dict[str, Any]:
    with _lock:
        key = _current["id"]
        entry = _datasets.get(key) if key else None
    items = [] if entry is None else entry["res"]["summary"]["anomalies"]
    if severity and severity.upper() != "ALL":
        items = [a for a in items if a["severity"] == severity.upper()]
    if category and category != "ALL":
        items = [a for a in items if a["predictedAttack"] == category]
    if search and search.strip():
        q = search.strip().lower()
        items = [a for a in items if q in " ".join([
            a["id"], a["predictedAttack"], a["packetRecord"]["destIp"], a["packetRecord"]["sourceIp"],
            str(a["packetRecord"]["destPort"]), " ".join(a["detectionMethods"])]).lower()]
    count = lambda s: sum(1 for a in items if a["severity"] == s)  # noqa: E731
    return {"threats": items, "total": len(items), "criticalCount": count("CRITICAL"),
            "highCount": count("HIGH"), "mediumCount": count("MEDIUM"), "lowCount": count("LOW")}


@app.get("/api/investigate/{threat_id}")
def investigate(threat_id: str) -> dict[str, Any]:
    entry = _get()
    for a in entry["res"]["summary"]["anomalies"]:
        if a["id"] == threat_id or a["packetRecord"]["id"] == threat_id:
            return {"threat": a, "featureBaselines": entry["res"]["summary"]["featureBaselines"],
                    "tacticalRecommendations": RECOMMENDATIONS.get(a["predictedAttack"], []),
                    "rawFlowDetails": a["packetRecord"].get("rawRowData", {})}
    raise HTTPException(status_code=404, detail=f"Threat '{threat_id}' not found.")


async def _body_or_empty(request: Request) -> dict[str, Any]:
    try:
        return await request.json()
    except Exception:  # noqa: BLE001
        return {}


@app.get("/api/evaluation")
def evaluation_get(dataset_id: str | None = None) -> dict[str, Any]:
    entry = _get(dataset_id)
    return E.evaluation(entry["res"], entry["id"], entry["name"])


@app.post("/api/evaluation")
async def evaluation_post(request: Request) -> dict[str, Any]:
    body = await _body_or_empty(request)
    entry = _get(body.get("dataset_id"))
    return E.evaluation(entry["res"], entry["id"], entry["name"])


@app.get("/api/ensemble")
def ensemble_get(dataset_id: str | None = None) -> dict[str, Any]:
    entry = _get(dataset_id)
    return E.ensemble(entry["res"], entry["id"], entry["name"])


@app.post("/api/ensemble")
async def ensemble_post(request: Request) -> dict[str, Any]:
    body = await _body_or_empty(request)
    entry = _get(body.get("dataset_id"))
    return E.ensemble(entry["res"], entry["id"], entry["name"])

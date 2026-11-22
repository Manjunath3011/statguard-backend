"""STATGUARD detection engine (pure pandas / numpy / scikit-learn, no web code).

Pipeline: parse CSV -> robust Z-score, IQR, moving average, Isolation Forest,
(entropy + rare-port signals when the data supports them) -> composite score ->
severity / category -> JSON shaped like the frontend's TypeScript types.
"""
from __future__ import annotations

import io
import math
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

MAX_RECORDS_RETURNED = 5000
MAX_ANOMALIES_RETURNED = 2000
PORT_RARE_FREQ = 0.002
RANDOM_SEED = 42

CATEGORIES = [
    "DDoS Attack", "DoS Attack", "Port Scanning", "Brute Force Attack",
    "Suspicious Data Exfiltration", "Network Flooding",
    "Unusual Network Access", "Normal Traffic",
]

ALIASES: dict[str, list[str]] = {
    "timestamp": ["timestamp", "time", "date", "ts", "start_time", "flow_start", "datetime"],
    "sourceIp": ["source_ip", "src_ip", "srcip", "sourceip", "ip_src", "source_address"],
    "destIp": ["destination_ip", "dest_ip", "dst_ip", "dstip", "destip", "ip_dst", "destination_address"],
    "sourcePort": ["source_port", "src_port", "sport", "srcport"],
    "destPort": ["destination_port", "dest_port", "dst_port", "dport", "dstport"],
    "protocol": ["protocol", "proto", "protocol_name"],
    "packetCount": ["packet_count", "packets", "total_packets", "tot_packets", "pkt_count"],
    "fwdPackets": ["total_fwd_packets", "fwd_packets", "tot_fwd_pkts"],
    "bwdPackets": ["total_backward_packets", "bwd_packets", "tot_bwd_pkts"],
    "totalBytes": ["total_bytes", "bytes", "tot_bytes"],
    "fwdBytes": ["total_length_of_fwd_packets", "fwd_bytes"],
    "bwdBytes": ["total_length_of_bwd_packets", "bwd_bytes"],
    "flowDuration": ["flow_duration", "duration", "dur"],
    "frequency": ["flow_packets_s", "packets_per_sec", "pps", "connection_frequency", "frequency"],
    "label": ["label", "class", "attack", "attack_cat", "target"],
}

BENIGN_LABELS = {"benign", "normal", "0", "0.0", "false", "legitimate", "none", "-"}


class DatasetError(ValueError):
    """Raised for unusable uploads (shown to the user as an HTTP 400)."""


def _norm(h: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(h).strip().lower()).strip("_")


@dataclass
class Settings:
    z_score_threshold: float = 3.5
    iqr_multiplier: float = 1.5
    isolation_forest_contamination: float = 0.10
    entropy_window_size: int = 25
    entropy_low_threshold: float = 1.2
    moving_avg_window_size: int = 10
    critical: float = 75
    high: float = 55
    medium: float = 35

    @classmethod
    def from_payload(cls, p: dict | None) -> "Settings":
        s = cls()
        if not p:
            return s
        s.z_score_threshold = float(p.get("z_score_threshold", s.z_score_threshold))
        s.iqr_multiplier = float(p.get("iqr_multiplier", s.iqr_multiplier))
        s.isolation_forest_contamination = min(0.5, max(0.01, float(
            p.get("isolation_forest_contamination", s.isolation_forest_contamination))))
        s.entropy_window_size = int(p.get("entropy_window_size", s.entropy_window_size))
        s.entropy_low_threshold = float(p.get("entropy_low_threshold", s.entropy_low_threshold))
        s.moving_avg_window_size = max(3, int(p.get("moving_avg_window_size", s.moving_avg_window_size)))
        st = p.get("severity_thresholds") or {}
        s.critical = float(st.get("critical", s.critical))
        s.high = float(st.get("high", s.high))
        s.medium = float(st.get("medium", s.medium))
        return s


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------
def parse_csv(content: bytes, filename: str) -> dict[str, Any]:
    try:
        raw = pd.read_csv(io.BytesIO(content), low_memory=False, skipinitialspace=True)
    except Exception as exc:  # noqa: BLE001
        raise DatasetError(f"Could not read CSV: {exc}") from exc
    if raw.empty:
        raise DatasetError("The CSV file has no rows.")

    original_headers = {_norm(c): str(c).strip() for c in raw.columns}
    raw.columns = [_norm(c) for c in raw.columns]

    mapping: dict[str, str] = {}
    for field_name, aliases in ALIASES.items():
        for a in aliases:
            if a in raw.columns:
                mapping[field_name] = a
                break

    def num(field_name: str) -> pd.Series | None:
        col = mapping.get(field_name)
        if col is None:
            return None
        return pd.to_numeric(raw[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

    fwd, bwd = num("fwdPackets"), num("bwdPackets")
    packets = num("packetCount")
    if packets is None and fwd is not None and bwd is not None:
        packets = fwd + bwd
    fbytes, bbytes = num("fwdBytes"), num("bwdBytes")
    total_bytes = num("totalBytes")
    if total_bytes is None and fbytes is not None and bbytes is not None:
        total_bytes = fbytes + bbytes

    if packets is None or total_bytes is None:
        missing = []
        if packets is None:
            missing.append("packet count (e.g. 'Total Fwd Packets' + 'Total Backward Packets')")
        if total_bytes is None:
            missing.append("byte count (e.g. 'Total Length of Fwd/Bwd Packets' or 'Bytes')")
        raise DatasetError("Required column(s) not found: " + "; ".join(missing)
                           + f". Columns seen: {', '.join(list(original_headers.values())[:25])}")

    warnings: list[str] = []
    duration = num("flowDuration")
    if duration is not None:
        # CIC-IDS2017 'Flow Duration' is in microseconds; a bare 'duration' column is treated as ms.
        if mapping["flowDuration"] == "flow_duration" and fwd is not None:
            duration = duration / 1000.0
        else:
            pass
    else:
        warnings.append("No duration column found: duration set to 1 ms (duration-based signals unreliable).")
        duration = pd.Series(1.0, index=raw.index)

    freq = num("frequency")
    calc_freq = packets / (duration.clip(lower=1.0) / 1000.0)
    if freq is None:
        freq = calc_freq
    else:
        freq = freq.where(freq.notna(), calc_freq)

    dest_ip = raw[mapping["destIp"]].astype(str).str.strip() if "destIp" in mapping else pd.Series("", index=raw.index)
    src_ip = raw[mapping["sourceIp"]].astype(str).str.strip() if "sourceIp" in mapping else pd.Series("", index=raw.index)
    if "destIp" not in mapping:
        warnings.append("No destination IP column found: entropy detection disabled.")
    sport = num("sourcePort") if "sourcePort" in mapping else pd.Series(0, index=raw.index)
    dport = num("destPort") if "destPort" in mapping else pd.Series(0, index=raw.index)
    if "destPort" not in mapping:
        warnings.append("No destination port column found: rare-port signal disabled.")
    proto = raw[mapping["protocol"]].astype(str).str.upper() if "protocol" in mapping else pd.Series("UNKNOWN", index=raw.index)

    df = pd.DataFrame({
        "rowIndex": np.arange(1, len(raw) + 1),
        "sourceIp": src_ip, "destIp": dest_ip,
        "sourcePort": sport.fillna(0).astype(int), "destPort": dport.fillna(0).astype(int),
        "protocol": proto,
        "packetCount": packets, "totalBytes": total_bytes,
        "flowDuration": duration, "connectionFrequency": freq,
        "fwdPackets": fwd if fwd is not None else (packets * 0.6).round(),
        "bwdPackets": bwd if bwd is not None else (packets * 0.4).round(),
    })
    if "label" in mapping:
        df["label"] = raw[mapping["label"]].astype(str).str.strip()
    else:
        df["label"] = None

    # Timestamps
    ts_ok = False
    if "timestamp" in mapping:
        ts = pd.to_datetime(raw[mapping["timestamp"]], errors="coerce", utc=True)
        ts_ok = bool(ts.notna().all())
        if ts_ok:
            df["timestamp"] = ts
    if not ts_ok:
        base = datetime.now(timezone.utc) - timedelta(seconds=10 * len(df))
        df["timestamp"] = [base + timedelta(seconds=10 * i) for i in range(len(df))]
        warnings.append("No usable timestamp column: sequential timestamps (10 s apart) were assigned in file order.")

    required = ["packetCount", "totalBytes", "flowDuration", "connectionFrequency"]
    bad = df[required].isna().any(axis=1) | (df["packetCount"] <= 0) | (df["totalBytes"] < 0)
    skipped = int(bad.sum())
    df = df[~bad].copy()
    if skipped:
        warnings.append(f"{skipped} row(s) skipped: missing or invalid packet/byte/duration values.")
    if len(df) < 10:
        raise DatasetError(f"Only {len(df)} usable rows after cleaning; need at least 10.")

    df["packetLength"] = (df["totalBytes"] / df["packetCount"]).round()
    df["fwdPackets"] = df["fwdPackets"].fillna(0).astype(int)
    df["bwdPackets"] = df["bwdPackets"].fillna(0).astype(int)
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True) if ts_ok else df.reset_index(drop=True)

    preprocessing = {
        "totalRawRows": int(len(raw)),
        "validRows": int(len(df)),
        "duplicatesRemoved": 0,
        "missingValuesHandled": skipped,
        "featuresNormalized": ["packetCount", "totalBytes", "connectionFrequency", "flowDuration"],
        "columnMapping": {k: original_headers.get(v, v) for k, v in mapping.items()},
        "parsingErrors": warnings,
    }
    return {"df": df, "preprocessing": preprocessing, "hasLabels": "label" in mapping}


# --------------------------------------------------------------------------
# Statistics helpers
# --------------------------------------------------------------------------
def feature_stats(x: np.ndarray, iqr_mult: float) -> dict[str, float]:
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    iqr = q3 - q1
    return {
        "mean": float(x.mean()), "stdDev": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
        "variance": float(x.var(ddof=1)) if len(x) > 1 else 0.0, "median": float(med),
        "iqr": float(iqr), "q1": float(q1), "q3": float(q3),
        "lowerBound": float(q1 - iqr_mult * iqr), "upperBound": float(q3 + iqr_mult * iqr),
        "min": float(x.min()), "max": float(x.max()),
    }


def modified_z(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad == 0:
        mad = np.mean(np.abs(x - med)) * 1.2533
    if mad == 0:
        return np.zeros_like(x, dtype=float)
    return 0.6745 * (x - med) / mad


def shannon(values: list[str]) -> float:
    if not values:
        return 0.0
    _, counts = np.unique(values, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------
def analyze(df: pd.DataFrame, settings: Settings, dataset_id: str, dataset_name: str,
            preprocessing: dict, has_labels: bool) -> dict[str, Any]:
    n = len(df)
    pc = df["packetCount"].to_numpy(float)
    tb = df["totalBytes"].to_numpy(float)
    cf = df["connectionFrequency"].to_numpy(float)
    fd = df["flowDuration"].to_numpy(float)
    pl = df["packetLength"].to_numpy(float)

    base = {k: feature_stats(v, settings.iqr_multiplier) for k, v in
            {"packetCount": pc, "totalBytes": tb, "connectionFrequency": cf,
             "flowDuration": fd, "packetLength": pl}.items()}

    # --- Robust Z-score (median / MAD) on volume features only --------------
    zp, zb, zf = modified_z(pc), modified_z(tb), modified_z(cf)
    max_abs_z = np.maximum.reduce([np.abs(zp), np.abs(zb), np.abs(zf)])
    z_flag = max_abs_z >= settings.z_score_threshold

    # --- IQR (Tukey fences) --------------------------------------------------
    def out(x, key):
        return (x > base[key]["upperBound"]) | (x < base[key]["lowerBound"])
    iqr_p, iqr_b, iqr_f = out(pc, "packetCount"), out(tb, "totalBytes"), out(cf, "connectionFrequency")
    iqr_flag = iqr_p | iqr_b | iqr_f

    # --- Moving average: window uses PREVIOUS points only -------------------
    s = pd.Series(pc)
    roll = s.shift(1).rolling(settings.moving_avg_window_size, min_periods=3)
    ma_mean = roll.mean().to_numpy()
    ma_std = roll.std().to_numpy()
    ma_upper = ma_mean + 2 * ma_std
    ma_flag = np.nan_to_num(pc > ma_upper, nan=False).astype(bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        ma_dev = np.where(ma_mean > 0, (pc - ma_mean) / ma_mean, 0.0)
    ma_dev = np.nan_to_num(ma_dev)

    # --- Isolation Forest (seeded, log-scaled features, percentile cutoff) ----
    feats = np.log1p(np.column_stack([pc, tb, cf, fd]))
    feats = StandardScaler().fit_transform(feats)
    forest = IsolationForest(n_estimators=100, max_samples=min(256, n),
                             random_state=RANDOM_SEED, n_jobs=1)
    forest.fit(feats)
    if_score = -forest.score_samples(feats)           # ~0.3 (normal) .. 0.8 (anomalous)
    if_threshold = float(np.percentile(if_score, 100 * (1 - settings.isolation_forest_contamination)))
    if_flag = if_score >= if_threshold

    # --- Entropy of destination IPs (only if the data has IP variety) --------
    dest_ips = df["destIp"].tolist()
    has_ips = len({ip for ip in dest_ips if ip and ip.lower() != "nan"}) > 1
    ent_flag = np.zeros(n, dtype=bool)
    ent_local = np.zeros(n)
    ent_dev = np.zeros(n)
    global_ent = 0.0
    if has_ips and n > 15:
        global_ent = shannon(dest_ips)
        w = settings.entropy_window_size
        for i in range(n):
            a = max(0, i - w // 2)
            ent_local[i] = shannon(dest_ips[a:a + w])
        if global_ent > 0:
            ent_dev = (global_ent - ent_local) / global_ent
        ent_flag = ent_local < min(settings.entropy_low_threshold, 0.5 * global_ent)

    # --- Rare destination port ------------------------------------------------
    ports = df["destPort"].to_numpy()
    port_signal = n >= 200 and len(np.unique(ports)) >= 3
    port_freq = np.ones(n)
    port_rare = np.zeros(n, dtype=bool)
    if port_signal:
        vals, counts = np.unique(ports, return_counts=True)
        lookup = dict(zip(vals.tolist(), (counts / n).tolist()))
        port_freq = np.array([lookup[p] for p in ports.tolist()])
        port_rare = port_freq <= PORT_RARE_FREQ

    # --- Composite score & records -------------------------------------------
    st = settings
    records_out: list[dict] = []
    anomalies: list[dict] = []
    flagged = np.zeros(n, dtype=bool)
    cat_counts = {c: 0 for c in CATEGORIES}
    proto_counts: dict[str, int] = {}
    sev_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    comp_all = np.zeros(n)

    fq = base["connectionFrequency"]
    pq = base["packetCount"]
    bq = base["totalBytes"]
    ts_iso = [t.isoformat().replace("+00:00", "Z") for t in df["timestamp"]]

    for i in range(n):
        row = df.iloc[i]
        rid = f"row-{int(row['rowIndex'])}"
        proto = str(row["protocol"])
        proto_counts[proto] = proto_counts.get(proto, 0) + 1
        rec = {
            "id": rid, "timestamp": ts_iso[i], "sourceIp": row["sourceIp"] if row["sourceIp"] != "nan" else "",
            "destIp": row["destIp"] if row["destIp"] != "nan" else "",
            "sourcePort": int(row["sourcePort"]), "destPort": int(row["destPort"]), "protocol": proto,
            "packetCount": float(pc[i]), "packetLength": float(pl[i]), "flowDuration": float(fd[i]),
            "fwdPackets": int(row["fwdPackets"]), "bwdPackets": int(row["bwdPackets"]),
            "totalBytes": float(tb[i]), "connectionFrequency": float(cf[i]),
        }
        if has_labels and row["label"] is not None:
            rec["rawRowData"] = {"Label": str(row["label"])}
        if i < MAX_RECORDS_RETURNED:
            records_out.append(rec)

        score = min(30.0, max_abs_z[i] / (2 * st.z_score_threshold) * 30) if max_abs_z[i] > 0 else 0.0
        nviol = int(iqr_p[i]) + int(iqr_b[i]) + int(iqr_f[i])
        if nviol:
            score += min(20, nviol * 8)
        if ma_flag[i]:
            score += min(15, max(5, ma_dev[i] * 10))
        if if_score[i] > 0.5:
            score += min(20.0, (if_score[i] - 0.5) / 0.3 * 20)
        if ent_flag[i]:
            score += min(15, ent_dev[i] * 15)
        if port_rare[i]:
            score += min(15, 8 + (PORT_RARE_FREQ - port_freq[i]) * 3000)
        composite = int(min(100, max(0, round(score))))
        comp_all[i] = composite

        methods: list[str] = []
        reasons: list[str] = []
        if z_flag[i]:
            methods.append(f"Modified Z-Score Analysis (|M| = {max_abs_z[i]:.2f} ≥ {st.z_score_threshold})")
            if abs(zp[i]) >= st.z_score_threshold:
                reasons.append(f"Packet count deviates {zp[i]:+.1f} robust σ from the median baseline.")
            if abs(zb[i]) >= st.z_score_threshold:
                reasons.append(f"Byte volume deviates {zb[i]:+.1f} robust σ from the median baseline.")
            if abs(zf[i]) >= st.z_score_threshold:
                reasons.append(f"Packet rate deviates {zf[i]:+.1f} robust σ from the median baseline.")
        if iqr_flag[i]:
            methods.append("Interquartile Range (IQR Outlier Boundary)")
            if iqr_p[i]:
                reasons.append(f"Packet count ({pc[i]:,.0f}) outside IQR fences "
                               f"[{pq['lowerBound']:,.0f}, {pq['upperBound']:,.0f}].")
            if iqr_b[i]:
                reasons.append(f"Byte volume ({tb[i] / 1024:,.1f} KB) outside IQR fences.")
            if iqr_f[i]:
                reasons.append(f"Packet rate ({cf[i]:,.0f}/s) outside IQR fences.")
        if ma_flag[i]:
            methods.append("Moving Average Surge (Bollinger Upper Band Breached)")
            reasons.append(f"Packet count exceeded the {st.moving_avg_window_size}-flow moving-average band "
                           "(computed from preceding flows only).")
        if ent_flag[i]:
            methods.append("Shannon Entropy Anomaly (Target IP Concentration)")
            reasons.append(f"Destination-IP entropy fell to {ent_local[i]:.2f} bits (focused on few targets).")
        if if_flag[i]:
            methods.append(f"Isolation Forest ML (Path Isolation Score {if_score[i]:.2f})")
            reasons.append("Multivariate feature vector was isolated quickly by the seeded forest.")
        if port_rare[i]:
            methods.append(f"Rare Destination Port (seen in {port_freq[i] * 100:.2f}% of flows)")
            reasons.append(f"Destination port {int(ports[i])} is used by only {port_freq[i] * 100:.2f}% of flows.")

        is_anom = len(methods) >= 2 or composite >= st.medium
        flagged[i] = is_anom
        if not is_anom:
            cat_counts["Normal Traffic"] += 1
            continue

        severity = ("CRITICAL" if composite >= st.critical else "HIGH" if composite >= st.high
                    else "MEDIUM" if composite >= st.medium else "LOW")
        dport = int(ports[i])
        if pc[i] <= 12 and (port_rare[i] or ((dport < 1024 or dport > 30000) and fd[i] < 500 and cf[i] > fq["median"])):
            cat = "Port Scanning"
        elif dport in (21, 22, 23, 3389) and cf[i] >= fq["upperBound"] and pc[i] < 60:
            cat = "Brute Force Attack"
        elif (pc[i] > pq["upperBound"] * 2.5 or zp[i] > 2 * st.z_score_threshold) and (ent_flag[i] or cf[i] > fq["upperBound"]):
            cat = "DDoS Attack"
        elif pc[i] > pq["upperBound"] or zp[i] > st.z_score_threshold:
            cat = "DoS Attack"
        elif tb[i] > bq["upperBound"] and tb[i] / max(pc[i], 1) > 1000:
            cat = "Suspicious Data Exfiltration"
        elif cf[i] > fq["upperBound"]:
            cat = "Network Flooding"
        else:
            cat = "Unusual Network Access"
        cat_counts[cat] += 1
        sev_counts[severity] += 1

        anomalies.append({
            "id": f"anom-{rid}", "packetRecord": rec, "compositeScore": composite, "severity": severity,
            "predictedAttack": cat, "detectionMethods": methods,
            "primaryReasons": reasons or ["Multivariate statistical deviation from baseline."],
            "metrics": {
                "zScore": float(max_abs_z[i]), "zScoreFlagged": bool(z_flag[i]),
                "iqrFlagged": bool(iqr_flag[i]), "iqrValue": pq["iqr"],
                "iqrLowerBound": pq["lowerBound"], "iqrUpperBound": pq["upperBound"],
                "movingAvgDeviation": float(ma_dev[i]), "movingAvgFlagged": bool(ma_flag[i]),
                "entropyValue": float(ent_local[i]), "entropyDeviation": float(ent_dev[i]),
                "entropyFlagged": bool(ent_flag[i]), "isolationForestScore": float(if_score[i]),
                "isolationForestFlagged": bool(if_flag[i]), "compositeScore": composite,
            },
            "status": "Under Investigation", "updatedAt": datetime.now(timezone.utc).isoformat(),
            "_index": i,
        })

    anomalies.sort(key=lambda a: a["compositeScore"], reverse=True)
    n_anom = len(anomalies)
    risk = min(100, round((sev_counts["CRITICAL"] * 25 + sev_counts["HIGH"] * 12 + sev_counts["MEDIUM"] * 4
                           + sev_counts["LOW"]) / max(1.0, n * 0.15)))

    # time series (<= 60 buckets, file order)
    buckets = min(60, n)
    edges = np.linspace(0, n, buckets + 1).astype(int)
    series = []
    for b in range(buckets):
        lo, hi = edges[b], edges[b + 1]
        if hi <= lo:
            continue
        fl = flagged[lo:hi]
        series.append({"time": ts_iso[lo][11:19], "packetCount": float(pc[lo:hi].sum()),
                       "bytesKB": round(float(tb[lo:hi].sum()) / 1024, 1),
                       "isAnomaly": float(pc[lo:hi][fl].sum()) if fl.any() else None})

    summary = {
        "datasetId": dataset_id, "datasetName": dataset_name,
        "analyzedAt": datetime.now(timezone.utc).isoformat(),
        "totalRecords": n, "totalPackets": float(pc.sum()),
        "normalRecords": n - n_anom, "anomalousRecords": n_anom,
        "normalPercentage": round((n - n_anom) / n * 100, 2),
        "suspiciousPercentage": round(n_anom / n * 100, 2), "anomalyRate": round(n_anom / n * 100, 2),
        "criticalCount": sev_counts["CRITICAL"], "highCount": sev_counts["HIGH"],
        "mediumCount": sev_counts["MEDIUM"], "lowCount": sev_counts["LOW"],
        "networkRiskScore": int(risk), "categoryCounts": cat_counts, "protocolCounts": proto_counts,
        "records": records_out,
        "anomalies": [{k: v for k, v in a.items() if k != "_index"} for a in anomalies[:MAX_ANOMALIES_RETURNED]],
        "preprocessing": preprocessing, "timeSeries": series,
        "methodCounts": {"zScore": int(z_flag.sum()), "iqr": int(iqr_flag.sum()), "movingAvg": int(ma_flag.sum()),
                         "isolationForest": int(if_flag.sum()),
                         "entropyValue": float(ent_local.mean()) if has_ips else None},
        "featureBaselines": base,
    }
    return {
        "summary": summary, "anomalies": anomalies, "n": n,
        "flags": {"z_score": z_flag, "iqr": iqr_flag, "moving_average": ma_flag,
                  "isolation_forest": if_flag, "combined": flagged},
        "scores": {"z_score": np.minimum(100, max_abs_z / (2 * st.z_score_threshold) * 100),
                   "iqr": np.where(iqr_flag, np.minimum(100, 50 + (iqr_p.astype(int) + iqr_b + iqr_f) * 16.5), 10),
                   "moving_average": np.minimum(100, np.abs(ma_dev) * 45),
                   "isolation_forest": np.clip(if_score * 100, 0, 100)},
        "raw": {"z_score": max_abs_z, "iqr": pc, "moving_average": ma_dev, "isolation_forest": if_score},
        "thresholds": {"z_score": st.z_score_threshold, "iqr": base["packetCount"]["upperBound"],
                       "isolation_forest": if_threshold},
        "composite": comp_all, "df": df, "has_labels": has_labels, "has_ips": has_ips,
        "settings": st,
    }


# --------------------------------------------------------------------------
# Evaluation & ensemble views
# --------------------------------------------------------------------------
def _metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    tp = int((pred & truth).sum()); fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum()); tn = int((~pred & ~truth).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {
        "accuracy": round((tp + tn) / max(1, len(truth)) * 100, 2), "precision": round(p * 100, 2),
        "recall": round(r * 100, 2), "f1Score": round(f1 * 100, 2),
        "falsePositiveRate": round(fp / (fp + tn) * 100, 2) if fp + tn else 0.0,
        "confusionMatrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "supportAttack": int(truth.sum()), "supportNormal": int((~truth).sum()),
    }


def evaluation(res: dict[str, Any], dataset_id: str, dataset_name: str) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    if not res["has_labels"]:
        return {"hasLabels": False, "message": "Evaluation unavailable: upload a CSV with a Label column.",
                "datasetId": dataset_id, "datasetName": dataset_name, "testSampleCount": 0,
                "evaluatedAt": now, "algorithms": []}
    labels = res["df"]["label"].astype(str).str.strip().str.lower()
    truth = (~labels.isin(BENIGN_LABELS)).to_numpy()
    names = {"z_score": "Modified Z-Score (median/MAD)", "iqr": "IQR Outlier Detection",
             "moving_average": "Moving Average Detection", "isolation_forest": "Isolation Forest ML",
             "combined": "Combined Pipeline (all signals)"}
    algos = []
    for key in ["z_score", "iqr", "moving_average", "isolation_forest", "combined"]:
        m = _metrics(res["flags"][key], truth)
        algos.append({"id": key, "name": names[key], "available": True, **m})
    return {"hasLabels": True, "message": "Metrics computed server-side against the uploaded Label column.",
            "datasetId": dataset_id, "datasetName": dataset_name, "testSampleCount": int(len(truth)),
            "evaluatedAt": now, "labelColumn": "Label", "positiveLabel": "Any non-benign label",
            "algorithms": algos}


def ensemble(res: dict[str, Any], dataset_id: str, dataset_name: str, limit: int = 1500) -> dict[str, Any]:
    df, n = res["df"], res["n"]
    meta = [("z_score", "Z-Score Anomaly Detection"), ("iqr", "IQR Outlier Detection"),
            ("moving_average", "Moving Average Detection"), ("isolation_forest", "Isolation Forest ML")]
    votes = np.sum([res["flags"][k] for k, _ in meta], axis=0)
    comp = res["composite"]
    st = res["settings"]
    cls = np.where((comp >= st.high) | (votes >= 3), "High Risk",
                   np.where((comp >= st.medium) | (votes >= 1), "Suspicious", "Normal"))
    summaries = []
    for k, name in meta:
        f = res["flags"][k]
        sc = res["scores"][k]
        summaries.append({"algorithm": k, "name": name, "flaggedCount": int(f.sum()), "totalFlows": n,
                          "anomalyRate": round(float(f.mean()) * 100, 2),
                          "averageScore": round(float(sc.mean()), 2), "maxScore": round(float(sc.max()), 2)})

    def label(v: int) -> str:
        return ("Consensus Normal (0/4)" if v == 0 else "Unanimous (4/4)" if v == 4 else
                "Majority (3/4)" if v == 3 else "Split (2/4)" if v == 2 else "Minority (1/4)")

    anomaly_ids = {a["_index"]: a for a in res["anomalies"]}
    order = np.argsort(-comp)[:limit]
    flows = []
    for i in order:
        i = int(i)
        row = df.iloc[i]
        decisions = [{"algorithm": k, "name": name, "prediction": "Anomaly" if res["flags"][k][i] else "Normal",
                      "flagged": bool(res["flags"][k][i]),
                      "anomalyScore": round(float(res["scores"][k][i]), 1),
                      "rawScore": round(float(res["raw"][k][i]), 3),
                      "threshold": round(float(res["thresholds"].get(k, 0)), 3) if k in res["thresholds"] else None}
                     for k, name in meta]
        anom = anomaly_ids.get(i)
        flows.append({
            "flowId": f"row-{int(row['rowIndex'])}", "timestamp": row["timestamp"].isoformat(),
            "sourceIp": row["sourceIp"], "destIp": row["destIp"], "sourcePort": int(row["sourcePort"]),
            "destPort": int(row["destPort"]), "protocol": str(row["protocol"]),
            "features": {"packetCount": float(row["packetCount"]), "totalBytes": float(row["totalBytes"]),
                         "connectionFrequency": float(row["connectionFrequency"]),
                         "flowDuration": float(row["flowDuration"]), "packetLength": float(row["packetLength"]),
                         "fwdPackets": int(row["fwdPackets"]), "bwdPackets": int(row["bwdPackets"])},
            "algorithmDecisions": decisions, "votesForAnomaly": int(votes[i]), "totalAlgorithms": 4,
            "agreementPercentage": round(float(votes[i]) / 4 * 100, 1), "agreementStatus": label(int(votes[i])),
            "combinedEnsembleScore": float(comp[i]), "finalClassification": str(cls[i]),
            "riskCategory": anom["predictedAttack"] if anom else "Normal Traffic",
            "severity": anom["severity"] if anom else "LOW",
            "explanation": anom["primaryReasons"] if anom else ["No significant statistical deviation."],
            "linkedThreatId": anom["id"] if anom else None,
        })
    return {"datasetId": dataset_id, "datasetName": dataset_name,
            "analyzedAt": datetime.now(timezone.utc).isoformat(), "totalFlows": n,
            "normalCount": int((cls == "Normal").sum()), "suspiciousCount": int((cls == "Suspicious").sum()),
            "highRiskCount": int((cls == "High Risk").sum()),
            "averageEnsembleScore": round(float(comp.mean()), 2),
            "unanimousAgreementCount": int((votes == 4).sum()),
            "majorityAgreementCount": int((votes >= 3).sum()),
            "algorithmSummaries": summaries, "flowDecisions": flows,
            "message": f"Showing the {len(flows)} highest-scoring of {n} flows."}


def new_dataset_id() -> str:
    return "ds-" + uuid.uuid4().hex[:10]

"""A3 - Aplicacion web para probar el clasificador de mociones.

Entrena el RandomForest al arrancar y expone:
  GET /        -> formulario HTML con muestras pre-cargadas
  POST /predict -> JSON con prediccion y probabilidades

Escucha solo en localhost (127.0.0.1) en el puerto 8000.
"""
import json, os, re, random
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

try:
    import cache as cl_cache
    import courtlistener as cl_api
except Exception as _imp_err:
    cl_cache = None
    cl_api = None
    print(f"[boot] modulos cache/courtlistener no disponibles: {_imp_err}", flush=True)

A3 = os.environ.get("A3_DATA_DIR", os.path.dirname(os.path.abspath(__file__)))
SEED = 42
CLASSES = ["ifp", "counsel", "extension"]

def normalize_judge(s):
    if not s: return ""
    if "," in s:
        parts = [p.strip() for p in s.split(",")]
        if len(parts) >= 2: s = parts[1] + " " + parts[0]
    x = s.lower().strip()
    x = re.sub(r"[.,]", " ", x)
    x = re.sub(r"\b(judge|district|magistrate|chief|hon|honorable|sr|jr|usdj|usmj)\b", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    tokens = x.split()
    return tokens[-1] if tokens else ""

def count_dockets(s):
    if not s: return 0
    parts = re.split(r"[,;]|\sand\s", s)
    return len([p for p in parts if p.strip()])

def is_pro_se(a):
    if not a: return False
    return bool(re.search(r"\bpro\s*se\b", a, re.IGNORECASE))

def extract_row(r, label=None):
    op0 = (r.get("opinions") or [{}])[0]
    snippet = op0.get("snippet") or ""
    citation = r.get("citation") or []
    cites = op0.get("cites") or []
    docket = r.get("docketNumber") or ""
    attorney = r.get("attorney") or ""
    date = r.get("dateFiled") or ""
    return {
        "motion_type": label,
        "cluster_id": r.get("cluster_id"),
        "court_id": r.get("court_id") or "unknown",
        "year": int(date.split("-")[0]) if date else None,
        "month": int(date.split("-")[1]) if date else None,
        "judge_norm": normalize_judge(r.get("judge") or ""),
        "citeCount": r.get("citeCount"),
        "n_citation": len(citation),
        "n_opcites": len(cites),
        "len_snippet": len(snippet),
        "len_caseName": len(r.get("caseName") or ""),
        "n_dockets": count_dockets(docket),
        "source": r.get("source") or "unknown",
        "is_pro_se": is_pro_se(attorney),
        "is_in_re": (r.get("caseName") or "").lower().startswith("in re"),
        "attorney_len": len(attorney),
        "_caseName_raw": r.get("caseName") or "",
        "_judge_raw": r.get("judge") or "",
        "_attorney_raw": attorney,
        "_date": date,
        "_docket": docket,
    }

print("[boot] cargando datos crudos.", flush=True)
raw_rows = []
RAW_BY_ID = {}
RAW_RECORDS = []
for c in CLASSES:
    data = json.load(open(f"{A3}/full_{c}.json"))
    if isinstance(data, dict) and "results" in data: data = data["results"]
    for r in data:
        raw_rows.append(extract_row(r, c))
        cid = r.get("cluster_id")
        if cid is not None and cid not in RAW_BY_ID:
            RAW_BY_ID[int(cid)] = r
            RAW_RECORDS.append(r)
df_raw = pd.DataFrame(raw_rows)
df_raw = df_raw.drop_duplicates(subset=["cluster_id"], keep="first")
g = df_raw.groupby("cluster_id")["motion_type"].nunique()
multi = g[g > 1].index.tolist()
if multi: df_raw = df_raw[~df_raw["cluster_id"].isin(multi)]

minc = df_raw["motion_type"].value_counts().min()
parts = []
for cls, n in df_raw["motion_type"].value_counts().items():
    parts.append(df_raw[df_raw["motion_type"]==cls].sample(n=minc, random_state=42))
df_bal = pd.concat(parts, ignore_index=True)

meta_cols = ["_caseName_raw","_judge_raw","_attorney_raw","_date","_docket","cluster_id"]
df = df_bal.drop(columns=meta_cols).copy()

for col in ["citeCount","n_opcites","len_caseName","n_dockets"]:
    df[col] = df[col].fillna(0).clip(lower=0).astype(float)
    df[col + "_log"] = np.log1p(df[col]); df = df.drop(columns=[col])

NUMERICAS = [c for c in df.columns if c.endswith("_log")] + ["n_citation","len_snippet","attorney_len","year","month"]
IMPUTE_VALS = {}
for col in NUMERICAS:
    if col not in df.columns: continue
    s = df[col].dropna().astype(float)
    val = s.mean() if abs(s.skew()) < 0.5 else s.median()
    IMPUTE_VALS[col] = float(val); df[col] = df[col].fillna(val)
CAP_BOUNDS = {}
for col in NUMERICAS:
    if col not in df.columns: continue
    s = df[col].astype(float)
    q1, q3 = s.quantile(0.25), s.quantile(0.75); iqr = q3 - q1
    CAP_BOUNDS[col] = (float(q1 - 1.5*iqr), float(q3 + 1.5*iqr))
    df[col] = s.clip(*CAP_BOUNDS[col])

FREQ_JUDGE = df["judge_norm"].value_counts().to_dict()
df["judge_freq"] = df["judge_norm"].map(FREQ_JUDGE).fillna(0).astype(int)
df = df.drop(columns=["judge_norm"])
df = pd.get_dummies(df, columns=["court_id","source"], prefix=["court","src"], drop_first=False)
for col in ["is_pro_se","is_in_re"]: df[col] = df[col].astype(int)

LE = LabelEncoder()
y = LE.fit_transform(df["motion_type"])
X = df.drop(columns=["motion_type"]).astype(float)
FEATURE_COLS = list(X.columns)

print(f"[boot] entrenando RF con {len(X)} filas y {len(FEATURE_COLS)} features.", flush=True)
RF = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1, max_depth=12, min_samples_leaf=3)
RF.fit(X.values, y)
CLASS_ORDER = list(LE.classes_)
print(f"[boot] listo. Clases: {CLASS_ORDER}", flush=True)

def featurize(inp):
    """Aplica el mismo pipeline a un input dict del usuario."""
    row = {
        "court_id": inp.get("court_id") or "unknown",
        "year": int(inp.get("year")) if inp.get("year") else None,
        "month": int(inp.get("month")) if inp.get("month") else None,
        "judge_norm": normalize_judge(inp.get("judge") or ""),
        "citeCount": float(inp.get("citeCount") or 0),
        "n_citation": int(inp.get("n_citation") or 0),
        "n_opcites": float(inp.get("n_opcites") or 0),
        "len_snippet": int(inp.get("len_snippet") or 0),
        "len_caseName": len(inp.get("caseName") or ""),
        "n_dockets": count_dockets(inp.get("docket") or ""),
        "source": inp.get("source") or "unknown",
        "is_pro_se": is_pro_se(inp.get("attorney") or ""),
        "is_in_re": (inp.get("caseName") or "").lower().startswith("in re"),
        "attorney_len": len(inp.get("attorney") or ""),
    }
    d = pd.DataFrame([row])
    for col in ["citeCount","n_opcites","len_caseName","n_dockets"]:
        d[col] = d[col].fillna(0).clip(lower=0).astype(float)
        d[col + "_log"] = np.log1p(d[col]); d = d.drop(columns=[col])
    for col in NUMERICAS:
        if col not in d.columns: continue
        d[col] = d[col].fillna(IMPUTE_VALS.get(col, 0)).astype(float)
        lo, hi = CAP_BOUNDS.get(col, (None, None))
        if lo is not None: d[col] = d[col].clip(lo, hi)
    d["judge_freq"] = d["judge_norm"].map(FREQ_JUDGE).fillna(0).astype(int)
    d = d.drop(columns=["judge_norm"])
    d = pd.get_dummies(d, columns=["court_id","source"], prefix=["court","src"], drop_first=False)
    for col in ["is_pro_se","is_in_re"]: d[col] = d[col].astype(int)
    for c in FEATURE_COLS:
        if c not in d.columns: d[c] = 0
    d = d[FEATURE_COLS].astype(float)
    return d.values, row

def featurize_from_raw(raw):
    """Toma un JSON crudo tipo CourtListener y arma el feature vector."""
    row = extract_row(raw)
    feats = {k: v for k, v in row.items() if not k.startswith("_") and k not in ("motion_type", "cluster_id")}
    d = pd.DataFrame([feats])
    for col in ["citeCount","n_opcites","len_caseName","n_dockets"]:
        d[col] = d[col].fillna(0).clip(lower=0).astype(float)
        d[col + "_log"] = np.log1p(d[col]); d = d.drop(columns=[col])
    for col in NUMERICAS:
        if col not in d.columns: continue
        d[col] = d[col].fillna(IMPUTE_VALS.get(col, 0)).astype(float)
        lo, hi = CAP_BOUNDS.get(col, (None, None))
        if lo is not None: d[col] = d[col].clip(lo, hi)
    d["judge_freq"] = d["judge_norm"].map(FREQ_JUDGE).fillna(0).astype(int)
    d = d.drop(columns=["judge_norm"])
    d = pd.get_dummies(d, columns=["court_id","source"], prefix=["court","src"], drop_first=False)
    for col in ["is_pro_se","is_in_re"]: d[col] = d[col].astype(int)
    for c in FEATURE_COLS:
        if c not in d.columns: d[c] = 0
    d = d[FEATURE_COLS].astype(float)
    summary = {
        "caseName": row["_caseName_raw"],
        "docket": row["_docket"],
        "court_id": row["court_id"],
        "date": row["_date"],
        "judge": row["_judge_raw"],
        "is_pro_se": bool(row["is_pro_se"]),
        "attorney_len": int(row["attorney_len"]),
    }
    return d.values, summary

COURT_HUMAN = {
    "cacd": "C.D. Cal.", "cand": "N.D. Cal.",
    "flmd": "M.D. Fla.", "flsd": "S.D. Fla.",
    "ilnd": "N.D. Ill.", "njd": "D.N.J.",
    "nyed": "E.D.N.Y.", "nysd": "S.D.N.Y.",
    "paed": "E.D. Pa.", "txsd": "S.D. Tex.",
}
COURT_TOKENS = {
    "nysd": ["nysd","sdny","s.d.n.y.","s.d. n.y.","southern district new york"],
    "nyed": ["nyed","edny","e.d.n.y.","e.d. n.y.","eastern district new york"],
    "ilnd": ["ilnd","ndill","n.d. ill.","northern district illinois"],
    "paed": ["paed","edpa","e.d. pa.","eastern district pennsylvania"],
    "cacd": ["cacd","cdcal","c.d. cal.","central district california"],
    "cand": ["cand","ndcal","n.d. cal.","northern district california"],
    "flmd": ["flmd","mdfla","m.d. fla.","middle district florida"],
    "flsd": ["flsd","sdfla","s.d. fla.","southern district florida"],
    "njd":  ["njd","dnj","d.n.j.","district new jersey"],
    "txsd": ["txsd","sdtex","s.d. tex.","southern district texas"],
}

URL_RX = re.compile(r"courtlistener\.com/(?:docket|opinion|cluster)/(\d+)", re.IGNORECASE)
NUM_RX = re.compile(r"^\d{4,9}$")
DOCKET_RX = re.compile(r"\b(?:\d{1,2}:)?\d{2}[\-\s]?(?:cv|cr|mc|mj|bk|md|po)[\-\s]?\d{1,6}\b", re.IGNORECASE)
OLD_DOCKET_RX = re.compile(r"\b\d{2}\s+[A-Z]\.?\s+\d{2,6}\b")  # 92 C 5381 (Northern Illinois old style)
CIV_ACTION_RX = re.compile(r"\bciv(?:il)?\.?\s*(?:a(?:ction)?\.?\s*)?(?:nos?\.?\s*)?\d{2,4}[\-\s]?(?:cv[\-\s]?)?\d{2,6}\b", re.IGNORECASE)
CASE_VS_RX = re.compile(r"\bv\.?\b", re.IGNORECASE)
IN_RE_RX = re.compile(r"^\s*in\s+re\b", re.IGNORECASE)
JUDGE_RX = re.compile(r"^\s*(?:juez|judge|hon\.?|honorable|the\s+honorable)\s+([A-Za-z'\-]+)", re.IGNORECASE)
YEAR_RX = re.compile(r"\b(19|20)\d{2}\b")
SINGLE_PARTY_RX = re.compile(r"^[A-Z][A-Za-z\-'\.]{1,}(?:\s+[A-Z][A-Za-z\-'\.]+){0,2}$")

def normalize_docket(s):
    if not s: return ""
    x = s.lower()
    x = re.sub(r"[^a-z0-9]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def detect_court_token(q):
    ql = " " + q.lower() + " "
    for code, toks in COURT_TOKENS.items():
        for t in toks:
            if " " + t + " " in ql or ql.strip() == t:
                return [code]
    qs = q.lower().strip()
    if 2 <= len(qs) <= 5 and re.fullmatch(r"[a-z]+", qs):
        prefs = [code for code in COURT_TOKENS.keys() if code.startswith(qs)]
        if prefs:
            return prefs
    return None

def route_query(q):
    if not q or len(q.strip()) < 2:
        return {"mode": "empty", "params": {}, "label": "Consulta vacia"}
    q = q.strip()
    m = URL_RX.search(q)
    if m:
        return {"mode": "url", "params": {"id": int(m.group(1))}, "label": "URL de CourtListener"}
    if NUM_RX.match(q):
        return {"mode": "id", "params": {"id": int(q)}, "label": "ID numerico"}
    if DOCKET_RX.search(q) or OLD_DOCKET_RX.search(q) or CIV_ACTION_RX.search(q):
        return {"mode": "docket", "params": {"docket": q}, "label": "Numero de docket"}
    if IN_RE_RX.search(q):
        return {"mode": "case_name", "params": {"name": q}, "label": "Nombre de caso (In re)"}
    if CASE_VS_RX.search(q):
        return {"mode": "case_name", "params": {"name": q}, "label": "Nombre de caso"}
    jm = JUDGE_RX.match(q)
    if jm:
        return {"mode": "judge", "params": {"judge": jm.group(1)}, "label": "Juez"}
    ct = detect_court_token(q)
    if ct and len(q.split()) <= 4:
        label = "Corte" if len(ct) == 1 else "Corte (varias coincidencias)"
        return {"mode": "court", "params": {"courts": ct, "court": ct[0], "free": q}, "label": label}
    if SINGLE_PARTY_RX.match(q):
        return {"mode": "party", "params": {"name": q}, "label": "Parte"}
    return {"mode": "free", "params": {"q": q}, "label": "Texto libre"}

def score_record(rec, route, q):
    s = 0.0
    name = (rec.get("caseName") or "").lower()
    docket = (rec.get("docketNumber") or "")
    judge = (rec.get("judge") or "").lower()
    court = rec.get("court_id") or ""
    cid = rec.get("cluster_id")
    ql = q.lower().strip()
    mode = route["mode"]
    if mode == "url" or mode == "id":
        if cid == route["params"].get("id"): return 1000.0
        return 0.0
    if mode == "docket":
        nd_q = normalize_docket(q)
        nd_r = normalize_docket(docket)
        if not nd_q: return 0.0
        if nd_q in nd_r: s += 50
        toks_q = nd_q.split()
        toks_r = nd_r.split()
        common = sum(1 for t in toks_q if t in toks_r)
        s += common * 5
        return s
    if mode == "case_name":
        toks = [t for t in re.split(r"\W+", ql) if t and t not in ("v","vs","in","re","the","of","and","or")]
        for t in toks:
            if t in name: s += 8
        if ql in name: s += 30
        return s
    if mode == "judge":
        jq = route["params"]["judge"].lower()
        if jq and jq in judge: s += 40
        return s
    if mode == "court":
        courts = route["params"].get("courts") or [route["params"].get("court")]
        if court in courts: s += 20
        for t in (route["params"].get("free") or "").lower().split():
            if t in name: s += 3
        return s
    if mode == "party":
        for t in ql.split():
            if t in name: s += 6
        return s
    if mode == "free":
        STOP = {"case","cases","court","order","motion","the","of","and","or","in","to","a","an","for","on","at","by"}
        snippet = (rec.get("snippet") or "").lower()
        name_words = set(re.findall(r"[a-z]+", name))
        judge_words = set(re.findall(r"[a-z]+", judge))
        toks = [x for x in re.split(r"\W+", ql) if len(x) >= 3 and x not in STOP]
        if not toks: return 0.0
        if ql in name: s += 25
        elif ql in judge: s += 20
        elif len(ql) >= 5 and ql in snippet: s += 12
        for t in toks:
            if t in name_words: s += 6
            if t in judge_words: s += 6
            if t == court: s += 10
        ym = YEAR_RX.search(q)
        if ym and (rec.get("dateFiled") or "").startswith(ym.group(0)): s += 8
        return s if s >= 8 else 0.0
    return 0.0

def search_local(q, limit=5):
    route = route_query(q)
    if route["mode"] == "empty":
        return route, []
    if route["mode"] in ("url","id"):
        target = route["params"]["id"]
        rec = RAW_BY_ID.get(target)
        if rec: return route, [rec]
        return route, []
    scored = []
    for rec in RAW_RECORDS:
        sc = score_record(rec, route, q)
        if sc > 0: scored.append((sc, rec))
    scored.sort(key=lambda x: x[0], reverse=True)
    return route, [r for _, r in scored[:limit]]

def candidate_card(rec):
    op0 = (rec.get("opinions") or [{}])[0]
    snippet = (op0.get("snippet") or "")[:160]
    return {
        "id": int(rec.get("cluster_id") or 0),
        "caseName": rec.get("caseName") or "",
        "court_id": rec.get("court_id") or "",
        "court_human": COURT_HUMAN.get(rec.get("court_id") or "", rec.get("court_id") or ""),
        "date": rec.get("dateFiled") or "",
        "docket": rec.get("docketNumber") or "",
        "judge": rec.get("judge") or "",
        "snippet": snippet,
    }

app = Flask(__name__)

HTML = """
<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Clasificador de mociones pro se</title>
<style>
:root {
  --bg: #faf9f6;
  --fg: #0f0f0f;
  --muted: #6b6b6b;
  --line: #e6e4de;
  --card: #ffffff;
  --accent: #0f0f0f;
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body {
  font-family: -apple-system, "Inter", "Helvetica Neue", system-ui, sans-serif;
  background: var(--bg); color: var(--fg);
  font-size: clamp(17px, 2.4vw, 19px);
  line-height: 1.55;
  -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 720px; margin: 0 auto; padding: clamp(2.5rem, 9vw, 5rem) clamp(1.25rem, 5vw, 2.5rem) 4rem; }
header { margin-bottom: clamp(2.5rem, 8vw, 4rem); }
.eyebrow {
  font-size: .72rem; letter-spacing: .22em; text-transform: uppercase;
  color: var(--muted); margin-bottom: 1.5rem;
}
h1 {
  font-family: Georgia, "Times New Roman", serif;
  font-weight: 500;
  font-size: clamp(2.4rem, 8vw, 4rem);
  line-height: 1.02; letter-spacing: -.025em;
  margin: 0 0 1.5rem;
}
.lead {
  font-size: clamp(1.1rem, 2.8vw, 1.35rem);
  color: #2a2a2a; max-width: 32ch; margin: 0;
}
section { margin-top: clamp(3rem, 8vw, 4.5rem); }
section h2 {
  font-family: Georgia, serif; font-weight: 500;
  font-size: clamp(1.5rem, 4.5vw, 2rem);
  letter-spacing: -.015em; margin: 0 0 1.25rem;
}
section p { color: #2a2a2a; margin: 0 0 1rem; max-width: 60ch; }
.cta {
  display: inline-flex; align-items: center; justify-content: center; gap: .6rem;
  background: var(--accent); color: #fff;
  padding: 1.1rem 1.6rem; border-radius: 999px;
  font-size: 1.02rem; font-weight: 500; text-decoration: none;
  border: 0; cursor: pointer;
  transition: transform .12s ease, opacity .12s ease;
}
.cta:hover { opacity: .88; }
.cta:active { transform: scale(.98); }
.cta.ghost { background: transparent; color: var(--fg); border: 1px solid var(--fg); }
.cta.full { width: 100%; padding: 1.2rem; font-size: 1.05rem; }
.coming {
  font-size: .68rem; letter-spacing: .15em; text-transform: uppercase;
  color: var(--muted); font-weight: 500;
}
.dl { margin: 1.5rem 0 2.5rem; }
.label {
  display: block; font-size: .72rem; letter-spacing: .14em; text-transform: uppercase;
  color: var(--muted); margin: 1.5rem 0 .5rem;
}
input, select, textarea {
  width: 100%; font: inherit; font-size: 1.02rem;
  background: var(--card); color: var(--fg);
  border: 1px solid var(--line); border-radius: 10px;
  padding: .9rem 1rem;
  -webkit-appearance: none; appearance: none;
  transition: border-color .12s ease;
}
input:focus, select:focus, textarea:focus { outline: 0; border-color: var(--fg); }
textarea { resize: vertical; min-height: 96px; font-family: inherit; }
.row { display: grid; grid-template-columns: 1fr; gap: 1rem; }
@media (min-width: 640px) {
  .row.cols-2 { grid-template-columns: 1fr 1fr; }
  .row.cols-4 { grid-template-columns: 1fr 1fr; }
}
@media (min-width: 880px) {
  .row.cols-4 { grid-template-columns: 1fr 1fr 1fr 1fr; }
}
.submit { margin-top: 2rem; }
.result {
  display: none; margin-top: 2.5rem;
  padding: clamp(1.5rem, 5vw, 2rem);
  background: var(--card); border: 1px solid var(--line); border-radius: 16px;
}
.result.show { display: block; }
.pred-label {
  font-size: .72rem; letter-spacing: .22em; text-transform: uppercase;
  color: var(--muted); margin-bottom: .6rem;
}
.pred {
  font-family: Georgia, serif; font-weight: 500;
  font-size: clamp(1.9rem, 5.5vw, 2.6rem);
  letter-spacing: -.015em; line-height: 1.1;
}
.prob { margin-top: 1.75rem; }
.probrow {
  display: grid; grid-template-columns: 1fr 60px;
  align-items: center; gap: .75rem .9rem;
  margin: .9rem 0; font-size: .98rem;
}
.probrow > .pn { grid-column: 1 / -1; color: #2a2a2a; }
.probbar { height: 6px; background: var(--line); border-radius: 999px; overflow: hidden; }
.probval { text-align: right; color: var(--muted); font-variant-numeric: tabular-nums; font-size: .92rem; }
.probfill { height: 100%; background: var(--accent); transition: width .5s ease; }
.hint { font-size: .9rem; color: var(--muted); margin: .6rem 0 0; max-width: 60ch; }
.searchbox { position: relative; }
.searchbox input { padding-right: 2.4rem; }
.dot {
  position: absolute; right: 1rem; top: 50%; transform: translateY(-50%);
  width: 8px; height: 8px; border-radius: 999px; background: transparent;
  transition: background-color .25s ease;
}
.dot.busy { background: var(--muted); animation: pulse 1s ease-in-out infinite; }
.dot.ok { background: #1f6f3a; }
.dot.warn { background: #b06000; }
.dot.err { background: #8a1f1f; }
@keyframes pulse { 0%,100% { opacity: .35 } 50% { opacity: 1 } }
.echo { font-size: .85rem; color: var(--muted); margin: .6rem 0 0; min-height: 1.2em; letter-spacing: .03em; }
.echo b { color: var(--fg); font-weight: 500; }
.chips { margin: 1rem 0 1.75rem; display: flex; flex-wrap: wrap; gap: .5rem .8rem; align-items: center; }
.chiplbl { font-size: .72rem; letter-spacing: .14em; text-transform: uppercase; color: var(--muted); }
.chip {
  font: inherit; font-size: .88rem;
  background: transparent; color: var(--fg);
  border: 0; padding: .35rem .6rem; border-radius: 999px;
  text-decoration: underline; text-decoration-color: var(--line);
  text-underline-offset: 4px; cursor: pointer;
}
.chip:hover { text-decoration-color: var(--fg); }
.status { margin-top: 1rem; font-size: .9rem; color: var(--muted); min-height: 1.2em; }
.status.bad { color: #8a1f1f; }
.candidates { margin-top: 1.5rem; display: grid; gap: 1rem; }
.cand {
  background: var(--card); border: 1px solid var(--line); border-radius: 14px;
  padding: 1.1rem 1.25rem; cursor: pointer; text-align: left; width: 100%;
  font: inherit; transition: border-color .12s ease, transform .12s ease;
}
.cand:hover { border-color: var(--fg); }
.cand:active { transform: scale(.995); }
.cand .name { font-family: Georgia, serif; font-size: 1.15rem; line-height: 1.25; margin-bottom: .35rem; }
.cand .meta { font-size: .82rem; color: var(--muted); display: flex; flex-wrap: wrap; gap: .25rem .85rem; }
.cand .meta b { color: var(--fg); font-weight: 500; }
.cand .snip { font-size: .86rem; color: #2a2a2a; margin-top: .55rem; line-height: 1.45; }
.confirm {
  display: none; margin-top: 1.5rem;
  background: var(--card); border: 1px solid var(--fg); border-radius: 16px;
  padding: 1.5rem;
}
.confirm.show { display: block; }
.confirm .lbl { font-size: .72rem; letter-spacing: .22em; text-transform: uppercase; color: var(--muted); margin-bottom: .8rem; }
.confirm .name { font-family: Georgia, serif; font-size: clamp(1.4rem, 4vw, 1.7rem); line-height: 1.15; margin-bottom: .9rem; }
.confirm .row2 { display: grid; grid-template-columns: 1fr; gap: .5rem; font-size: .92rem; margin-bottom: 1.25rem; }
.confirm .row2 span.k { color: var(--muted); text-transform: uppercase; letter-spacing: .1em; font-size: .7rem; }
.confirm .actions { display: flex; gap: .8rem; flex-wrap: wrap; }
.confirm .actions button { flex: 1 1 200px; }
.cta.outline { background: transparent; color: var(--fg); border: 1px solid var(--line); }
.cta.outline:hover { border-color: var(--fg); opacity: 1; }
.sugs { margin-top: .6rem; font-size: .9rem; color: var(--muted); }
.sugs button { background: transparent; border: 0; color: var(--fg); text-decoration: underline; cursor: pointer; padding: 0; font: inherit; }
.summary {
  margin-top: 1.75rem; padding-top: 1.5rem;
  border-top: 1px solid var(--line);
  display: grid; grid-template-columns: 1fr; gap: .6rem;
  font-size: .94rem;
}
.summary div { display: flex; justify-content: space-between; gap: 1rem; }
.summary .k { color: var(--muted); text-transform: uppercase; letter-spacing: .1em; font-size: .72rem; align-self: center; }
.summary .v { text-align: right; color: var(--fg); }
.error {
  margin-top: 1.5rem; padding: 1rem 1.2rem;
  background: #fff5f5; border: 1px solid #f3c2c2; border-radius: 10px;
  color: #8a1f1f; font-size: .95rem;
}
@media (min-width: 640px) {
  .probrow { grid-template-columns: 180px 1fr 60px; }
  .probrow > .pn { grid-column: 1; }
}
.quota {
  margin-top: .5rem; font-size: .78rem; color: var(--muted); text-align: right;
  letter-spacing: .04em; min-height: 1em;
}
.quota.soft { color: #8a5a00; }
.quota.hard { color: #8a1f1f; }
.cta.online {
  margin-top: .8rem; background: transparent; color: var(--accent);
  border: 1px solid var(--accent);
}
.cta.online:hover { background: var(--accent); color: white; }
.motions {
  display: grid; grid-template-columns: 1fr; gap: 1rem;
  margin: 1.5rem 0 0;
}
@media (min-width: 720px) { .motions { grid-template-columns: repeat(3, 1fr); } }
.motion {
  background: var(--card); border: 1px solid var(--line); border-radius: 14px;
  padding: 1.25rem 1.3rem; position: relative;
}
.motion .tag {
  display: inline-block; font-size: .68rem; letter-spacing: .18em; text-transform: uppercase;
  color: var(--muted); margin-bottom: .55rem;
}
.motion h3 {
  font-family: Georgia, "Times New Roman", serif; font-weight: 500;
  font-size: 1.2rem; line-height: 1.2; margin: 0 0 .35rem; letter-spacing: -.01em;
}
.motion h3 em { font-style: italic; color: var(--muted); font-weight: 400; font-size: .92rem; }
.motion p { margin: .4rem 0 0; font-size: .95rem; color: #2a2a2a; line-height: 1.45; }
.motion .plain { font-weight: 500; color: var(--fg); }
.motion::before {
  content: ""; position: absolute; left: 1.3rem; top: 0;
  width: 28px; height: 3px; background: var(--accent); border-radius: 0 0 3px 3px;
}
.docnav {
  margin-top: 1.5rem; font-size: .9rem;
}
.docnav a { color: var(--fg); text-decoration: underline; text-underline-offset: 4px; }
footer {
  margin-top: 5rem; padding-top: 2rem; border-top: 1px solid var(--line);
  color: var(--muted); font-size: .82rem; text-align: center; letter-spacing: .04em;
}
</style>
</head>
<body>
<main class="wrap">
  <header>
    <div class="eyebrow">Trabajo Practico 3 - Aplicaciones Reales de Data Science</div>
    <h1>Clasificador de mociones judiciales pro se.</h1>
    <p class="lead">Un modelo de aprendizaje automatico que lee los datos de una orden judicial y predice de que tipo de pedido se trata.</p>
  </header>

  <section>
    <h2>Como funciona</h2>
    <p>Cuando una persona se representa a si misma en la justicia federal de Estados Unidos, suele presentar uno de tres pedidos. Estos tres son las protagonistas del modelo:</p>

    <div class="motions" role="list">
      <article class="motion" role="listitem">
        <span class="tag">Mocion 1</span>
        <h3>IFP <em>in forma pauperis</em></h3>
        <p><span class="plain">Litigar sin pagar costas judiciales.</span></p>
        <p>Pedido de exencion de tasas para quienes no pueden afrontarlas.</p>
      </article>
      <article class="motion" role="listitem">
        <span class="tag">Mocion 2</span>
        <h3>Counsel <em>appointment of counsel</em></h3>
        <p><span class="plain">Pedir que el tribunal designe un abogado</span> al litigante que se representa solo.</p>
      </article>
      <article class="motion" role="listitem">
        <span class="tag">Mocion 3</span>
        <h3>Extension <em>motion for extension of time</em></h3>
        <p><span class="plain">Pedir mas tiempo para responder</span> o cumplir un plazo procesal.</p>
      </article>
    </div>

    <p style="margin-top:1.75rem;">Este modelo analiza el nombre del caso, la corte, el juez, los abogados involucrados y otras seniales del expediente, y estima a cual de esos tres pedidos corresponde la orden. Fue entrenado sobre 667 ordenes federales reales y acierta cerca del 60 por ciento de las veces, mas del doble de lo que lograria el azar.</p>

    <p class="docnav">Documento academico completo: <a href="/informe">Informe del Trabajo Practico 3</a>.</p>
  </section>

  <section>
    <h2>Probar el modelo</h2>
    <p>Busque un caso por numero de docket, nombre, parte, juez o URL de CourtListener. El sistema detecta el formato automaticamente.</p>

    <form id="searchform" novalidate role="search">
      <label class="label" for="q">Buscar caso</label>
      <div class="searchbox">
        <input id="q" name="q" type="search" autocomplete="off" spellcheck="false" placeholder="Numero de docket, nombre del caso, juez o URL." aria-describedby="echo">
        <span class="dot" id="dot" aria-hidden="true"></span>
      </div>
      <p class="echo" id="echo" aria-live="polite">&nbsp;</p>
      <div class="chips" id="examples">
        <span class="chiplbl">Pruebe con:</span>
        <button type="button" class="chip" data-q="15-CV-6684">15-CV-6684</button>
        <button type="button" class="chip" data-q="Jones v. Warden">Jones v. Warden</button>
        <button type="button" class="chip" data-q="Juez Moran">Juez Moran</button>
        <button type="button" class="chip" data-q="nysd">nysd</button>
      </div>
      <button type="submit" class="cta full submit">Buscar</button>
      <p class="quota" id="quota" aria-live="polite">&nbsp;</p>
    </form>

    <div id="status" class="status" aria-live="polite"></div>
    <div id="candidates" class="candidates" aria-live="polite"></div>
    <div id="confirm" class="confirm" aria-live="polite"></div>
    <div id="result" class="result" aria-live="polite"></div>
  </section>

  <footer>Random Forest - scikit-learn - CRISP-DM - 2026</footer>
</main>

<script>
const NICE = { ifp: "Litigar sin costas", counsel: "Designacion de abogado", extension: "Prorroga de plazos" };
const MODELBL = {
  url: "URL de CourtListener", id: "ID numerico", docket: "numero de docket",
  case_name: "nombre de caso", judge: "juez", court: "corte",
  party: "parte", free: "texto libre", empty: ""
};

const $q = document.getElementById("q");
const $dot = document.getElementById("dot");
const $echo = document.getElementById("echo");
const $status = document.getElementById("status");
const $cands = document.getElementById("candidates");
const $confirm = document.getElementById("confirm");
const $result = document.getElementById("result");
const $quota = document.getElementById("quota");

async function refreshQuota() {
  try {
    const r = await fetch("/quota");
    const j = await r.json();
    if (!j.available) { $quota.textContent = ""; return; }
    const txt = "CourtListener en linea: " + j.used + " / " + j.limit + " llamadas hoy.";
    $quota.textContent = txt;
    $quota.className = "quota" + (j.hard_block ? " hard" : (j.soft_block ? " soft" : ""));
  } catch { $quota.textContent = ""; }
}
refreshQuota();
const $form = document.getElementById("searchform");

function esc(s) { return String(s == null ? "" : s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c])); }
function clear(el) { el.innerHTML = ""; el.classList.remove("show"); }
function setDot(cls) { $dot.className = "dot" + (cls ? " " + cls : ""); }

let echoTimer = null;
async function updateEcho() {
  const q = $q.value.trim();
  if (q.length < 2) { $echo.innerHTML = "&nbsp;"; setDot(""); return; }
  setDot("busy");
  try {
    const r = await fetch("/route_preview?q=" + encodeURIComponent(q));
    const j = await r.json();
    const lbl = MODELBL[j.mode] || j.mode;
    $echo.innerHTML = "Interpretado como <b>" + esc(lbl) + "</b>.";
    setDot(j.mode === "free" ? "warn" : "ok");
  } catch { setDot("err"); }
}
$q.addEventListener("input", () => {
  clearTimeout(echoTimer);
  echoTimer = setTimeout(updateEcho, 180);
});

document.querySelectorAll(".chip").forEach(c => {
  c.addEventListener("click", () => { $q.value = c.dataset.q; updateEcho(); $form.requestSubmit(); });
});

$form.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const q = $q.value.trim();
  clear($cands); clear($confirm); clear($result);
  $status.className = "status"; $status.textContent = "";
  if (q.length < 2) { $status.className = "status bad"; $status.textContent = "Escriba al menos dos caracteres."; return; }
  setDot("busy"); $status.textContent = "Buscando.";
  try {
    const r = await fetch("/search?q=" + encodeURIComponent(q));
    const j = await r.json();
    if (!r.ok) {
      setDot("err"); $status.className = "status bad"; $status.textContent = j.error || "Error en la busqueda."; return;
    }
    setDot(j.confidence === "none" ? "warn" : "ok");
    const lbl = MODELBL[j.mode] || j.mode;
    $status.textContent = "Interpretado como " + lbl + ". " + j.candidates.length + " coincidencia" + (j.candidates.length===1?"":"s") + ".";
    if (j.candidates.length === 0) {
      let html = '<div class="error">No se encontraron casos en el indice local.';
      if (j.suggestions && j.suggestions.length) {
        html += '<div class="sugs">Pruebe con: ';
        html += j.suggestions.map(s => '<button type="button" data-sug="'+esc(s)+'">'+esc(s)+'</button>').join(" | ");
        html += '</div>';
      }
      html += '<button class="cta online" type="button" id="goOnline">Consultar en CourtListener</button>';
      html += '</div>';
      $result.innerHTML = html; $result.classList.add("show");
      $result.querySelectorAll("[data-sug]").forEach(b => b.addEventListener("click", () => { $q.value = b.dataset.sug; $form.requestSubmit(); }));
      document.getElementById("goOnline").addEventListener("click", () => searchOnline(q));
      return;
    }
    renderCandidates(j.candidates, j.confidence === "high" && (j.mode === "url" || j.mode === "id"));
  } catch (e) {
    setDot("err"); $status.className = "status bad"; $status.textContent = "Fallo de red: " + e.message;
  }
});

function renderCandidates(cards, autoConfirm) {
  if (autoConfirm) { showConfirm(cards[0]); return; }
  let html = "";
  for (const c of cards) {
    html += '<button class="cand" type="button" data-id="'+c.id+'">';
    html += '<div class="name">'+esc(c.caseName || "(sin nombre)")+'</div>';
    html += '<div class="meta">';
    if (c.court_human) html += '<span><b>'+esc(c.court_human)+'</b></span>';
    if (c.date) html += '<span>'+esc(c.date)+'</span>';
    if (c.judge) html += '<span>Juez '+esc(c.judge)+'</span>';
    if (c.docket) html += '<span>Docket '+esc(c.docket.slice(0,40))+(c.docket.length>40?"…":"")+'</span>';
    html += '</div>';
    if (c.snippet) html += '<div class="snip">'+esc(c.snippet)+'…</div>';
    html += '</button>';
  }
  $cands.innerHTML = html;
  $cands.querySelectorAll(".cand").forEach((b, i) => {
    b.addEventListener("click", () => showConfirm(cards[i]));
  });
}

function showConfirm(c) {
  clear($cands); clear($result);
  let html = '<div class="lbl">Confirme que es este el caso</div>';
  html += '<div class="name">'+esc(c.caseName || "(sin nombre)")+'</div>';
  html += '<div class="row2">';
  const rows = [["Corte", c.court_human || c.court_id], ["Fecha", c.date], ["Juez", c.judge || "(sin dato)"], ["Docket", c.docket || "(sin dato)"]];
  for (const [k, v] of rows) html += '<div><span class="k">'+esc(k)+'</span><br><span>'+esc(v)+'</span></div>';
  html += '</div>';
  html += '<div class="actions">';
  html += '<button class="cta" type="button" id="doClassify">Clasificar este caso</button>';
  html += '<button class="cta outline" type="button" id="doBack">Volver a los resultados</button>';
  html += '</div>';
  $confirm.innerHTML = html; $confirm.classList.add("show");
  $confirm.scrollIntoView({behavior:"smooth", block:"nearest"});
  document.getElementById("doClassify").addEventListener("click", () => classify(c));
  document.getElementById("doBack").addEventListener("click", () => { clear($confirm); $form.requestSubmit(); });
}

async function searchOnline(q) {
  clear($cands); clear($confirm); clear($result);
  setDot("busy"); $status.className = "status"; $status.textContent = "Consultando CourtListener.";
  try {
    const r = await fetch("/api/search?q=" + encodeURIComponent(q));
    const j = await r.json();
    refreshQuota();
    if (r.status === 429) {
      setDot("err"); $status.className = "status bad";
      $status.textContent = "Cuota diaria agotada. Reintente maniana.";
      return;
    }
    if (!r.ok) {
      setDot("err"); $status.className = "status bad";
      $status.textContent = j.error || "CourtListener no respondio.";
      return;
    }
    if (!j.candidates || j.candidates.length === 0) {
      setDot("warn"); $status.textContent = "Sin resultados en CourtListener tampoco.";
      return;
    }
    setDot("ok");
    $status.textContent = "CourtListener devolvio " + j.candidates.length + " resultado" + (j.candidates.length===1?"":"s") + ".";
    renderCandidates(j.candidates, j.candidates.length === 1);
  } catch (e) {
    setDot("err"); $status.className = "status bad"; $status.textContent = "Fallo de red: " + e.message;
  }
}

async function classify(c) {
  const btn = document.getElementById("doClassify");
  const orig = btn.textContent; btn.disabled = true; btn.textContent = "Analizando.";
  try {
    const r = await fetch("/classify_docket", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({cluster_id: c.id})});
    const j = await r.json();
    if (!r.ok || j.error) {
      $result.innerHTML = '<div class="error">'+esc(j.error || "Error desconocido")+'</div>'; $result.classList.add("show"); return;
    }
    const pretty = NICE[j.prediction] || j.prediction;
    let html = '<div class="pred-label">Prediccion</div>';
    html += '<div class="pred">'+esc(pretty)+'</div>';
    html += '<div class="prob">';
    for (const cls of j.classes) {
      const pct = (j.probabilities[cls]*100).toFixed(1);
      html += '<div class="probrow"><div class="pn">'+esc(NICE[cls]||cls)+'</div><div class="probbar"><div class="probfill" style="width:'+pct+'%"></div></div><div class="probval">'+pct+'%</div></div>';
    }
    html += '</div>';
    const s = j.summary || {};
    html += '<div class="summary">';
    const fields = [["Caso", s.caseName], ["Docket", s.docket], ["Corte", s.court_id], ["Fecha", s.date], ["Juez", s.judge || "(sin dato)"], ["Pro se", s.is_pro_se ? "si" : "no"]];
    for (const [k, v] of fields) { if (!v && v !== false && v !== 0) continue; html += '<div><span class="k">'+esc(k)+'</span><span class="v">'+esc(v)+'</span></div>'; }
    html += '</div>';
    $result.innerHTML = html; $result.classList.add("show");
    $result.scrollIntoView({behavior:"smooth", block:"nearest"});
  } catch (e) {
    $result.innerHTML = '<div class="error">No se pudo clasificar: '+esc(e.message)+'</div>'; $result.classList.add("show");
  } finally { btn.disabled = false; btn.textContent = orig; }
}
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)

INFORME_HTML = """
<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Informe TP3 - Clasificador pro se</title>
<style>
:root { --bg:#faf9f6; --fg:#0f0f0f; --muted:#6b6b6b; --line:#e6e4de; --card:#ffffff; --accent:#0f0f0f; }
* { box-sizing: border-box; }
html, body { margin:0; padding:0; }
body { font-family: -apple-system, "Inter", "Helvetica Neue", system-ui, sans-serif;
       background: var(--bg); color: var(--fg); font-size: 17px; line-height: 1.65; }
.wrap { max-width: 780px; margin: 0 auto; padding: 3.5rem 1.5rem 5rem; }
.eyebrow { font-size: .72rem; letter-spacing: .22em; text-transform: uppercase; color: var(--muted); margin-bottom: 1.5rem; }
.back { display: inline-block; margin-bottom: 2rem; color: var(--fg); text-decoration: underline; text-underline-offset: 4px; font-size: .92rem; }
article h1 { font-family: Georgia, serif; font-weight: 500; font-size: 2.6rem; line-height: 1.05; letter-spacing: -.02em; margin: 0 0 1.5rem; }
article h2 { font-family: Georgia, serif; font-weight: 500; font-size: 1.75rem; letter-spacing: -.015em; margin: 3rem 0 1rem; padding-top: 1.25rem; border-top: 1px solid var(--line); }
article h3 { font-family: Georgia, serif; font-weight: 500; font-size: 1.25rem; margin: 2rem 0 .75rem; }
article h4 { font-weight: 600; font-size: 1.02rem; margin: 1.5rem 0 .5rem; }
article p { margin: 0 0 1rem; }
article ul, article ol { margin: 0 0 1.25rem 1.25rem; padding: 0; }
article li { margin: .3rem 0; }
article strong { font-weight: 600; }
article em { font-style: italic; }
article code { background: #f1efe9; padding: .12em .35em; border-radius: 4px; font-size: .88em; }
article pre { background: #f1efe9; padding: 1rem 1.1rem; border-radius: 10px; overflow-x: auto; font-size: .85rem; line-height: 1.5; }
article pre code { background: transparent; padding: 0; }
article blockquote { border-left: 3px solid var(--accent); margin: 1rem 0; padding: .25rem 0 .25rem 1rem; color: #2a2a2a; font-style: italic; }
article hr { border: 0; border-top: 1px solid var(--line); margin: 2.5rem 0; }
article table { width: 100%; border-collapse: collapse; margin: 1rem 0 1.5rem; font-size: .94rem; }
article th, article td { border: 1px solid var(--line); padding: .55rem .75rem; text-align: left; vertical-align: top; }
article th { background: #f1efe9; font-weight: 600; }
article a { color: var(--fg); text-underline-offset: 4px; }
footer { margin-top: 4rem; padding-top: 2rem; border-top: 1px solid var(--line); color: var(--muted); font-size: .82rem; text-align: center; letter-spacing: .04em; }
</style>
</head>
<body>
<main class="wrap">
  <div class="eyebrow">Trabajo Practico 3 - Data Science Real World Applications</div>
  <a href="/" class="back">&larr; Volver al clasificador</a>
  <article>{{ body|safe }}</article>
  <footer>Informe academico - Lopez Perez - 2026</footer>
</main>
</body>
</html>
"""

INFORME_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "A3_Lopez_Perez.md")
FIGURAS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figuras")

@app.route("/informe")
def informe():
    try:
        with open(INFORME_PATH, "r", encoding="utf-8") as f:
            md_text = f.read()
    except FileNotFoundError:
        return "Informe no disponible.", 404
    try:
        import markdown as _md
        body = _md.markdown(md_text, extensions=["tables", "fenced_code", "toc"])
    except Exception:
        body = "<pre>" + (md_text.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")) + "</pre>"
    return render_template_string(INFORME_HTML, body=body)

@app.route("/figuras/<path:fname>")
def figura(fname):
    from flask import send_from_directory
    return send_from_directory(FIGURAS_DIR, fname)

@app.route("/predict", methods=["POST"])
def predict():
    inp = request.get_json() or {}
    X_one, row_used = featurize(inp)
    pred_idx = int(RF.predict(X_one)[0])
    proba = RF.predict_proba(X_one)[0]
    return jsonify({
        "prediction": CLASS_ORDER[pred_idx],
        "classes": CLASS_ORDER,
        "probabilities": {c: float(p) for c,p in zip(CLASS_ORDER, proba)},
        "features_used": {k: (float(v) if isinstance(v,(np.floating,float)) else (int(v) if isinstance(v,(np.integer,bool)) else v)) for k,v in row_used.items()},
    })

@app.route("/predict_raw", methods=["POST"])
def predict_raw():
    payload = request.get_json() or {}
    raw_text = (payload.get("raw") or "").strip()
    if not raw_text:
        return jsonify({"error": "El JSON esta vacio."}), 400
    try:
        raw = json.loads(raw_text)
    except Exception as e:
        return jsonify({"error": f"JSON invalido: {e}"}), 400
    if isinstance(raw, dict) and "results" in raw and isinstance(raw["results"], list) and raw["results"]:
        raw = raw["results"][0]
    if isinstance(raw, list) and raw:
        raw = raw[0]
    if not isinstance(raw, dict):
        return jsonify({"error": "El JSON debe ser un objeto (orden) o una lista de objetos."}), 400
    try:
        X_one, summary = featurize_from_raw(raw)
    except Exception as e:
        return jsonify({"error": f"No se pudo procesar la orden: {e}"}), 400
    pred_idx = int(RF.predict(X_one)[0])
    proba = RF.predict_proba(X_one)[0]
    return jsonify({
        "prediction": CLASS_ORDER[pred_idx],
        "classes": CLASS_ORDER,
        "probabilities": {c: float(p) for c,p in zip(CLASS_ORDER, proba)},
        "summary": summary,
    })

@app.route("/route_preview")
def route_preview():
    q = (request.args.get("q") or "").strip()
    route = route_query(q)
    return jsonify({"mode": route["mode"], "label": route["label"]})

@app.route("/search")
def search():
    q = (request.args.get("q") or "").strip()
    if len(q) < 2:
        return jsonify({"error": "Escriba al menos dos caracteres.", "mode": "empty"}), 400
    route, hits = search_local(q, limit=5)
    cards = [candidate_card(r) for r in hits]
    if not cards:
        sugs = []
        if "-" in q and DOCKET_RX.search(q):
            sugs.append(re.sub(r"[\-]", " ", q))
        ym = YEAR_RX.search(q)
        if ym and route["mode"] == "free":
            sugs.append(q.replace(ym.group(0), "").strip())
        if len(q.split()) >= 3 and route["mode"] == "case_name":
            sugs.append(q.split()[0])
        return jsonify({
            "mode": route["mode"], "label": route["label"],
            "confidence": "none", "candidates": [],
            "suggestions": [s for s in sugs if s and s != q][:3],
        })
    confidence = "high" if (route["mode"] in ("url","id") or len(cards) == 1) else "ambiguous"
    return jsonify({
        "mode": route["mode"], "label": route["label"],
        "confidence": confidence, "candidates": cards, "suggestions": [],
    })

@app.route("/classify_docket", methods=["POST"])
def classify_docket():
    payload = request.get_json() or {}
    try:
        cid = int(payload.get("cluster_id") or payload.get("id") or 0)
    except Exception:
        return jsonify({"error": "ID invalido."}), 400
    rec = RAW_BY_ID.get(cid)
    if rec is None and cl_api:
        try:
            rec = cl_api.get_cluster(cid)
            if rec is not None:
                RAW_BY_ID[cid] = rec
        except Exception as e:
            print(f"[classify] fetch fallback fallo: {e}", flush=True)
    if rec is None:
        return jsonify({"error": "El caso no esta en el indice local ni en CourtListener."}), 404
    X_one, summary = featurize_from_raw(rec)
    pred_idx = int(RF.predict(X_one)[0])
    proba = RF.predict_proba(X_one)[0]
    return jsonify({
        "prediction": CLASS_ORDER[pred_idx],
        "classes": CLASS_ORDER,
        "probabilities": {c: float(p) for c,p in zip(CLASS_ORDER, proba)},
        "summary": summary,
    })

@app.route("/healthz")
def healthz():
    return jsonify({
        "ok": True,
        "local_cases": len(RAW_BY_ID),
        "cache_enabled": bool(cl_cache and cl_cache.enabled()),
    })

@app.route("/quota")
def quota():
    if not cl_api:
        return jsonify({"used": 0, "limit": 0, "remaining": 0, "cache_enabled": False, "available": False})
    q = cl_api.quota_status()
    q["available"] = True
    return jsonify(q)

def _record_to_card(rec):
    op0 = (rec.get("opinions") or [{}])[0]
    snippet = (op0.get("snippet") or "")[:160]
    return {
        "id": int(rec.get("cluster_id") or 0),
        "caseName": rec.get("caseName") or "",
        "court_id": rec.get("court_id") or "",
        "court_human": COURT_HUMAN.get(rec.get("court_id") or "", rec.get("court_id") or ""),
        "date": rec.get("dateFiled") or "",
        "docket": rec.get("docketNumber") or "",
        "judge": rec.get("judge") or "",
        "snippet": snippet,
        "source": "courtlistener",
    }

@app.route("/api/search")
def api_search():
    q = (request.args.get("q") or "").strip()
    if len(q) < 2:
        return jsonify({"error": "Escriba al menos dos caracteres."}), 400
    if not cl_api:
        return jsonify({"error": "Modulo CourtListener no disponible."}), 503
    route = route_query(q)
    mode = route["mode"]
    try:
        results = cl_api.search(q, mode, limit=5)
    except cl_api.NoToken:
        return jsonify({"error": "Falta configurar COURTLISTENER_TOKEN."}), 503
    except cl_api.QuotaExceeded as e:
        return jsonify({"error": str(e), "quota": cl_api.quota_status()}), 429
    except Exception as e:
        return jsonify({"error": f"CourtListener: {e}"}), 502
    for r in results:
        cid = r.get("cluster_id")
        if cid is not None:
            RAW_BY_ID[int(cid)] = r
    cards = [_record_to_card(r) for r in results]
    confidence = "high" if len(cards) == 1 else ("ambiguous" if cards else "none")
    return jsonify({
        "mode": mode, "label": route["label"],
        "confidence": confidence, "candidates": cards,
        "quota": cl_api.quota_status(),
    })

@app.route("/api/case/<int:cluster_id>")
def api_case(cluster_id):
    if not cl_api:
        return jsonify({"error": "Modulo CourtListener no disponible."}), 503
    try:
        rec = cl_api.get_cluster(cluster_id)
    except cl_api.NoToken:
        return jsonify({"error": "Falta configurar COURTLISTENER_TOKEN."}), 503
    except cl_api.QuotaExceeded as e:
        return jsonify({"error": str(e), "quota": cl_api.quota_status()}), 429
    except Exception as e:
        return jsonify({"error": f"CourtListener: {e}"}), 502
    if rec is None:
        return jsonify({"error": "Caso no encontrado."}), 404
    RAW_BY_ID[int(cluster_id)] = rec
    return jsonify({"case": _record_to_card(rec)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    host = os.environ.get("HOST", "127.0.0.1")
    print(f"[run] http://{host}:{port} | {len(RAW_BY_ID)} casos en indice local | cache={bool(cl_cache and cl_cache.enabled())}", flush=True)
    app.run(host=host, port=port, debug=False)

"""A3 - Procesamiento de datos crudos hacia tabla unica de modelado.

Pipeline:
  1. Carga full_ifp.json, full_counsel.json, full_extension.json
  2. Extraccion de features candidatas (validadas en tests_normalizacion.py)
  3. Deduplicacion intra y entre clases por cluster_id
  4. log1p sobre variables sesgadas (citeCount, n_opcites, len_caseName, n_dockets)
  5. Imputacion: MEDIA donde la distribucion final es simetrica (skew<0.5), MEDIANA si no
  6. Capeo de outliers (IQR x 1.5) sobre numericas
  7. One-hot de court_id, source. Frequency encoding de judge.
  8. Balanceo por submuestreo a la clase minoritaria
  9. Persiste dataset_a3.csv + reporte de pasos en proceso.log

Decisiones tomadas con el usuario:
  - Camino A: media donde corresponde por simetria, mediana donde no.
  - judge via frequency encoding (cardinalidad 0.53 verificada).
  - per_curiam, posture, procedural_history, suitNature, syllabus, lexisCite descartados.
  - op_type descartado por colinealidad con source.
  - Query 'extension' endurecida con "motion for extension of time".
"""
import json, math, os, re, sys
import numpy as np
import pandas as pd
from datetime import datetime

A3 = os.path.dirname(os.path.abspath(__file__))
LOG = f"{A3}/proceso.log"
OUT = f"{A3}/dataset_a3.csv"

CLASSES = ["ifp", "counsel", "extension"]

def log(m):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    open(LOG, "a").write(line + "\n")

def normalize_judge(s):
    if not s: return ""
    if "," in s:
        parts = [p.strip() for p in s.split(",")]
        if len(parts) >= 2:
            s = parts[1] + " " + parts[0]
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

def extract_row(r, label):
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
    }

def load_class(cls):
    p = f"{A3}/full_{cls}.json"
    if not os.path.exists(p):
        log(f"  NO EXISTE {p}, salteo")
        return []
    data = json.load(open(p))
    if isinstance(data, dict) and "results" in data:
        data = data["results"]
    return [extract_row(r, cls) for r in data]

def cap_iqr(s, k=1.5):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - k*iqr, q3 + k*iqr
    return s.clip(lo, hi)

def main():
    open(LOG, "w").close()
    log("=== PROCESAMIENTO A3 ===")
    rows = []
    for c in CLASSES:
        r = load_class(c)
        log(f"Clase {c}: {len(r)} filas crudas")
        rows.extend(r)
    if not rows:
        log("Sin datos. Aborto."); return
    df = pd.DataFrame(rows)
    log(f"Total filas crudas: {len(df)}; columnas: {list(df.columns)}")

    log("--- Deduplicacion ---")
    pre = len(df)
    df = df.drop_duplicates(subset=["cluster_id"], keep="first")
    log(f"  por cluster_id: {pre} -> {len(df)}")
    pre2 = len(df)
    g = df.groupby("cluster_id")["motion_type"].nunique()
    multi = g[g > 1].index.tolist()
    if multi:
        df = df[~df["cluster_id"].isin(multi)]
        log(f"  cluster_id en multiples clases: {len(multi)} removidos, quedan {len(df)}")

    log("--- Conteo por clase post-dedup ---")
    log(f"  {dict(df['motion_type'].value_counts())}")

    log("--- Transformaciones log1p ---")
    for col in ["citeCount", "n_opcites", "len_caseName", "n_dockets"]:
        df[col] = df[col].fillna(0).clip(lower=0)
        df[col + "_log"] = np.log1p(df[col].astype(float))
        df = df.drop(columns=[col])
        log(f"  log1p aplicado a {col} -> {col}_log")

    log("--- Imputacion por skewness ---")
    numericas = [c for c in df.columns if c.endswith("_log")] + ["n_citation","len_snippet","attorney_len","year","month"]
    decisiones = {}
    for col in numericas:
        if col not in df.columns: continue
        s = df[col].dropna().astype(float)
        skew = s.skew()
        if abs(skew) < 0.5:
            val = s.mean(); metodo = "MEDIA"
        else:
            val = s.median(); metodo = "MEDIANA"
        df[col] = df[col].fillna(val)
        decisiones[col] = (metodo, round(skew,3), round(float(val),3))
        log(f"  {col}: skew={skew:+.2f} -> {metodo} = {val:.3f}")

    log("--- Capeo IQR (k=1.5) sobre numericas ---")
    for col in numericas:
        if col not in df.columns: continue
        df[col] = cap_iqr(df[col].astype(float))

    log("--- Frequency encoding judge ---")
    freq = df["judge_norm"].value_counts().to_dict()
    df["judge_freq"] = df["judge_norm"].map(freq).fillna(0).astype(int)
    df = df.drop(columns=["judge_norm"])
    log(f"  jueces unicos: {len(freq)}; freq max: {max(freq.values()) if freq else 0}")

    log("--- One-hot court_id y source ---")
    df = pd.get_dummies(df, columns=["court_id","source"], prefix=["court","src"], drop_first=False)

    log("--- Booleanos a int ---")
    for col in ["is_pro_se","is_in_re"]:
        df[col] = df[col].astype(int)

    log("--- Balanceo por submuestreo ---")
    counts = df["motion_type"].value_counts()
    minc = counts.min()
    log(f"  clase minoritaria: {minc}")
    parts = []
    for cls, n in counts.items():
        sub = df[df["motion_type"]==cls].sample(n=minc, random_state=42)
        parts.append(sub)
    df_bal = pd.concat(parts, ignore_index=True)
    log(f"  dataset balanceado: {len(df_bal)} filas, distribucion: {dict(df_bal['motion_type'].value_counts())}")

    log("--- Persistencia ---")
    df_bal = df_bal.drop(columns=["cluster_id"])
    df_bal.to_csv(OUT, index=False)
    log(f"  guardado {OUT} con {df_bal.shape[0]} filas y {df_bal.shape[1]} cols")
    log(f"  columnas finales: {list(df_bal.columns)}")

    log("=== FIN PROCESAMIENTO ===")

if __name__ == "__main__":
    main()

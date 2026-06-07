"""A3 - Evaluacion en hold-out fresco.

Toma 5 ordenes que el modelo NUNCA vio (ni en train ni en test) porque fueron
descartadas en el balanceo por submuestreo, las pasa por el pipeline y muestra
la prediccion con sus probabilidades para verificar que el modelo tenga sentido.

Reentrena el RandomForest con el mismo seed=42 para reproducibilidad.
"""
import json, os, re, sys, random
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

A3 = os.path.dirname(os.path.abspath(__file__))
SEED_MODEL = 42
SEED_SAMPLE = 7
N_PRUEBAS = 5
CLASSES = ["ifp", "counsel", "extension"]

def log(m): print(m, flush=True)

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
        "_caseName_raw": r.get("caseName") or "",
        "_judge_raw": r.get("judge") or "",
        "_attorney_raw": (attorney[:80] + "...") if len(attorney) > 80 else attorney,
        "_date": date,
    }

def cap_iqr(s, k=1.5):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return s.clip(q1 - k*iqr, q3 + k*iqr)

def process_pipeline(df_raw, freq_judge=None, cap_bounds=None, impute_vals=None, train_columns=None):
    """Aplica el mismo pipeline de procesamiento.py. Si recibe parametros aprendidos
    de train, los reutiliza (esto importa para el hold-out)."""
    df = df_raw.copy()
    for col in ["citeCount", "n_opcites", "len_caseName", "n_dockets"]:
        df[col] = df[col].fillna(0).clip(lower=0).astype(float)
        df[col + "_log"] = np.log1p(df[col])
        df = df.drop(columns=[col])

    numericas = [c for c in df.columns if c.endswith("_log")] + ["n_citation","len_snippet","attorney_len","year","month"]
    if impute_vals is None:
        impute_vals = {}
        for col in numericas:
            if col not in df.columns: continue
            s = df[col].dropna().astype(float)
            skew = s.skew()
            val = s.mean() if abs(skew) < 0.5 else s.median()
            impute_vals[col] = float(val)
    for col, val in impute_vals.items():
        if col in df.columns: df[col] = df[col].fillna(val)

    if cap_bounds is None:
        cap_bounds = {}
        for col in numericas:
            if col not in df.columns: continue
            s = df[col].astype(float)
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            cap_bounds[col] = (float(q1 - 1.5*iqr), float(q3 + 1.5*iqr))
    for col, (lo, hi) in cap_bounds.items():
        if col in df.columns: df[col] = df[col].astype(float).clip(lo, hi)

    if freq_judge is None:
        freq_judge = df["judge_norm"].value_counts().to_dict()
    df["judge_freq"] = df["judge_norm"].map(freq_judge).fillna(0).astype(int)
    df = df.drop(columns=["judge_norm"])

    df = pd.get_dummies(df, columns=["court_id","source"], prefix=["court","src"], drop_first=False)
    for col in ["is_pro_se","is_in_re"]:
        df[col] = df[col].astype(int)

    if train_columns is not None:
        for c in train_columns:
            if c not in df.columns: df[c] = 0
        meta_cols = [c for c in df.columns if c.startswith("_")]
        keep_meta = df[meta_cols] if meta_cols else None
        df = df[train_columns + (["motion_type"] if "motion_type" in df.columns else [])]
        if keep_meta is not None:
            for c in meta_cols: df[c] = keep_meta[c].values

    return df, freq_judge, cap_bounds, impute_vals

def main():
    log("="*70)
    log("EVALUACION EN HOLD-OUT FRESCO")
    log("="*70)

    log("\n1) Cargando registros crudos y dataset balanceado.")
    raw_rows = []
    for c in CLASSES:
        data = json.load(open(f"{A3}/full_{c}.json"))
        if isinstance(data, dict) and "results" in data: data = data["results"]
        for r in data:
            raw_rows.append(extract_row(r, c))
    df_raw = pd.DataFrame(raw_rows)
    df_raw = df_raw.drop_duplicates(subset=["cluster_id"], keep="first")
    g = df_raw.groupby("cluster_id")["motion_type"].nunique()
    multi = g[g > 1].index.tolist()
    if multi:
        df_raw = df_raw[~df_raw["cluster_id"].isin(multi)]
    log(f"   Crudo limpio (dedup + sin multiclase): {len(df_raw)} filas")
    log(f"   Por clase: {dict(df_raw['motion_type'].value_counts())}")

    dataset_bal = pd.read_csv(f"{A3}/dataset_a3.csv")
    log(f"   Dataset entrenamiento: {len(dataset_bal)} filas")

    log("\n2) Reentrenando modelo con el dataset balanceado (seed=42).")
    from sklearn.model_selection import train_test_split
    df_train_full = dataset_bal.copy()
    y_full = LabelEncoder()
    y_enc = y_full.fit_transform(df_train_full["motion_type"])
    X_full = df_train_full.drop(columns=["motion_type"]).astype(float)
    feature_cols = list(X_full.columns)
    X_tr, X_te, y_tr, y_te = train_test_split(X_full, y_enc, test_size=0.2, random_state=SEED_MODEL, stratify=y_enc)

    train_cluster_ids = set()
    rf = RandomForestClassifier(n_estimators=200, random_state=SEED_MODEL, n_jobs=-1, max_depth=12, min_samples_leaf=3)
    rf.fit(X_tr, y_tr)
    log(f"   Train={len(X_tr)} Test={len(X_te)} | acc en test: {rf.score(X_te, y_te):.3f}")

    log("\n3) Reproduciendo parametros del pipeline (impute, cap, judge_freq).")
    minc = df_raw["motion_type"].value_counts().min()
    parts = []
    for cls, n in df_raw["motion_type"].value_counts().items():
        parts.append(df_raw[df_raw["motion_type"]==cls].sample(n=minc, random_state=42))
    df_train_subset = pd.concat(parts, ignore_index=True)
    train_cluster_ids = set(df_train_subset["cluster_id"].tolist())
    _, freq_judge, cap_bounds, impute_vals = process_pipeline(df_train_subset.drop(columns=[c for c in df_train_subset.columns if c.startswith("_")]))
    log(f"   judges aprendidos: {len(freq_judge)}, impute keys: {len(impute_vals)}")

    log("\n4) Identificando hold-out fresco (registros NO usados en train ni test).")
    holdout_pool = df_raw[~df_raw["cluster_id"].isin(train_cluster_ids)].copy()
    log(f"   Hold-out disponible: {len(holdout_pool)} filas")
    log(f"   Por clase: {dict(holdout_pool['motion_type'].value_counts())}")

    log(f"\n5) Tomando {N_PRUEBAS} muestras aleatorias estratificadas (seed={SEED_SAMPLE}).")
    samples = []
    rng = random.Random(SEED_SAMPLE)
    per_class = max(1, N_PRUEBAS // 3)
    extra = N_PRUEBAS - per_class * 3
    for cls in CLASSES:
        pool = holdout_pool[holdout_pool["motion_type"]==cls]
        if len(pool) == 0: continue
        n = per_class + (1 if extra > 0 else 0); extra = max(0, extra-1)
        idx = rng.sample(list(pool.index), min(n, len(pool)))
        samples.extend(idx)
    df_samples = holdout_pool.loc[samples].reset_index(drop=True)
    log(f"   Tomadas: {len(df_samples)} filas")

    log("\n6) Aplicando pipeline a las muestras con parametros del train.")
    df_proc, _, _, _ = process_pipeline(df_samples, freq_judge=freq_judge,
                                         cap_bounds=cap_bounds, impute_vals=impute_vals,
                                         train_columns=feature_cols)
    meta_cols = [c for c in df_proc.columns if c.startswith("_")]
    X_holdout = df_proc[feature_cols].astype(float)
    y_real = df_proc["motion_type"].values
    y_pred = rf.predict(X_holdout)
    y_proba = rf.predict_proba(X_holdout)
    classes_order = list(y_full.classes_)

    log("\n" + "="*70)
    log("RESULTADOS - 5 ORDENES HOLD-OUT")
    log("="*70)
    aciertos = 0
    for i in range(len(df_proc)):
        real = y_real[i]
        pred = classes_order[y_pred[i]]
        ok = "ACIERTO" if real == pred else "ERROR  "
        if real == pred: aciertos += 1
        log(f"\n--- Prueba {i+1} [{ok}] ---")
        log(f"  Caso     : {df_samples.loc[i,'_caseName_raw']}")
        log(f"  Fecha    : {df_samples.loc[i,'_date']}  Corte: {df_samples.loc[i,'court_id']}")
        log(f"  Juez     : {df_samples.loc[i,'_judge_raw'] or '(sin dato)'}")
        log(f"  Attorney : {df_samples.loc[i,'_attorney_raw'] or '(sin dato)'}")
        log(f"  pro_se={bool(df_samples.loc[i,'is_pro_se'])}  attorney_len={int(df_samples.loc[i,'attorney_len'])}")
        log(f"  REAL     : {real}")
        log(f"  PREDICHO : {pred}")
        probs = " ".join(f"{c}={p:.2f}" for c,p in zip(classes_order, y_proba[i]))
        log(f"  Probabs  : {probs}")

    log("\n" + "="*70)
    log(f"RESUMEN: {aciertos}/{len(df_proc)} aciertos = {100*aciertos/len(df_proc):.1f}%")
    log("="*70)

if __name__ == "__main__":
    main()

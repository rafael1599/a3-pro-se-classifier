"""A3 - Evaluacion caso a caso sobre el TEST SET formal (167 filas).

Reproduce el pipeline completo conservando cluster_id, replica el split 80/20
con seed=42 para identificar exactamente las mismas 167 filas que el modelo
no vio en entrenamiento, toma 5 muestras estratificadas, y muestra para cada
una el caso real, lo predicho y las probabilidades.

Como bono: corre tambien la prediccion sobre los 85 registros descartados por
el balanceo (hold-out adicional sin clase extension).
"""
import json, os, re, random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

A3 = os.path.dirname(os.path.abspath(__file__))
SEED = 42
SEED_SAMPLE = 11
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
        "_attorney_raw": attorney,
        "_date": date,
        "_docket": docket,
    }

def main():
    log("="*72)
    log("EVALUACION CASO A CASO SOBRE EL TEST SET FORMAL (HOLD-OUT REAL)")
    log("="*72)

    log("\n1) Cargando registros crudos.")
    raw_rows = []
    for c in CLASSES:
        data = json.load(open(f"{A3}/full_{c}.json"))
        if isinstance(data, dict) and "results" in data: data = data["results"]
        for r in data:
            raw_rows.append(extract_row(r, c))
    df_raw = pd.DataFrame(raw_rows)

    log("2) Dedup por cluster_id y exclusion de cluster_ids en multiples clases.")
    df_raw = df_raw.drop_duplicates(subset=["cluster_id"], keep="first")
    g = df_raw.groupby("cluster_id")["motion_type"].nunique()
    multi = g[g > 1].index.tolist()
    if multi: df_raw = df_raw[~df_raw["cluster_id"].isin(multi)]
    log(f"   {len(df_raw)} filas limpias. Por clase: {dict(df_raw['motion_type'].value_counts())}")

    log("3) Balanceo a clase minoritaria con seed=42 (igual que procesamiento.py).")
    minc = df_raw["motion_type"].value_counts().min()
    parts = []
    for cls, n in df_raw["motion_type"].value_counts().items():
        parts.append(df_raw[df_raw["motion_type"]==cls].sample(n=minc, random_state=42))
    df_bal = pd.concat(parts, ignore_index=True)
    log(f"   Balanceado: {len(df_bal)} filas a {minc} por clase.")

    log("4) Pipeline de features (log1p + imputacion skew-aware + cap IQR + encodings).")
    meta_cols = ["_caseName_raw","_judge_raw","_attorney_raw","_date","_docket","cluster_id"]
    df_meta = df_bal[meta_cols].copy()
    df = df_bal.drop(columns=meta_cols).copy()

    for col in ["citeCount","n_opcites","len_caseName","n_dockets"]:
        df[col] = df[col].fillna(0).clip(lower=0).astype(float)
        df[col + "_log"] = np.log1p(df[col]); df = df.drop(columns=[col])

    numericas = [c for c in df.columns if c.endswith("_log")] + ["n_citation","len_snippet","attorney_len","year","month"]
    for col in numericas:
        if col not in df.columns: continue
        s = df[col].dropna().astype(float)
        val = s.mean() if abs(s.skew()) < 0.5 else s.median()
        df[col] = df[col].fillna(val)
    for col in numericas:
        if col not in df.columns: continue
        s = df[col].astype(float)
        q1, q3 = s.quantile(0.25), s.quantile(0.75); iqr = q3 - q1
        df[col] = s.clip(q1 - 1.5*iqr, q3 + 1.5*iqr)

    freq = df["judge_norm"].value_counts().to_dict()
    df["judge_freq"] = df["judge_norm"].map(freq).fillna(0).astype(int)
    df = df.drop(columns=["judge_norm"])
    df = pd.get_dummies(df, columns=["court_id","source"], prefix=["court","src"], drop_first=False)
    for col in ["is_pro_se","is_in_re"]: df[col] = df[col].astype(int)

    y_label = df["motion_type"].values
    X = df.drop(columns=["motion_type"]).astype(float)
    feature_cols = list(X.columns)

    log("5) Split 80/20 estratificado con seed=42 (mismas filas que modelos.py).")
    le = LabelEncoder()
    y_enc = le.fit_transform(y_label)
    idx = np.arange(len(X))
    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X.values, y_enc, idx, test_size=0.2, random_state=SEED, stratify=y_enc)
    log(f"   Train={len(X_tr)}  Test={len(X_te)}")
    classes_order = list(le.classes_)

    log("6) Entrenando RandomForest (mismos hiperparametros).")
    rf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1, max_depth=12, min_samples_leaf=3)
    rf.fit(X_tr, y_tr)
    acc_test = rf.score(X_te, y_te)
    log(f"   Accuracy global en test: {acc_test:.3f} (coincide con metricas.json)")

    log("\n7) Muestreo estratificado de 5 casos del test set.")
    df_test = pd.DataFrame(X_te, columns=feature_cols)
    df_test["_y_real"] = [classes_order[v] for v in y_te]
    df_test["_meta_idx"] = idx_te
    rng = random.Random(SEED_SAMPLE)
    elegidos = []
    per_class = N_PRUEBAS // 3
    extra = N_PRUEBAS - per_class*3
    for cls in CLASSES:
        pool = df_test[df_test["_y_real"]==cls]
        n = per_class + (1 if extra > 0 else 0); extra = max(0, extra-1)
        sub = rng.sample(list(pool.index), min(n, len(pool)))
        elegidos.extend(sub)
    df_pick = df_test.loc[elegidos].reset_index(drop=True)
    X_pick = df_pick[feature_cols].values
    y_pred = rf.predict(X_pick)
    y_proba = rf.predict_proba(X_pick)

    log("\n" + "="*72)
    log("CINCO PRUEBAS - HOLD-OUT FORMAL (test set, jamas visto en entrenamiento)")
    log("="*72)
    aciertos = 0
    for i in range(len(df_pick)):
        meta = df_meta.iloc[df_pick.loc[i,"_meta_idx"]]
        real = df_pick.loc[i,"_y_real"]
        pred = classes_order[y_pred[i]]
        ok = "ACIERTO" if real == pred else "ERROR  "
        if real == pred: aciertos += 1
        case = meta["_caseName_raw"]
        attorney = meta["_attorney_raw"] or "(sin abogados registrados)"
        attorney_show = (attorney[:120] + "...") if len(attorney) > 120 else attorney
        judge = meta["_judge_raw"] or "(sin dato)"
        log(f"\n--- Prueba {i+1} [{ok}] ---")
        log(f"  Caso     : {case}")
        log(f"  Fecha    : {meta['_date']}   Corte: {df_bal.loc[df_pick.loc[i,'_meta_idx'],'court_id']}")
        log(f"  Juez     : {judge}")
        log(f"  Docket   : {meta['_docket'][:60] if meta['_docket'] else '(sin dato)'}")
        log(f"  Attorney : {attorney_show}")
        log(f"  pro_se={'SI' if df_bal.loc[df_pick.loc[i,'_meta_idx'],'is_pro_se'] else 'NO'}  "
            f"attorney_len={int(df_bal.loc[df_pick.loc[i,'_meta_idx'],'attorney_len'])}")
        log(f"  REAL     : {real}")
        log(f"  PREDICHO : {pred}")
        probs = "  ".join(f"{c}={p:.2f}" for c,p in zip(classes_order, y_proba[i]))
        log(f"  Probabs  : {probs}")
    log("\n" + "="*72)
    log(f"RESUMEN HOLD-OUT FORMAL: {aciertos}/{len(df_pick)} aciertos = {100*aciertos/len(df_pick):.1f}%")
    log(f"(Referencia esperada segun accuracy global: ~{int(acc_test*N_PRUEBAS)}/5)")
    log("="*72)

    log("\n" + "="*72)
    log("BONUS - PREDICCION SOBRE LOS 85 DESCARTADOS POR BALANCEO")
    log("="*72)
    train_cluster_ids = set(df_bal["cluster_id"].tolist())
    df_extra = df_raw[~df_raw["cluster_id"].isin(train_cluster_ids)].copy()
    log(f"Hold-out extra: {len(df_extra)} filas (por clase: {dict(df_extra['motion_type'].value_counts())})")

    df_extra_proc = df_extra.drop(columns=meta_cols).copy()
    for col in ["citeCount","n_opcites","len_caseName","n_dockets"]:
        df_extra_proc[col] = df_extra_proc[col].fillna(0).clip(lower=0).astype(float)
        df_extra_proc[col + "_log"] = np.log1p(df_extra_proc[col]); df_extra_proc = df_extra_proc.drop(columns=[col])
    for col in numericas:
        if col not in df_extra_proc.columns: continue
        df_extra_proc[col] = df_extra_proc[col].fillna(df[col].median())
    df_extra_proc["judge_freq"] = df_extra_proc["judge_norm"].map(freq).fillna(0).astype(int)
    df_extra_proc = df_extra_proc.drop(columns=["judge_norm"])
    df_extra_proc = pd.get_dummies(df_extra_proc, columns=["court_id","source"], prefix=["court","src"], drop_first=False)
    for col in ["is_pro_se","is_in_re"]: df_extra_proc[col] = df_extra_proc[col].astype(int)
    for c in feature_cols:
        if c not in df_extra_proc.columns: df_extra_proc[c] = 0
    X_extra = df_extra_proc[feature_cols].astype(float).values
    y_extra_real = df_extra_proc["motion_type"].values
    y_extra_pred = rf.predict(X_extra)
    y_extra_lbl = [classes_order[v] for v in y_extra_pred]
    aciertos_extra = sum(1 for r,p in zip(y_extra_real, y_extra_lbl) if r==p)
    log(f"Aciertos: {aciertos_extra}/{len(df_extra)} = {100*aciertos_extra/len(df_extra):.1f}%")
    log("Matriz de confusion (filas=real, cols=pred). Solo clases presentes:")
    cm = pd.crosstab(pd.Series(y_extra_real,name="REAL"), pd.Series(y_extra_lbl,name="PRED"))
    log(cm.to_string())
    log("\nClassification report:")
    log(classification_report(y_extra_real, y_extra_lbl, zero_division=0))
    log("="*72)

if __name__ == "__main__":
    main()

"""A3 - Entrenamiento de 3 modelos ML clasicos sobre dataset_a3.csv.

Cumple consigna del profesor: 1 split train/test, 3 modelos, justificacion clara.
Modelos: LogisticRegression (lineal), DecisionTree (no lineal interpretable), RandomForest (ensemble).
"""
import json, os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                              confusion_matrix, classification_report,
                              roc_auc_score)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

A3 = os.path.dirname(os.path.abspath(__file__))
DATA = f"{A3}/dataset_a3.csv"
LOG = f"{A3}/modelos.log"
METRICS = f"{A3}/metricas.json"
SEED = 42

def log(m):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    open(LOG, "a").write(line + "\n")

def plot_cm(y_true, y_pred, labels, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right"); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real"); ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center", color="black" if cm[i,j]<cm.max()/2 else "white")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)

def main():
    open(LOG, "w").close()
    log("=== ENTRENAMIENTO A3 ===")
    df = pd.read_csv(DATA)
    log(f"Dataset: {df.shape[0]} filas, {df.shape[1]} cols")
    log(f"Distrib motion_type: {dict(df['motion_type'].value_counts())}")

    le = LabelEncoder()
    y = le.fit_transform(df["motion_type"])
    X = df.drop(columns=["motion_type"]).astype(float)
    feature_names = list(X.columns)
    labels = list(le.classes_)
    log(f"Features ({len(feature_names)}): {feature_names}")
    log(f"Clases: {labels}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)
    log(f"Split 80/20 estratificado seed={SEED}: train={len(X_train)} test={len(X_test)}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    log("StandardScaler ajustado en train, aplicado a test")

    modelos = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=SEED),
        "DecisionTree": DecisionTreeClassifier(random_state=SEED, max_depth=8, min_samples_leaf=5),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1, max_depth=12, min_samples_leaf=3),
    }

    resultados = {}
    for nombre, model in modelos.items():
        log(f"--- {nombre} ---")
        if nombre == "LogisticRegression":
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)
            y_proba = model.predict_proba(X_test_s)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
        prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
        try:
            auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
        except Exception:
            auc = None

        log(f"  accuracy={acc:.3f} f1_macro={f1_m:.3f} f1_weighted={f1_w:.3f} auc_ovr={auc}")
        log("  reporte por clase:\n" + classification_report(y_test, y_pred, target_names=labels, zero_division=0))

        cm_path = f"{A3}/cm_{nombre}.png"
        plot_cm(y_test, y_pred, labels, f"Matriz de confusion - {nombre}", cm_path)
        log(f"  matriz guardada en {cm_path}")

        resultados[nombre] = {
            "accuracy": round(acc, 4),
            "precision_macro": round(prec_m, 4),
            "recall_macro": round(rec_m, 4),
            "f1_macro": round(f1_m, 4),
            "precision_weighted": round(prec_w, 4),
            "recall_weighted": round(rec_w, 4),
            "f1_weighted": round(f1_w, 4),
            "roc_auc_ovr_macro": round(auc, 4) if auc is not None else None,
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "report_per_class": {labels[i]: {
                "precision": float(precision_recall_fscore_support(y_test, y_pred, labels=[i], zero_division=0)[0][0]),
                "recall": float(precision_recall_fscore_support(y_test, y_pred, labels=[i], zero_division=0)[1][0]),
                "f1": float(precision_recall_fscore_support(y_test, y_pred, labels=[i], zero_division=0)[2][0]),
            } for i in range(len(labels))},
        }

        if nombre in ("DecisionTree","RandomForest"):
            imp = sorted(zip(feature_names, model.feature_importances_), key=lambda x: -x[1])
            resultados[nombre]["feature_importance_top10"] = [(f, float(round(v,4))) for f,v in imp[:10]]
            log(f"  top10 features: {imp[:10]}")
        else:
            coef_means = np.abs(model.coef_).mean(axis=0)
            imp = sorted(zip(feature_names, coef_means), key=lambda x: -x[1])
            resultados[nombre]["coef_abs_mean_top10"] = [(f, float(round(v,4))) for f,v in imp[:10]]
            log(f"  top10 features por |coef|: {imp[:10]}")

    json.dump({"labels": labels, "feature_names": feature_names, "models": resultados},
              open(METRICS, "w"), indent=2)
    log(f"Metricas guardadas en {METRICS}")

    log("=== RESUMEN COMPARATIVO ===")
    for nombre, m in resultados.items():
        log(f"  {nombre}: acc={m['accuracy']} f1_macro={m['f1_macro']} auc={m['roc_auc_ovr_macro']}")
    log("=== FIN ENTRENAMIENTO ===")

if __name__ == "__main__":
    main()

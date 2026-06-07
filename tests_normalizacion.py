"""Tests unitarios de normalizacion y decisiones de tratamiento.

Objetivo: con los samples locales, medir empiricamente:
  - Cardinalidad de 'judge' antes y despues de normalizar -> decide si usarlo
  - Skewness antes/despues de log1p sobre features numericas -> decide media vs mediana
  - Duplicados cross-clase y por cluster -> impacta tamano final del dataset
  - Distribucion de court_id y desbalance de clase
  - Pruebas de robustez de las regex de feature engineering
"""
import json, math, os, re, sys
from collections import Counter
from glob import glob

A3 = os.path.dirname(os.path.abspath(__file__))
SAMPLES = {
    "ifp": [f"{A3}/sample_federal_district.json", f"{A3}/unidad_ifp.json", f"{A3}/verif_ifp.json"],
    "counsel": [f"{A3}/unidad_counsel.json", f"{A3}/verif_counsel.json"],
    "extension": [f"{A3}/verif_extension.json"],
}

def load_class(paths):
    out = []
    for p in paths:
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        for r in d.get("results", []):
            out.append(r)
    return out

def stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"n": 0}
    n = len(xs)
    mu = sum(xs)/n
    var = sum((x-mu)**2 for x in xs)/n if n>1 else 0
    sd = math.sqrt(var)
    m3 = sum((x-mu)**3 for x in xs)/n if n>1 else 0
    skew = m3/(sd**3) if sd>0 else 0.0
    return {"n": n, "min": min(xs), "max": max(xs), "mean": mu, "sd": sd, "skew": skew}

def normalize_judge(s):
    if not s: return ""
    x = s.lower().strip()
    x = re.sub(r"[.,]", " ", x)
    x = re.sub(r"\b(judge|district|magistrate|chief|hon|honorable|sr|jr|usdj|usmj)\b", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    if "," in s:
        parts = [p.strip() for p in s.split(",")]
        if len(parts) >= 2:
            x = (parts[1] + " " + parts[0]).lower()
            x = re.sub(r"[.,]", " ", x)
            x = re.sub(r"\s+", " ", x).strip()
    tokens = x.split()
    if not tokens: return ""
    return tokens[-1]

def count_dockets(s):
    if not s: return 0
    parts = re.split(r"[,;]|\sand\s", s)
    parts = [p.strip() for p in parts if p.strip()]
    return len(parts)

def is_pro_se(attorney):
    if not attorney: return False
    return bool(re.search(r"\bpro\s*se\b", attorney, re.IGNORECASE))

def extract_row(r, label):
    op0 = (r.get("opinions") or [{}])[0]
    snippet = op0.get("snippet") or ""
    citation = r.get("citation") or []
    cites = op0.get("cites") or []
    docket = r.get("docketNumber") or ""
    attorney = r.get("attorney") or ""
    date = r.get("dateFiled") or ""
    return {
        "label": label,
        "cluster_id": r.get("cluster_id"),
        "court_id": r.get("court_id"),
        "dateFiled": date,
        "year": int(date.split("-")[0]) if date else None,
        "month": int(date.split("-")[1]) if date else None,
        "judge_raw": r.get("judge") or "",
        "judge_norm": normalize_judge(r.get("judge") or ""),
        "citeCount": r.get("citeCount"),
        "n_citation": len(citation),
        "n_opcites": len(cites),
        "len_snippet": len(snippet),
        "len_caseName": len(r.get("caseName") or ""),
        "n_dockets": count_dockets(docket),
        "source": r.get("source"),
        "op_type": op0.get("type"),
        "per_curiam": op0.get("per_curiam"),
        "is_pro_se": is_pro_se(attorney),
        "is_in_re": (r.get("caseName") or "").lower().startswith("in re"),
        "attorney_len": len(attorney),
    }

def header(t):
    print()
    print("="*70)
    print(t)
    print("="*70)

def main():
    rows = []
    for cls, paths in SAMPLES.items():
        for r in load_class(paths):
            rows.append(extract_row(r, cls))

    print(f"Total filas cargadas: {len(rows)}")
    print(f"Por clase: {dict(Counter(r['label'] for r in rows))}")

    header("TEST 1 - Deduplicacion por cluster_id")
    cid = [r["cluster_id"] for r in rows]
    dups = [x for x,c in Counter(cid).items() if c>1]
    print(f"cluster_id unicos: {len(set(cid))} / total {len(cid)}")
    print(f"duplicados detectados: {len(dups)} -> ids: {dups[:5]}{'...' if len(dups)>5 else ''}")
    seen = set(); rows_dedup = []
    for r in rows:
        if r["cluster_id"] in seen: continue
        seen.add(r["cluster_id"]); rows_dedup.append(r)
    print(f"Tras dedup: {len(rows_dedup)} filas")
    print(f"Por clase tras dedup: {dict(Counter(r['label'] for r in rows_dedup))}")

    header("TEST 2 - Cardinalidad de judge: crudo vs normalizado")
    judges_raw = [r["judge_raw"] for r in rows_dedup if r["judge_raw"]]
    judges_norm = [r["judge_norm"] for r in rows_dedup if r["judge_norm"]]
    print(f"N con judge poblado: {len(judges_raw)} ({100*len(judges_raw)/len(rows_dedup):.1f}%)")
    print(f"Unicos crudos: {len(set(judges_raw))}")
    print(f"Unicos normalizados (apellido): {len(set(judges_norm))}")
    cnt = Counter(judges_norm).most_common(8)
    print("Top jueces normalizados:")
    for j,c in cnt:
        print(f"  {j!r}: {c}")
    ratio_norm = len(set(judges_norm)) / max(1, len(judges_norm))
    print(f"Ratio unicos/total normalizado: {ratio_norm:.2f}")
    print(f"VEREDICTO: {'ALTA cardinalidad, mejor descartar' if ratio_norm > 0.6 else 'cardinalidad razonable, frequency encoding viable'}")

    header("TEST 3 - Skewness de numericas: crudo vs log1p")
    numericas = ["citeCount", "n_citation", "n_opcites", "len_snippet", "len_caseName", "n_dockets", "attorney_len"]
    for col in numericas:
        vals = [r[col] for r in rows_dedup if r[col] is not None]
        s_raw = stats(vals)
        vals_log = [math.log1p(v) for v in vals if v >= 0]
        s_log = stats(vals_log)
        gana = "log1p" if abs(s_log["skew"]) < abs(s_raw["skew"]) else "crudo"
        sim_raw = "simetrica" if abs(s_raw["skew"]) < 0.5 else ("moderada" if abs(s_raw["skew"]) < 1 else "alta")
        sim_log = "simetrica" if abs(s_log["skew"]) < 0.5 else ("moderada" if abs(s_log["skew"]) < 1 else "alta")
        print(f"{col:>14}: skew_raw={s_raw['skew']:+.2f} ({sim_raw}) | skew_log1p={s_log['skew']:+.2f} ({sim_log}) -> usar {gana}")

    header("TEST 4 - Justificacion del uso de MEDIA vs mediana")
    print("Criterio: usar MEDIA si skew < 0.5 (aprox simetrica) tras transformacion elegida.")
    for col in numericas:
        vals = [r[col] for r in rows_dedup if r[col] is not None]
        s_raw = stats(vals)
        vals_log = [math.log1p(v) for v in vals if v >= 0]
        s_log = stats(vals_log)
        skew_final = s_log["skew"] if abs(s_log["skew"]) < abs(s_raw["skew"]) else s_raw["skew"]
        decision = "MEDIA" if abs(skew_final) < 0.5 else "MEDIANA"
        print(f"{col:>14}: skew_final={skew_final:+.2f} -> {decision}")

    header("TEST 5 - Distribucion court_id (impacta one-hot)")
    courts = Counter(r["court_id"] for r in rows_dedup)
    for c,n in courts.most_common():
        print(f"  {c}: {n} ({100*n/len(rows_dedup):.1f}%)")

    header("TEST 6 - Variables degeneradas (varianza ~0)")
    for col in ["per_curiam", "is_pro_se", "is_in_re", "source", "op_type"]:
        vals = [r[col] for r in rows_dedup]
        c = Counter(vals)
        unique = len(c)
        mayor = c.most_common(1)[0][1] if c else 0
        pct_mayor = 100*mayor/len(vals) if vals else 0
        flag = "DESCARTABLE" if pct_mayor > 95 else "OK"
        print(f"  {col}: distrib={dict(c)} mayoritario={pct_mayor:.1f}% -> {flag}")

    header("TEST 7 - is_pro_se por clase (revalidacion creencia)")
    by_class = {}
    for r in rows_dedup:
        by_class.setdefault(r["label"], []).append(r["is_pro_se"])
    for cls, vs in by_class.items():
        pct = 100*sum(1 for v in vs if v)/len(vs)
        print(f"  {cls}: {pct:.1f}% pro_se en {len(vs)} muestras")

    header("TEST 8 - Trade-off tamano dataset: 330 vs 970 por clase")
    print("Datos visibles globalmente (counts API totales):")
    print("  IFP=1448, counsel=976, extension=2846")
    print("Escenarios:")
    print("  A) 330 por clase -> total ~990 filas, requests aprox: 330/20 * 3 = 49.5 -> ~50 req")
    print("  B) 970 por clase -> total ~2910 filas, requests aprox: 970/20 * 3 = 145.5 -> ~146 req")
    print(f"Limite diario API: 125 req/dia, restantes hoy aprox: ~115")
    print(f"Recomendacion: A entra hoy completo; B requiere 2 dias o saltarse limite")

    header("TEST 9 - Sanidad de regex de feature engineering")
    casos = [
        ("92 C 5381, 92 C 5551, 92 C 5656, 92 C 7407, 93 C 0069, 93 C 1443, 93 C 2725, 93 C 3001, 93 C 3750, 93 C 5474, 93 C 5707, 93 C 5708, 94 C 4018, 94 C 5270, 95 C 3405 and 95 C 5245", 16),
        ("No. 13-CV-659 (MKB)", 1),
        ("04 Cr. 0793; 09 Civ. 6501", 2),
        ("", 0),
    ]
    for s,exp in casos:
        got = count_dockets(s)
        ok = "PASS" if got==exp else "FAIL"
        preview = (s[:40] + "...") if len(s)>40 else s
        print(f"  count_dockets({preview!r}) = {got} (esperado {exp}) [{ok}]")

    pro_se_cases = [
        ("Jones, pro se", True),
        ("Brenda Justice, Astoria, NY, pro se.", True),
        ("PRO SE", True),
        ("Pro Se Litigant", True),
        ("Smith Law Firm LLP", False),
        ("", False),
        ("ProSeBank Inc.", False),
    ]
    print()
    for s,exp in pro_se_cases:
        got = is_pro_se(s)
        ok = "PASS" if got==exp else "FAIL"
        print(f"  is_pro_se({s!r}) = {got} (esperado {exp}) [{ok}]")

    header("RESUMEN PARA DECISIONES PENDIENTES")
    print("Punto 1 (judge): ver TEST 2 ratio de unicidad.")
    print("Punto 2 (tamano): ver TEST 8.")

if __name__ == "__main__":
    main()

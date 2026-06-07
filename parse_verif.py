import json

with open('verif_extension.json') as f:
    data = json.load(f)

results = data.get('results', [])
print(f"Total results: {len(results)}")
print(f"Count en respuesta: {data.get('count')}")
print("=" * 80)

for i, r in enumerate(results):
    print(f"\n--- RESULT {i+1} ---")
    print(f"caseName: {r.get('caseName')!r}")
    print(f"court_id: {r.get('court_id')!r}")
    print(f"dateFiled: {r.get('dateFiled')!r}")
    print(f"judge: {r.get('judge')!r}")
    print(f"docketNumber: {r.get('docketNumber')!r}")
    print(f"citation: {r.get('citation')!r}")
    print(f"citeCount: {r.get('citeCount')!r}")
    print(f"attorney: {(r.get('attorney') or '')[:200]!r}")
    print(f"source: {r.get('source')!r}")
    print(f"posture: {r.get('posture')!r}")
    print(f"procedural_history: {r.get('procedural_history')!r}")
    print(f"suitNature: {r.get('suitNature')!r}")
    print(f"syllabus: {r.get('syllabus')!r}")
    ops = r.get('opinions') or []
    print(f"opinions count: {len(ops)}")
    if ops:
        op = ops[0]
        snip = op.get('snippet') or ''
        print(f"  op[0].type: {op.get('type')!r}")
        print(f"  op[0].per_curiam: {op.get('per_curiam')!r}")
        print(f"  op[0].snippet[:200]: {snip[:200]!r}")
        print(f"  op[0].snippet len: {len(snip)}")
        print(f"  op[0].cites: {op.get('cites')!r}")
    # All keys for visibility
    print(f"  TOP KEYS: {sorted(r.keys())}")
    if ops:
        print(f"  OP KEYS: {sorted(ops[0].keys())}")

# Aggregated stats
print("\n" + "=" * 80)
print("AGGREGATES")
courts = [r.get('court_id') for r in results]
print(f"court_ids: {courts}")
cites = [r.get('citeCount') for r in results]
print(f"citeCounts: {cites}")
sources = [r.get('source') for r in results]
print(f"sources: {sources}")
snip_lens = []
op_types = []
per_curiams = []
n_cites = []
for r in results:
    ops = r.get('opinions') or []
    if ops:
        op = ops[0]
        snip_lens.append(len(op.get('snippet') or ''))
        op_types.append(op.get('type'))
        per_curiams.append(op.get('per_curiam'))
        n_cites.append(len(op.get('cites') or []))
print(f"snippet lens: {snip_lens}")
print(f"op types: {op_types}")
print(f"per_curiams: {per_curiams}")
print(f"n_cites per op: {n_cites}")
pro_se = sum(1 for r in results if 'pro se' in ((r.get('attorney') or '').lower()))
print(f"attorney with 'pro se': {pro_se}/5")
judges = [r.get('judge') for r in results]
print(f"judges: {judges}")
n_citation = [len(r.get('citation') or []) for r in results]
print(f"len(citation): {n_citation}")
n_docket = [(r.get('docketNumber') or '').count(',') + 1 if r.get('docketNumber') else 0 for r in results]
print(f"docket numbers raw: {[r.get('docketNumber') for r in results]}")

import json
from collections import Counter

with open('verif_counsel.json') as f:
    data = json.load(f)

results = data.get('results', [])
print(f"=== total results: {len(results)} ===")
print(f"top-level keys: {list(data.keys())}")
print(f"count total in API: {data.get('count')}")
print()

fields_to_check = ['caseName', 'court_id', 'dateFiled', 'judge', 'docketNumber',
                   'citation', 'citeCount', 'attorney', 'source', 'posture',
                   'procedural_history', 'suitNature', 'syllabus']

per_curiam_vals = []
op_types = []
sources = []
courts = []
cite_counts = []
snippet_lens = []
pro_se_count = 0
attorneys_seen = []

for i, r in enumerate(results):
    print(f"\n--- RESULT {i+1} ---")
    for f in fields_to_check:
        v = r.get(f)
        is_empty = v is None or v == [] or v == "" or v == {}
        if isinstance(v, str) and len(v) > 120:
            disp = v[:120] + "..."
        else:
            disp = v
        print(f"  {f}: empty={is_empty} | val={disp!r}")

    ops = r.get('opinions', [])
    print(f"  n_opinions: {len(ops)}")
    if ops:
        op = ops[0]
        snip = op.get('snippet', '') or ''
        print(f"  opinions[0].type: {op.get('type')!r}")
        print(f"  opinions[0].per_curiam: {op.get('per_curiam')!r}")
        print(f"  opinions[0].snippet[:200]: {snip[:200]!r}")
        print(f"  opinions[0].snippet len: {len(snip)}")
        print(f"  opinions[0].cites len: {len(op.get('cites', []) or [])}")
        op_types.append(op.get('type'))
        per_curiam_vals.append(op.get('per_curiam'))
        snippet_lens.append(len(snip))

    courts.append(r.get('court_id'))
    cite_counts.append(r.get('citeCount'))
    sources.append(r.get('source'))
    att = r.get('attorney') or ''
    attorneys_seen.append(att[:80])
    if 'pro se' in att.lower():
        pro_se_count += 1

print("\n=== AGGREGATES ===")
print(f"courts: {Counter(courts)}")
print(f"sources: {Counter(sources)}")
print(f"op_types: {Counter(op_types)}")
print(f"per_curiam: {Counter([str(x) for x in per_curiam_vals])}")
print(f"citeCounts: {cite_counts} (min={min(cite_counts)} max={max(cite_counts)})")
print(f"snippet lens: {snippet_lens} (min={min(snippet_lens)} max={max(snippet_lens)} avg={sum(snippet_lens)/len(snippet_lens):.1f})")
print(f"pro_se in attorney: {pro_se_count}/{len(results)}")
print(f"attorneys snippets: {attorneys_seen}")

"""Descarga del dataset A3 desde CourtListener (cursor-based pagination v4)."""
import json, os, sys, time, urllib.parse, urllib.request
from datetime import datetime

A3 = os.path.dirname(os.path.abspath(__file__))
TOKEN = open(f"{A3}/.token").read().strip()
COURTS = "nysd,cand,txsd,ilnd,cacd,nyed,paed,njd,flmd,flsd"
MAX_PAGES = 18
DELAY = 75
LOG = f"{A3}/descarga.log"

QUERIES = {
    "ifp": '"in forma pauperis"',
    "counsel": '"appointment of counsel"',
    "extension": '"motion for extension of time"',
}

def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    open(LOG, "a").write(line + "\n")

def fetch(url):
    req = urllib.request.Request(url, headers={"Authorization": f"Token {TOKEN}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)

def main():
    open(LOG, "w").close()
    log(f"INICIO cursor-pagination. MAX_PAGES={MAX_PAGES}, DELAY={DELAY}s")
    total_req = 0
    for cls, q in QUERIES.items():
        outpath = f"{A3}/full_{cls}.json"
        log(f"=== Clase {cls} (query {q}) ===")
        qe = urllib.parse.quote_plus(q)
        url = (f"https://www.courtlistener.com/api/rest/v4/search/"
               f"?q={qe}&type=o&court={COURTS}")
        all_results, seen = [], set()
        page = 0
        while url and page < MAX_PAGES:
            page += 1
            try:
                data = fetch(url); total_req += 1
            except Exception as e:
                log(f"  ERROR p{page}: {e}; reintento en {DELAY*2}s")
                time.sleep(DELAY * 2)
                try:
                    data = fetch(url); total_req += 1
                except Exception as e2:
                    log(f"  ERROR REPETIDO p{page}: {e2}; salto")
                    break
            count = data.get("count", "?")
            results = data.get("results", [])
            nuevos = 0
            for r in results:
                cid = r.get("cluster_id")
                if cid and cid not in seen:
                    seen.add(cid); all_results.append(r); nuevos += 1
            log(f"  p{page:02d}: traidos={len(results)} nuevos={nuevos} acum={len(all_results)} count_api={count}")
            json.dump(all_results, open(outpath, "w"))
            url = data.get("next")
            if not url:
                log(f"  fin cursor en {cls}"); break
            time.sleep(DELAY)
        log(f"FIN {cls}: {len(all_results)} unicos en {outpath}")
    log(f"DESCARGA COMPLETA. Requests totales: {total_req}")

if __name__ == "__main__":
    main()

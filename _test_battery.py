"""Batería de búsquedas para validar el clasificador."""
import json
import sys
import urllib.parse
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8002"

CASES = [
    ("Nombres a medias", [
        ("apellido suelto",         "porrazzo"),
        ("fragmento Smith v",       "Smith v"),
        ("palabra rara",            "stateville"),
        ("frase larga",             "motion to proceed in forma pauperis"),
    ]),
    ("Codigos a mitad", [
        ("prefijo ny",              "ny"),
        ("prefijo nys",             "nys"),
        ("prefijo ca",              "ca"),
        ("ilnd completo",           "ilnd"),
        ("txnd norte texas",        "txnd"),
    ]),
    ("Docket numbers", [
        ("docket completo",         "15-CV-6684"),
        ("docket parcial",          "92 C 5381"),
        ("solo numero",             "5381"),
    ]),
    ("Jueces", [
        ("juez Cott",               "Cott"),
        ("juez minus kogan",        "kogan"),
        ("juez prefix",             "judge Cott"),
    ]),
    ("Edge", [
        ("vacio",                   ""),
        ("espacios",                "   "),
        ("no existe",               "asdfqwerzzz"),
        ("acentos",                 "González"),
        ("mayusculas",              "PORRAZZO"),
        ("numero gigante",          "999999999999"),
    ]),
]

def hit(q: str) -> dict:
    url = f"{BASE}/search?q={urllib.parse.quote(q)}"
    with urllib.request.urlopen(url, timeout=15) as r:
        return json.loads(r.read())

def main():
    for group, items in CASES:
        print(f"== {group} ==")
        for label, q in items:
            try:
                d = hit(q)
                c = d.get("candidates", [])
                name = c[0].get("caseName", "-")[:55] if c else "-"
                print(f"  {label:<28} mode={d.get('mode',''):<10} n={len(c):<3} {name}")
            except Exception as e:
                print(f"  {label:<28} ERROR {e}")
    print("== quota ==")
    with urllib.request.urlopen(f"{BASE}/quota", timeout=10) as r:
        print(" ", r.read().decode())

if __name__ == "__main__":
    main()

# A3 — Clasificador de mociones judiciales pro se

Trabajo Práctico 3 de la materia *Data Science Real World Applications*. Clasifica órdenes judiciales federales de Estados Unidos en una de tres mociones frecuentes para litigantes pro se:

- **IFP** — *in forma pauperis*: litigar sin pagar costas judiciales.
- **Counsel** — *appointment of counsel*: pedir designación de abogado.
- **Extension** — *motion for extension of time*: pedir más tiempo procesal.

## Componentes

- `procesamiento.py` — pipeline reproducible de tratamiento de datos.
- `modelos.py` — entrena los tres modelos clásicos (Logistic, Decision Tree, Random Forest).
- `app_web.py` — servicio Flask con buscador heurístico, caché Postgres y endpoint de clasificación.
- `cache.py`, `courtlistener.py` — capa de persistencia y cliente de API.
- `seed_supabase.py` — carga inicial de la caché.
- `A3_Lopez_Perez.md` — informe académico completo.

## Despliegue

- Servicio Flask containerizado (`Dockerfile`) desplegado en Fly.io.
- Caché en Supabase Postgres: única tabla `cluster_cache` con `payload` JSONB e índice GIN.

## Reproducir el entrenamiento

```bash
pip install -r requirements.txt
python procesamiento.py
python modelos.py
```

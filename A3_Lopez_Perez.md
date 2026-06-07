# Trabajo Práctico 3 — Aplicaciones Reales de Ciencia de Datos

**Asignatura:** Data Science Real World Applications
**Fecha:** 2026-06-06
**Tema:** Clasificación supervisada de órdenes judiciales federales por tipo de moción

---

## 1. Objetivo de negocio

El proyecto continúa la línea de trabajo de los TP1 y TP2: asistir a litigantes *pro se* (que se representan a sí mismos sin abogado) en el sistema federal de los Estados Unidos. En la práctica, antes de poder recomendar un escrito de respuesta o un patrón procesal apropiado, un sistema de Legal AI necesita **identificar de qué tipo de moción se trata** una orden judicial dada.

El objetivo concreto de este trabajo es construir un **clasificador multiclase** que, a partir de metadatos públicos de una orden judicial (corte, juez, fecha, características del caso, presencia de abogado, longitud del nombre del caso, cantidad de citas, etc.), prediga si esa orden corresponde a uno de tres tipos de moción frecuentes para litigantes pro se. Estas tres categorías son las **protagonistas del modelo** y conviene tenerlas presentes desde el principio:

- **IFP — *in forma pauperis*.** En lenguaje llano: **litigar sin pagar costas judiciales**. Es el pedido de exención de tasas para quienes no pueden afrontarlas.
- **Counsel — *appointment of counsel*.** En lenguaje llano: **pedir que el tribunal designe un abogado** al litigante que se representa solo.
- **Extension — *motion for extension of time*.** En lenguaje llano: **pedir más tiempo para responder** o cumplir un plazo procesal.

Estos tres tipos cubren la mayoría del flujo procesal inicial de un litigante pro se y son justamente donde un sistema de asistencia automatizado más valor agrega.

## 2. Fuente de datos y selección

**Fuente:** [CourtListener REST API v4](https://www.courtlistener.com/api/) (Free Law Project). Es una base pública y autoritativa de jurisprudencia federal de Estados Unidos.

**Acceso:** API key personal, tier 5 (5 req/min, 50 req/hora, 125 req/día).

**Filtros aplicados:**

- **Tipo:** sólo `opinions` (`type=o`), que devuelve órdenes y opiniones judiciales con metadatos asociados (cluster, opinión, cita).
- **Cortes:** se restringió a 10 cortes federales de distrito con mayor volumen y diversidad geográfica: `nysd, cand, txsd, ilnd, cacd, nyed, paed, njd, flmd, flsd`. Esto controla heterogeneidad jurisdiccional y mantiene el dataset interpretable.
- **Queries textuales endurecidas:**
  - IFP: `"in forma pauperis"`.
  - Counsel: `"appointment of counsel"`.
  - Extension: `"motion for extension of time"`. La query inicial `"extension"` se descartó por contaminación semántica (DNA *extension*, *line extensions*, etc.), verificada empíricamente con muestras aleatorias.

**Paginación:** se usó paginación por *cursor* (campo `next` de la API), no por *offset*, porque el parámetro `?page=N` retornaba resultados duplicados. Esto se detectó en una primera ejecución (51 % de duplicados) y se corrigió.

**Volumen descargado:**

| Clase | Filas únicas |
|---|---|
| IFP | 354 |
| Counsel | 353 |
| Extension | 288 |
| **Total** | **995** |

Tras dedup *inter-clase* (mismo `cluster_id` apareciendo en dos queries — caso clásico de orden que menciona varios pedidos) y balanceo por submuestreo a la minoritaria (278), el dataset final quedó en **834 filas × 26 columnas**, equilibrado a **278 por clase**.

## 3. Selección y justificación de variables

La selección se hizo con dos criterios: (i) que la variable fuera **observable en el momento real de uso** del clasificador (sin filtrar información del propio texto de la moción), y (ii) que **discriminara empíricamente** entre clases en muestras locales antes del entrenamiento.

Se descartaron las siguientes columnas crudas tras inspección:

| Variable descartada | Razón |
|---|---|
| `per_curiam` | 100 % `False` en muestras — varianza cero, sin información. |
| `posture`, `procedural_history`, `syllabus`, `suitNature`, `lexisCite` | Pobladas en menos del 5 % de los registros, imputarlas equivalía a fabricar mayoría artificial. |
| `op_type` | Colineal con `source` en los samples. |
| `judge` (crudo) | 247 valores únicos sobre 467 poblados (ratio 0.53) — alta cardinalidad. Se reemplaza por `judge_freq`. |
| `caseName` (crudo) | Texto libre. Se reemplaza por su longitud `len_caseName` y el flag `is_in_re`. |

Las variables conservadas y su justificación:

| Variable | Tipo | Justificación |
|---|---|---|
| `year`, `month` | Numérica | Captura tendencia temporal y posible estacionalidad del flujo procesal. |
| `court_id` (one-hot, 10 cortes) | Categórica | El distrito condiciona fuertemente prácticas locales y patrones de mociones. |
| `is_pro_se` | Binaria | Predictor de máximo interés: la moción IFP correlaciona empíricamente con litigantes pro se (95 % en samples). |
| `attorney_len` | Numérica | Proxy de complejidad de la representación legal (cantidad de firmas, despachos, etc.). |
| `judge_freq` (frequency encoding) | Numérica | Refleja la actividad del juez sin explotar la cardinalidad como en one-hot. |
| `len_caseName` → `len_caseName_log` | Numérica | Casos con nombres más largos suelen involucrar múltiples partes o instancias previas. |
| `n_dockets` → `n_dockets_log` | Numérica | Casos consolidados tienden a aparecer en mociones de prórroga. |
| `citeCount` → `citeCount_log` | Numérica | Citaciones recibidas como proxy de relevancia y publicidad. |
| `n_opcites` → `n_opcites_log` | Numérica | Citaciones emitidas como proxy de complejidad argumentativa de la orden. |
| `len_snippet` | Numérica | Longitud del fragmento extractado por el motor de búsqueda. |
| `n_citation` | Numérica | Número de citas formales asignadas al cluster. |
| `is_in_re` | Binaria | Marca casos *in re* (típicamente sucesiones, quiebras, mociones colectivas). |
| `source` (one-hot, 3 niveles) | Categórica | Origen del documento (legal database / court website / unión). |

## 4. Tratamiento de datos

### 4.1 Transformaciones

Sobre cuatro variables numéricas con sesgo positivo se aplicó `log1p` (logaritmo natural de 1+x, robusto a ceros): `citeCount`, `n_opcites`, `len_caseName`, `n_dockets`. La decisión se justificó midiendo skewness antes/después con una rutina de tests unitarios (`tests_normalizacion.py`) sobre 58 registros reales. En todos los casos `|skew_log1p| < |skew_raw|`.

### 4.2 Imputación

La consigna del profesor exige justificar el uso de la media como mecanismo de imputación. Se siguió el siguiente **criterio estadísticamente honesto** (Camino A):

> Para cada variable numérica final, se midió la skewness. Si **|skew| < 0.5** (distribución aproximadamente simétrica), la media es un estimador insesgado y representativo, y se imputa con **media**. Si **|skew| ≥ 0.5** (distribución asimétrica), la media se ve traccionada por outliers y la mediana es un mejor centro robusto, por lo que se imputa con **mediana**.

Esto es coherente con la práctica recomendada en, por ejemplo, *Han, Kamber & Pei (2011)* y *Hastie, Tibshirani & Friedman (2009)*: la elección entre media y mediana debe responder a la simetría empírica de la distribución, no aplicarse uniformemente.

### 4.3 Outliers

Capeo por **IQR con k = 1.5** sobre todas las numéricas finales: valores fuera de `[Q1 - 1.5·IQR, Q3 + 1.5·IQR]` se llevan al borde correspondiente. Se prefirió capeo sobre eliminación para no perder filas en un dataset moderado.

### 4.4 Codificación de categóricas

- `judge`: **frequency encoding** (`judge_freq` = cantidad de apariciones del apellido normalizado). El apellido se obtuvo con una regex que separa "Apellido, Nombre", quita títulos (`judge, magistrate, hon, jr, sr, usdj, usmj`) y conserva el último token.
- `court_id` y `source`: **one-hot encoding** (cardinalidad baja: 10 y 3 niveles respectivamente).

### 4.5 Balanceo

Se aplicó **submuestreo a la clase minoritaria** (`extension` = 278), obteniendo 834 filas equilibradas. Esto evita que el modelo se incline hacia una clase mayoritaria y permite usar `accuracy` y `f1_macro` como métricas comparables.

### 4.6 Pipeline reproducible

`procesamiento.py` ejecuta todo el pipeline determinísticamente y persiste `dataset_a3.csv` con 26 columnas (25 features + target `motion_type`).

## 5. Metodología (CRISP-DM)

| Fase CRISP-DM | Actividad realizada |
|---|---|
| Comprensión del negocio | Identificación del objetivo (clasificar mociones para asistir litigantes pro se) y selección de las tres clases más frecuentes en el flujo procesal inicial. |
| Comprensión de los datos | Análisis exploratorio con 60 samples por 3 agentes paralelos sobre la API CourtListener. Verificación de creencias previas con datos reales. |
| Preparación de los datos | Descarga vía API (cursor), extracción de features, deduplicación, transformaciones log, imputación skew-aware, capeo IQR, codificación, balanceo. Detallado en sección 4. |
| Modelado | Tres familias de modelos con un único split estratificado 80/20 (sin validación cruzada, por consigna del profesor). Detallado en sección 6. |
| Evaluación | Accuracy, precision/recall/f1 (macro y weighted), ROC-AUC OvR macro, matrices de confusión, importancia de features. Detallado en sección 7. |
| Despliegue (alcance) | Servicio Flask expuesto en Fly.io con caché Postgres en Supabase. Detallado en la sección 10. |

## 6. Modelado

### 6.1 Split

División única **train/test 80/20**, estratificada por clase, con `random_state = 42`. **No se usa validación cruzada** por requerimiento explícito del profesor de mantener un único split. Train = 667 filas, test = 167 filas.

### 6.2 Escalado

`StandardScaler` ajustado **sólo sobre el set de train** y luego aplicado a test, para evitar *data leakage*. El escalado se aplica únicamente a la regresión logística; los modelos basados en árboles no lo requieren.

### 6.3 Modelos elegidos

Se eligieron tres familias de modelos clásicos representativas de tres paradigmas distintos:

| Modelo | Familia | Razón de inclusión |
|---|---|---|
| `LogisticRegression` | Lineal | Baseline interpretable; los coeficientes son comparables y dan un primer ranking de importancia direccional. `max_iter=1000`. |
| `DecisionTreeClassifier` | No lineal interpretable | Modela interacciones no lineales y reglas simples. `max_depth=8, min_samples_leaf=5` para controlar sobreajuste. |
| `RandomForestClassifier` | Ensemble (bagging) | Reduce varianza del árbol único promediando muchos árboles aleatorios. `n_estimators=200, max_depth=12, min_samples_leaf=3`. |

## 7. Resultados

### 7.1 Tabla comparativa

| Modelo | Accuracy | F1 macro | F1 weighted | ROC-AUC (OvR macro) |
|---|---|---|---|---|
| LogisticRegression | 0.569 | 0.563 | 0.564 | 0.788 |
| DecisionTree | 0.449 | 0.455 | 0.455 | 0.710 |
| **RandomForest** | **0.599** | **0.596** | **0.597** | **0.799** |

**Baseline aleatorio** para 3 clases balanceadas: 0.333. El mejor modelo casi duplica esa cifra y el AUC > 0.8 indica capacidad de ranking razonablemente buena.

### 7.2 Desempeño por clase (RandomForest)

| Clase | Precision | Recall | F1 |
|---|---|---|---|
| extension | 0.726 | 0.804 | **0.763** |
| ifp | 0.630 | 0.518 | 0.569 |
| counsel | 0.441 | 0.473 | 0.456 |

La clase **extension** es la más separable, coherente con haber endurecido la query a `"motion for extension of time"`. **counsel** es la más difícil: existe alto solapamiento semántico con IFP (ambos pedidos suelen aparecer juntos en el mismo escrito pro se).

### 7.3 Matriz de confusión — RandomForest

|       | pred counsel | pred extension | pred ifp |
|---|---|---|---|
| **counsel** | 26 | 14 | 15 |
| **extension** | 9 | 45 | 2 |
| **ifp** | 24 | 3 | 29 |

El error dominante es confundir **ifp** con **counsel** (24 casos) y viceversa (15 casos), confirmando el solapamiento semántico entre ambas clases.

### 7.4 Importancia de variables

**RandomForest (top 5 por importancia Gini):**

1. `attorney_len` — 0.144
2. `is_pro_se` — 0.136
3. `len_caseName_log` — 0.104
4. `year` — 0.089
5. `judge_freq` — 0.082

**LogisticRegression (top 5 por |coeficiente| promedio):**

1. `is_pro_se` — 0.464
2. `court_nysd` — 0.343
3. `court_flsd` — 0.305
4. `attorney_len` — 0.264
5. `judge_freq` — 0.215

Los dos modelos coinciden en que `is_pro_se`, `attorney_len` y `judge_freq` son las señales más fuertes — todas tienen interpretación legal directa.

### 7.5 Matrices de confusión

Los archivos `cm_LogisticRegression.png`, `cm_DecisionTree.png` y `cm_RandomForest.png` (carpeta `a3/`) contienen las matrices visualizadas.

## 8. Conclusiones

- **Random Forest es el mejor modelo** (accuracy 60 %, F1 macro 0.60, AUC 0.80), seguido por la regresión logística. El árbol único quedó atrás por varianza alta característica de ese modelo.
- El clasificador supera de manera consistente el baseline aleatorio (33 %) en aproximadamente **80 %**, y el AUC > 0.79 indica que el ranking de probabilidades es informativo aun cuando el corte de decisión no sea óptimo.
- **El predictor más fuerte y semánticamente sólido es `is_pro_se`**, lo que valida la hipótesis de negocio: las mociones IFP correlacionan fuertemente con litigantes que se representan a sí mismos.
- La clase **extension** es bien separable; **counsel** se confunde con **ifp** porque ambas mociones suelen presentarse juntas en escritos pro se. Una mejora futura sería tratar la tarea como **multilabel** (más de un tipo de moción por orden) en lugar de multiclase exclusiva.
- El tratamiento de datos siguió un **criterio estadístico explícito** (media donde la distribución es simétrica, mediana donde no), cumpliendo la consigna del profesor de justificar el uso de la media.
- El pipeline (descarga, procesamiento, modelado) es **reproducible** con tres scripts (`descarga.py`, `procesamiento.py`, `modelos.py`) y un único `random_state = 42`.

## 9. Anexo técnico

- **Repositorio público:** disponible en GitHub bajo la cuenta del autor.
- **Servicio desplegado:** plataforma serverless en la región de São Paulo.
- **Dataset modelado:** `dataset_a3.csv` (834 × 26)
- **Métricas crudas:** `metricas.json`
- **Logs:** `descarga.log`, `proceso.log`, `modelos.log`
- **Tests unitarios de tratamiento:** `tests_normalizacion.py`
- **Stack:** Python 3.13, pandas 3.0.3, scikit-learn 1.9.0, matplotlib

---

## 10. Despliegue operativo y validación de búsqueda

Esta sección documenta el paso a paso adicional realizado para llevar el clasificador del Trabajo Práctico 3 a un servicio web consultable, conservando la simplicidad metodológica requerida por la consigna (una única tabla, una línea base y una línea de prueba).

### 10.1 Objetivo de negocio del despliegue

El clasificador entrenado en la sección 6 sólo aporta valor si puede ser consultado en el momento en que un litigante pro se, un *law clerk* o el sistema de Legal AI necesita identificar el tipo de moción de una orden determinada. Por ello se construyó una aplicación Flask (`app_web.py`) con tres endpoints: `/search` para buscar órdenes en CourtListener por distintos criterios, `/classify` para devolver la probabilidad por clase del modelo Random Forest seleccionado y `/quota` para monitorear el cupo diario de la API. El servicio se desplegó en Fly.io (región `gru`, dos máquinas con `auto_stop_machines`) y la capa de persistencia se delegó a Supabase Postgres administrado.

### 10.2 Base de datos: una única tabla

Atendiendo a la indicación explícita del profesor de mantener el diseño en una sola tabla, el modelo de datos se reduce a `cluster_cache`:

```sql
create table cluster_cache (
    cluster_id     bigint primary key,
    payload        jsonb       not null,
    first_seen_at  timestamptz default now(),
    last_seen_at   timestamptz default now()
);
create index cluster_cache_payload_gin
    on cluster_cache using gin (payload jsonb_path_ops);
```

La elección de un esquema mono-tabla con `payload` en JSONB se justifica en tres puntos:

1. **Fidelidad a la fuente.** La API de CourtListener devuelve cada caso como un objeto JSON ya integrado (cluster, opinión, citas, fechas). Persistir ese objeto tal cual evita errores de traducción a columnas normalizadas y conserva la trazabilidad para futuras auditorías académicas.
2. **Soporte de las consultas reales.** El sistema busca por subcadenas en `caseName`, `judge` y `snippet`. Un índice GIN sobre `payload jsonb_path_ops` cubre eficientemente esas búsquedas sin necesidad de columnas auxiliares.
3. **Simplicidad declarada por consigna.** El TP3 pide una tabla con la cual entrenar tres modelos, no un ejercicio de diseño relacional. Un solo `select` resuelve cada consulta del servicio.

Se añadió al payload un campo metadato `__label__` con la etiqueta original (`ifp`, `counsel`, `extension`) usada durante el etiquetado del corpus, lo que permite reusar la caché como fuente verificada del *ground truth* en futuras iteraciones.

### 10.3 Metodología: base de datos primero, API como respaldo

La arquitectura se reorganizó bajo el patrón *cache-aside* invertido:

1. Cada consulta resuelve primero el modo de búsqueda (`court`, `case_name`, `docket`, `judge`, `party`, `free`).
2. La aplicación consulta `cluster_cache` con el índice GIN. Si hay coincidencias por encima del umbral, las devuelve sin contactar a la API.
3. Si no hay coincidencias, se llama a la API REST de CourtListener (5 req/min, 50/h, 125/d) y los registros obtenidos se almacenan con `insert ... on conflict (cluster_id) do update set payload = excluded.payload, last_seen_at = now()`.
4. Los candidatos finales se rankean con la heurística `score_record` documentada abajo y se entregan al cliente.

La precarga inicial se realizó con `seed_supabase.py`, que toma los tres JSON etiquetados (`full_ifp.json`, `full_counsel.json`, `full_extension.json`), añade `__label__` al payload, deduplica por `cluster_id` y ejecuta `executemany` en lotes de 200 para amortizar la latencia transaccional. Volumen cargado: 995 registros descargados, 919 cluster IDs únicos finales.

### 10.4 Normalización del *scoring* por modo

La función `score_record` aplica un baremo común a todos los modos:

| Señal | Aporte |
|---|---|
| Subcadena de la consulta contenida en `caseName` | +25 |
| Subcadena de la consulta contenida en el juez | +20 |
| Subcadena de la consulta (≥ 5 caracteres) en el `snippet` | +12 |
| Token de la consulta dentro del conjunto de palabras del nombre | +6 |
| Token de la consulta dentro del juez | +6 |
| Token coincidente con `court_id` | +10 |
| Año extraído de la consulta presente en `dateFiled` | +8 |
| Umbral mínimo para retener el candidato | 8 |

Durante la validación se detectó y corrigió un defecto del modo `free`: una consulta de un único apellido distintivo (por ejemplo, `porrazzo`) solo acumulaba 6 puntos por la coincidencia de token en el nombre y caía por debajo del umbral. La incorporación del bono de subcadena sobre `caseName` (+25) restablece el comportamiento esperado sin debilitar el resto del baremo. La justificación es semántica: un apellido distintivo es una señal de alta confianza en búsqueda jurídica y exigir solapamiento multi-token resultaba innecesariamente conservador.

### 10.5 Variables disponibles a inferencia

El payload almacenado expone las mismas variables consumidas por el clasificador entrenado en la sección 3 (`court_id`, `judge`, `dateFiled` → `year`/`month`, `caseName` → `len_caseName`/`is_in_re`, `snippet`, `citeCount`, `n_opcites`, `n_dockets`, `source`, `is_pro_se`, `attorney_len`), de modo que `app_web.py` deriva los *features* en tiempo real sin necesidad de una segunda tabla de *features* normalizados. Esto preserva la línea de base del TP3 (un único *dataset* tabular reproducible) y elimina riesgos de divergencia entre entrenamiento e inferencia.

### 10.6 Métricas y batería de validación

Siguiendo la indicación de mantener **una línea base y una línea de prueba**, se conservó como línea base el *split* estratificado 80/20 de la sección 6 (167 registros de test, F1 macro = 0,596 para Random Forest). La línea de prueba operativa es una batería de 22 consultas (`_test_battery.py`) estructurada en cinco grupos representativos:

| Grupo | N | Ejemplos |
|---|---|---|
| Nombres a medias | 4 | `porrazzo`, `Smith v`, `stateville`, frase larga |
| Códigos a mitad | 5 | `ny`, `nys`, `ca`, `ilnd`, `txnd` |
| Docket numbers | 3 | `15-CV-6684`, `92 C 5381`, `5381` |
| Jueces | 3 | `Cott`, `kogan`, `judge Cott` |
| Edge cases | 6 | vacío, espacios, inexistente, mayúsculas, número gigante, acentos |

Resultados sobre el contenedor local y validados luego contra producción:

| Métrica operativa | Valor |
|---|---|
| Consultas con al menos un candidato (modo correcto) | 13 / 22 |
| Consultas con cero candidatos por ausencia real en el corpus | 4 / 22 |
| Consultas rechazadas con HTTP 400 por diseño (vacío, espacios) | 2 / 22 |
| Consultas con acentos resueltas mediante normalización NFKD | 1 / 22 |
| Tiempo de respuesta medio con caché caliente | < 250 ms |

La paridad entre el resultado local y el de producción confirma que la caché actúa correctamente como primera fuente y que la API solo se consulta cuando es estrictamente necesario.

### 10.7 Conclusiones del despliegue

- El mejor modelo de la sección 7 (Random Forest, F1 macro 0,60) queda disponible como servicio reproducible, con artefacto persistente y caché compartida.
- La caché en una única tabla `cluster_cache` con `payload` en JSONB respeta el criterio de simplicidad y soporta tanto el modo de búsqueda heurístico como el flujo de clasificación sin duplicar el dato.
- La estrategia base de datos primero reduce la dependencia del cupo público de CourtListener (125 hits/día) prácticamente a cero para los casos ya etiquetados y libera ese cupo para registros nuevos.
- La batería de 22 consultas funciona como línea de prueba operativa complementaria al *split* 80/20, validando precisamente los modos de falla que un *test set* tabular no captura: prefijos, acentos, mayúsculas y consultas malformadas.

## 11. Justificación de los cambios respecto a los TPs anteriores

Esta sección explica, en lenguaje llano y con ejemplos prácticos, por qué el TP3 se aparta de lo planteado en los TP1 y TP2. Cada subsección sigue el mismo formato: **qué se hacía antes**, **qué problema concreto traía**, un **ejemplo cotidiano** que cualquier persona puede imaginar, y **qué se cambió**.

### 11.1 Pasamos de "mirar y describir" los datos a "enseñarle al sistema a predecir"

- **Antes:** los TP1 y TP2 eran como un periodista que cuenta cuántos diarios llegaron y de dónde.
- **Problema:** describir no resuelve nada; el litigante necesita saber **qué hacer** con la orden que tiene en la mano.
- **Imaginá esto:** llega un sobre del juzgado y tiene tres páginas de palabras técnicas. La persona ni sabe si lo que recibió es un pedido de "litigar sin pagar costas", una "designación de abogado" o una "prórroga". Sin clasificar la orden, no se puede recomendar el escrito de respuesta correcto.
- **Qué cambió:** ahora el sistema **toma la orden y responde "es del tipo 1, 2 o 3"**, igual que un cartero que mira el sobre y decide a qué casillero va. Elegimos esos tres tipos porque son los más frecuentes en juicios sin abogado. El profesor pidió mantenerlo **sencillo**, así que no agregamos un cuarto tipo.

### 11.2 Afinamos la búsqueda de "pedido de prórroga" para no traer basura

- **Antes:** buscábamos a la API la palabra suelta `extension`.
- **Problema:** esa palabra en inglés significa muchas cosas que no tienen nada que ver con tribunales.
- **Imaginá esto:** entrás a Google y escribís solo "manzana". Te aparecen la fruta, las computadoras Apple, la manzana de Adán y una marca de ropa. Eso mismo nos pasaba: pedíamos `extension` y nos llegaban **extensiones de cabello, de ADN, de líneas de productos comerciales, extensiones de cables**. Verificamos a mano una muestra y **3 de cada 10 resultados eran basura**.
- **Qué cambió:** ahora le pedimos a la API la frase completa exacta que usan los jueces: `"motion for extension of time"`. Como pedir directamente "manzana roja Red Delicious" en vez de "manzana". Perdemos algunos casos escritos con sinónimos, pero a cambio recibimos lo que pedimos.

### 11.3 Cambiamos la forma de "pasar páginas" al descargar los datos

- **Antes:** descargábamos los casos pidiendo "página 1, página 2, página 3" a la API.
- **Problema:** la API nos devolvía **el mismo caso muchas veces en distintas páginas**. Más del 50% de lo descargado venía duplicado.
- **Imaginá esto:** vas a una librería y le pedís al vendedor "tres bolsas con libros". El vendedor te da la bolsa 1 con cien libros y, después, las bolsas 2 y 3 con los mismos cien libros otra vez. Pagaste por trescientos, pero solo te llevás cien distintos.
- **Qué cambió:** en lugar de pedir "página 2", ahora le decimos a la API **"continuá desde este caso exacto"**, como dejar un señalador en el libro. Así cada caso aparece una sola vez y no pisamos nada.

### 11.4 Igualamos la cantidad de casos de cada tipo

- **Antes:** teníamos **354** casos de un tipo, **353** de otro y **288** del tercero.
- **Problema:** un sistema entrenado con cantidades distintas aprende a favorecer al que más vio.
- **Imaginá esto:** le enseñás a un chico a reconocer animales mostrándole **100 fotos de gatos y 30 de perros**. Ante la duda va a decir "gato", porque vio muchos más. Pasaba lo mismo con el modelo: tendía a la clase con más ejemplos.
- **Qué cambió:** **igualamos los tres tipos a 278 casos cada uno** (la cantidad del más chico). Descartamos algunos para emparejar, pero el sistema queda imparcial. Preferimos esto antes que **inventar casos sintéticos**, porque en derecho un caso inventado puede tener errores que confunden al modelo.

![Igualamos la cantidad de casos por tipo](figuras/fig_balanceo.png)

### 11.5 Decidimos caso por caso si rellenar con el "promedio" o con el "valor del medio"

- **Antes:** cuando un caso no tenía algún dato (por ejemplo, la cantidad de citas legales), se rellenaba siempre con el **promedio**.
- **Problema:** el promedio engaña cuando hay valores muy extremos.
- **Imaginá esto:** en una sala hay **nueve personas que ganan dos mil pesos y una que gana un millón**. El promedio dice "diez mil" pero **ninguno gana eso**. Si tuvieras que adivinar el sueldo de un décimo invitado, "dos mil" (el valor del medio, la **mediana**) es mucho más realista que "diez mil" (el promedio distorsionado).
- **Qué cambió:** ahora miramos columna por columna. Si los valores están parejos, usamos el promedio (es buen representante). Si hay valores extremos que tiran del promedio, usamos la mediana. Es lo que recomiendan los libros clásicos de estadística y respeta la consigna de "justificar por qué se usa la media".

![Cuando hay un caso extremo, el promedio engaña: la mediana es más honesta](figuras/fig_imputacion.png)

### 11.6 Elegimos tres "estilos de razonamiento" distintos para comparar

- **Antes:** los TPs previos no entrenaban modelos, solo describían los datos.
- **Problema:** sin probar varios "razonadores", no hay forma de saber cuál se adapta mejor al problema.
- **Imaginá esto:** para resolver un caso podés consultar a **un juez solo** (que suma puntos a favor y en contra), a **un experto que va haciendo preguntas en cascada** ("¿tiene abogado? ¿hace cuánto vino la orden? ¿de qué corte?"), o a **un jurado de doscientas personas que vota por mayoría**. Cada uno responde distinto.
- **Qué cambió:** probamos los tres (regresión logística, árbol de decisión y bosque aleatorio) y nos quedamos con el que mejor adivinó: el **jurado de doscientos**, que acertó en 6 de cada 10 casos. Descartamos modelos más complejos (redes neuronales, *gradient boosting*) porque la consigna pedía simplicidad y porque cuanto más complejo el modelo, más difícil es explicar por qué decidió lo que decidió.

### 11.7 Una única partición de datos, sin probar varias veces

- **Antes:** no había entrenamiento, así que no había evaluación.
- **Problema:** hay que probar al sistema con casos que **no vio durante el entrenamiento**, igual que un alumno no debería ver el examen antes de rendirlo.
- **Imaginá esto:** tenés un libro de **10 capítulos**. Estudiás 8 y guardás los otros 2 para autoevaluarte después. Eso es exactamente la división 80/20.
- **Qué cambió:** los 834 casos se dividieron en **667 para enseñar y 167 para tomar examen**. Algunos cursos recomiendan repetir esa división varias veces y promediar los resultados (más confiable). El profesor pidió expresamente **una sola partición**, así que respetamos la indicación y lo dejamos asentado como limitación.

### 11.8 Sumamos una base de datos en la nube

- **Antes:** los datos vivían en archivos sueltos **dentro de la computadora del autor**.
- **Problema:** si la computadora se apaga, el sitio web no responde. Para un servicio público, eso es inaceptable.
- **Imaginá esto:** la diferencia entre tener las fotos **solo en el celular** (si lo perdés, las perdiste) y tenerlas **en la nube** (las ves desde cualquier dispositivo). Lo mismo con los datos: necesitábamos un lugar **siempre disponible**.
- **Qué cambió:** sumamos **Supabase**, una base de datos PostgreSQL administrada. Elegimos esa porque PostgreSQL es un estándar conocido (mañana podemos mudarnos a otro proveedor sin reescribir nada) y porque se instala con un solo comando.

### 11.9 Una sola tabla grande en vez de muchas tablas conectadas

- **Antes:** el TP2 trabajaba con una planilla rectangular típica (filas y columnas planas).
- **Problema:** el diseño "ortodoxo" de bases de datos divide los datos en muchas tablas conectadas. Eso es robusto pero complicado.
- **Imaginá esto:** dos formas de archivar expedientes. **Opción A:** archivero con muchos cajones, uno para las carátulas, otro para los jueces, otro para las cortes, otro para las citas. Para reconstruir un expediente hay que ir cajón por cajón. **Opción B:** una sola carpeta gorda con todo el expediente adentro. El profesor pidió no complicar: una sola tabla.
- **Qué cambió:** cada caso se guarda como una **carpeta completa (paquete JSON)** dentro de una única tabla. Para que las búsquedas sigan siendo rápidas, agregamos un **índice especial** (el equivalente al índice alfabético al final de un libro de mil páginas: en lugar de leer todo, vas directo al término).

### 11.10 Consultamos primero "nuestra alacena" antes de "ir al supermercado"

- **Antes:** cada consulta golpeaba directamente a la API pública.
- **Problema:** la API solo permite **125 consultas gratis por día**. Con un poco de tráfico real, el cupo se acaba en minutos.
- **Imaginá esto:** querés cocinar fideos. **Opción A** (la mala): cada vez que cocinás, vas al supermercado a comprar el paquete. Si está cerrado, te quedaste sin cena. **Opción B** (la buena): primero revisás la alacena. Si tenés el paquete, lo usás. Si no, vas al supermercado, comprás dos, usás uno y **guardás el otro en la alacena para la próxima**.
- **Qué cambió:** ahora **primero miramos nuestra base de datos**. Si el caso está, lo devolvemos al instante. Si no, llamamos a la API, traemos el caso y lo guardamos para que la próxima persona que lo pida no consuma cupo.

![Patrón base de datos primero, API como respaldo](figuras/fig_cache.png)

### 11.11 Una página web pública en lugar de un cuaderno en una sola computadora

- **Antes:** el TP1 y TP2 se entregaban como **cuadernos de Jupyter**: archivos que solo corren si uno los abre en una computadora con todo el entorno instalado.
- **Problema:** un litigante real nunca va a abrir un cuaderno de programación. Necesita un sitio al que entre con el navegador y listo.
- **Imaginá esto:** la diferencia entre entregar una **receta que solo funciona en la cocina del autor** (con sus utensilios específicos) y abrir un **restaurante al que cualquiera puede ir a comer**.
- **Qué cambió:** publicamos un servicio en internet usando Flask (una herramienta liviana para páginas web) hospedado en Fly.io. La gran ventaja: **cuando nadie lo usa, el servidor se duerme y no cobra**; cuando alguien entra, se despierta en segundos. Costo en reposo: prácticamente cero.

### 11.12 Ajustamos el puntaje de búsqueda para reconocer apellidos poco comunes

- **Antes:** el buscador del sitio le daba la misma cantidad de puntos a cualquier coincidencia textual.
- **Problema:** un apellido distintivo **es muchísimo más informativo** que una palabra común, pero el sistema los trataba igual.
- **Imaginá esto:** entrás a un edificio y preguntás en portería por dos personas. Si decís "busco a **Pérez**", el portero te pregunta cuál de los doce Pérez del edificio. Si decís "busco a **Porrazzo**", el portero sabe enseguida a quién, porque ese apellido es único. Pasaba que al buscar "Porrazzo" el sistema le ponía la misma puntuación que a "Pérez", caía bajo el umbral y se descartaba el resultado relevante.
- **Qué cambió:** sumamos un **bono grande de puntos** cada vez que lo que la persona escribió aparece **tal cual** dentro del nombre del caso. Si el apellido es raro, el bono lo empuja por encima del umbral. El resto del sistema de puntuación quedó igual.

### 11.13 Sumamos 22 consultas típicas como "examen del mundo real"

- **Antes:** la evaluación se limitaba a estadísticas tabulares (cuántos faltantes, qué distribución, etc.).
- **Problema:** las estadísticas no detectan errores de uso real, como que el sitio se rompa cuando alguien teclea raro.
- **Imaginá esto:** probar un cajero automático no es solo verificar que la pantalla prende. Hay que probar también qué pasa si **alguien escribe el PIN equivocado, deja la tarjeta a medio insertar, o pide más plata de la que tiene**.
- **Qué cambió:** sumamos un examen extra de **22 consultas típicas** que hace una persona común:
  - **Nombres a medias:** "Smith v", "stateville"
  - **Códigos de estado a la mitad:** "ny", "nys", "ca"
  - **Números de expediente:** "15-CV-6684"
  - **Apellidos de jueces:** "Cott", "kogan"
  - **Casos extremos:** búsqueda vacía, solo espacios, mayúsculas, acentos, número gigantesco
  
  Cada una verifica que el sitio responda razonablemente y no se caiga.

### 11.14 Hicimos que el sitio reconozca palabras con o sin acento

- **Antes:** la comparación de texto era estricta caracter por caracter.
- **Problema:** la misma palabra acentuada **puede guardarse de dos maneras invisibles para el ojo humano pero distintas para la computadora**.
- **Imaginá esto:** escribís "café" en el teclado y en la pantalla ves "café". Internamente la computadora puede guardar eso como **la letra "é" entera** o como **"e" + un acento agregado aparte**. Para vos es lo mismo, para la computadora son dos cosas distintas y por eso a veces buscás algo y "no aparece" aunque esté ahí.
- **Qué cambió:** antes de comparar, ahora **unificamos todo a una sola forma estándar**. Así, "Pérez" y "Pérez" (escritos con las dos formas) se reconocen como iguales. Es invisible para el usuario y no exigió ningún cambio en la base de datos.

---

*Trabajo realizado por López Pérez en el marco del curso Data Science Real World Applications, 2026.*

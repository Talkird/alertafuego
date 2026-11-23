# AlertaFuego: backend

API y detección de AlertaFuego. Cada 10 minutos descarga de Google Earth Engine la imagen más reciente del satélite GOES-19 sobre Argentina, corre dos modelos de segmentación TinyUNet (b-shift y d-noleak), guarda las detecciones dentro del territorio en PostgreSQL con PostGIS y avisa por correo electrónico y Telegram a los usuarios con zonas de alerta que contienen un foco. La API, hecha con FastAPI, sirve las detecciones, los reportes, las estadísticas, las zonas de alerta y la cuenta de cada usuario. El código del modelo (preparación de datos, entrenamiento, inferencia y evaluación) está en `model/`.

## Cómo correrlo

Requisitos: Python 3.12, un proyecto de Google Cloud con Google Earth Engine habilitado y una base PostgreSQL con PostGIS (por ejemplo, Supabase).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # PyTorch con CUDA (para entrenar)
# Sin GPU NVIDIA (por ejemplo, en macOS): pip install -r requirements-lambda.txt (PyTorch para CPU)
earthengine authenticate                 # una sola vez
cp .env.example .env                     # completar DATABASE_URL, SUPABASE_URL, EARTH_ENGINE_PROJECT_ID y DETECT_API_KEY
alembic upgrade head                     # crea PostGIS y las tablas
python -m backend.scripts.load_regions   # límites de provincias y departamentos del IGN, una vez por base
python -m uvicorn backend.app.main:app --reload   # http://localhost:8000
```

Los comandos se corren desde esta carpeta, para que los módulos `backend.*` y `model.*` se resuelvan. La documentación interactiva de la API está en http://localhost:8000/docs.

### Modelos

Los pesos de los dos modelos del sistema están en `model/checkpoints/b-shift/` y `model/checkpoints/d-noleak/`. Cada carpeta tiene `model_best.pt`, `norm_stats.json` (normalización de las bandas), `model_config.json` (tamaño del modelo y umbrales), y las métricas y curvas de su entrenamiento. La inferencia carga el modelo principal de `model/checkpoints/` y los demás de `model/checkpoints/ensemble/<nombre>/`, así que para detectar en local:

```bash
cd model/checkpoints
cp b-shift/model_best.pt b-shift/norm_stats.json b-shift/model_config.json .
mkdir -p ensemble/d-noleak
cp d-noleak/model_best.pt d-noleak/norm_stats.json d-noleak/model_config.json ensemble/d-noleak/
cd ../..
```

### Correr una detección a mano

Los endpoints de detección piden la clave `DETECT_API_KEY` en el encabezado `X-API-Key`:

```bash
curl -X POST -H "X-API-Key: $DETECT_API_KEY" http://localhost:8000/detect/demo        # zona chica, unos segundos
curl -X POST -H "X-API-Key: $DETECT_API_KEY" http://localhost:8000/detect/argentina   # todo el país, alrededor de un minuto
```

En producción, la detección programada corre en una segunda función Lambda, invocada por EventBridge Scheduler cada 10 minutos.

### Alertas

- Correo: Amazon SES, con el remitente de `ALERT_EMAIL_FROM`. Sin esa variable, las alertas se registran pero no se envían.
- Telegram: con `TELEGRAM_BOT_TOKEN`, `TELEGRAM_BOT_USERNAME` y `TELEGRAM_WEBHOOK_SECRET`. En local, el bot recibe los mensajes por *polling*: `python -m backend.app.telegram poll`.

## Cómo correr las pruebas

```bash
pip install -r requirements-dev.txt
pytest --cov
```

`tests/unit/` cubre la lógica del modelo y de la API sin red (Earth Engine simulado). `tests/api/` prueba cada endpoint contra una base PostGIS de prueba, que las pruebas migran con Alembic y vacían después de cada caso. Se configura con `TEST_DATABASE_URL` (por defecto, `postgresql+psycopg://postgres:postgres@localhost:5432/alertafuego_test`; la base tiene que existir). Para correr solo las pruebas que no usan la base: `pytest -m "not db"`.

## Cómo construir el *dataset*

El *dataset* de entrenamiento no se incluye en el repositorio. Se construye desde Earth Engine: para cada detección de VIIRS busca la imagen GOES-19 más cercana en el tiempo y guarda un recorte de 16 bandas calibradas con su máscara de fuego. Se exporta un mes por carpeta:

```bash
python -m model.scripts.export_dataset \
    --start-date 2025-09-01 --end-date 2025-10-01 \
    --train-end-date 2025-09-24 --val-end-date 2025-09-27 \
    --output-dir model/dataset/2025-09
```

- `--end-date` es exclusivo. Los días hasta `--train-end-date` van a entrenamiento, hasta `--val-end-date` a validación y el resto a prueba.
- Cada muestra es un `.npz` con `patch` (16 × 33 × 33) y `mask`, en `train/`, `val/` y `test/`, más un `manifest.csv`.
- Si la exportación se corta, el mismo comando retoma desde el último día completo.
- `--limit N` exporta pocas muestras, para una prueba rápida.

Después de exportar los meses, se combinan en un único manifiesto (`model/dataset/manifest.csv`), sin copiar archivos:

```bash
python -m model.scripts.merge_datasets
```

## Cómo entrenar

El entrenamiento usa GPU con CUDA si está disponible (con precisión mixta), o MPS en Apple; si no, CPU.

```bash
python -m model.scripts.train_model --epochs 30 --checkpoint-dir model/checkpoints/d-noleak
```

Esa es la configuración de d-noleak: recorte de 24 píxeles en una posición al azar, muestreo balanceado por hora UTC y pérdida de Tversky (`--fn-weight 0.7`). b-shift se entrenó con recortes de 32 píxeles desplazados hasta 8 píxeles y pérdida Dice (`--no-random-crop --max-shift 8 --crop-size 32 --no-balance-hours --fn-weight 0.5`). La configuración de cada modelo queda en su `results.json`.

Cada época guarda el *checkpoint* con mejor F2 en validación. Al terminar, el script elige el umbral más alto cuyo *recall* por foco en validación alcanza `--recall-target` (0,95 por defecto), evalúa el conjunto de prueba una sola vez y escribe en la carpeta del *checkpoint* `model_best.pt`, `norm_stats.json`, `model_config.json`, `results.json`, `metrics.csv` y `training_curves.png`.

Evaluación:

```bash
# Por recorte, con varios umbrales y por estrato horario
python -m model.scripts.evaluate_model --checkpoint-dir model/checkpoints/d-noleak --by-stratum

# Sobre escenas completas de Argentina, como en operación
python -m model.scripts.evaluate_scenes fetch        # descarga unas 30 escenas, una vez
python -m model.scripts.evaluate_scenes calibrate --checkpoint-dir model/checkpoints/b-shift
python -m model.scripts.evaluate_scenes score --checkpoint-dir model/checkpoints/b-shift --split test
python -m model.scripts.evaluate_scenes score --checkpoint-dir model/checkpoints/b-shift \
    --ensemble model/checkpoints/d-noleak --split test
```

`calibrate` elige los umbrales de día y de noche sobre las escenas de validación y los guarda en `model_config.json`. `score` escribe los resultados por escena en la carpeta del *checkpoint*.

## Endpoints

"Sesión" indica que el endpoint exige el *token* de Supabase en `Authorization: Bearer`. "Opcional" indica que funciona sin sesión, pero el filtro por zona propia (`area_id`) la requiere.

**Detección** (clave `X-API-Key`)

- `POST /detect/demo`: corre los modelos sobre una zona chica fija del noroeste y guarda las detecciones.
- `POST /detect/argentina`: corre los modelos sobre todo el país y guarda las detecciones.
- `POST /internal/scheduled-detection`: detección programada; procesa la última imagen si todavía no se procesó. Solo existe en la función del detector (`DETECTION_WORKER=1`).

**Detecciones** (sesión opcional)

- `GET /detections`: detecciones guardadas, filtradas por rectángulo (`west`, `south`, `east`, `north`), rango de fechas (`since`, `until`), provincia o departamento (`region_code`) o zona propia (`area_id`), con `limit`.
- `GET /detections/{id}/image`: recorte de la imagen satelital de la detección.

**Reportes de usuarios**

- `GET /detections/{id}/reports`: reportes de una detección.
- `POST /detections/{id}/reports`: reportar la detección como incendio o falso positivo (sesión).
- `PATCH /detections/{id}/reports`: editar el reporte propio (sesión).
- `GET /detections/{id}/reports/mine`: el reporte propio sobre la detección (sesión).

**Zonas de alerta y alertas** (sesión)

- `GET /alert-areas`: zonas del usuario.
- `POST /alert-areas`: crear una zona: punto y radio, polígono, o provincia o departamento, con una probabilidad mínima.
- `PATCH /alert-areas/{id}`: editar una zona.
- `DELETE /alert-areas/{id}`: eliminar una zona.
- `GET /alerts`: historial de alertas del usuario, cada una con su detección.

**Estadísticas** (sesión opcional; aceptan `since`, `until`, `region_code` y `area_id`)

- `GET /stats/regions?kind=province|department`: detecciones por provincia o departamento.
- `GET /stats/hourly`: detecciones por hora del día, en hora argentina.
- `GET /stats/reports`: reportes por veredicto y categoría.
- `GET /stats/recurrence`: celdas de unos 2 km con detecciones, con cuántos días y años distintos se quemaron.

**Cuenta** (sesión, salvo la confirmación)

- `GET /account`: canales de aviso del usuario.
- `PATCH /account`: activar o desactivar los avisos por correo.
- `POST /account/alert-email`: pedir los avisos en otra dirección; envía un enlace de confirmación.
- `POST /account/alert-email/confirm`: confirmar la dirección con el *token* del enlace (sin sesión).
- `DELETE /account/alert-email`: volver a la dirección de la cuenta.
- `DELETE /account/alert-email/pending`: cancelar un cambio sin confirmar.
- `POST /account/telegram/link`: enlace de un solo uso para vincular el bot de Telegram.
- `POST /account/telegram/test`: enviar un mensaje de prueba por Telegram.
- `DELETE /account/telegram`: desvincular Telegram.

**Telegram**

- `POST /telegram/webhook`: recibe los mensajes del bot (vinculación con `/start` y desvinculación con `/stop`). Exige `TELEGRAM_WEBHOOK_SECRET` y no figura en `/docs`.

## Despliegue

La API y el detector corren en AWS Lambda con la misma imagen Docker; la infraestructura está declarada en Terraform. Los pasos están en [`terraform/README.md`](terraform/README.md).

# Diabetes AI App — Backend

Flask REST API for an **Adaptive AI System for Diabetes Prediction Using Human Feedback** (research prototype).

- **Dataset:** BRFSS 2021
- **Features:** 49 selected processed features
- **Model:** Logistic Regression with optimized threshold (0.38)

The API provides diabetes risk prediction, explainable AI (feature contributions), human feedback collection with a reward signal, and adaptive model management, monitoring, and versioning.

## Project Structure

```
backend/
├── app.py                  # Entry point: py app.py
├── requirements.txt
├── test_backend.py         # Manual API test script
├── test_data/
│   └── test_patients.json  # Predefined test patient cases
└── app/
    ├── __init__.py         # Application factory (create_app)
    ├── config.py           # APP_MODE validation, CORS origins, paths
    ├── constants.py        # Model metadata, metrics, feature mappings, weights
    ├── stores.py           # In-memory feedback/prediction stores
    ├── services/
    │   ├── prediction.py   # Feature mapping, risk scoring, explanations, pipeline
    │   └── test_data.py    # Test patient loading
    └── routes/
        ├── health.py       # GET  /api/health
        ├── prediction.py   # POST /api/predict, GET /api/test/cases, POST /api/test/predict
        ├── feedback.py     # POST /api/feedback
        ├── model.py        # GET  /api/model/status, /versions, /comparison
        └── adaptive.py     # GET  /api/adaptive/metrics, POST /api/adaptive/update
```

## Setup

```powershell
pip install -r requirements.txt
```

## Running

```powershell
py app.py
```

The server starts on `http://127.0.0.1:5000` (all interfaces, port 5000) with CORS enabled for the Angular dev server (`localhost:4200`).

## Application Modes

Set via the `APP_MODE` environment variable (default: `test`):

| Mode         | Behavior                                                        |
|--------------|-----------------------------------------------------------------|
| `test`       | Test endpoints enabled; adaptive model updates blocked (403)    |
| `production` | Test endpoints disabled (403); adaptive updates enabled         |

```powershell
$env:APP_MODE = "production"; py app.py
```

## API Endpoints

### Health

| Method | Endpoint       | Description                              |
|--------|----------------|------------------------------------------|
| GET    | `/api/health`  | Health check, mode, and update status    |

### Prediction

| Method | Endpoint            | Description                                              |
|--------|---------------------|----------------------------------------------------------|
| POST   | `/api/predict`      | Risk prediction from feature values (`{"features": {...}}`) |
| GET    | `/api/test/cases`   | List predefined test patient cases (test mode only)      |
| POST   | `/api/test/predict` | Predict using a test case ID, e.g. `{"test_case_id": "TEST-001"}` (test mode only) |

Prediction response:

```json
{
  "prediction": 1,
  "probability": 0.7415,
  "risk_category": "Higher Risk",
  "explanations": [ ... ],
  "mode": "test"
}
```

Risk categories: `Lower Risk` (< 0.4), `Moderate Risk` (0.4–0.7), `Higher Risk` (≥ 0.7).

Both `/api/predict` and `/api/test/predict` share the same prediction pipeline, so test and real predictions cannot diverge.

### Feedback

| Method | Endpoint        | Description                                             |
|--------|-----------------|---------------------------------------------------------|
| POST   | `/api/feedback` | Record user feedback; returns a reward signal (+1/−1)   |

Request body: `{"prediction_id": "...", "feedback": "agree"|"disagree", "helpfulness": "...", "comment": "..."}`

### Model

| Method | Endpoint                 | Description                          |
|--------|--------------------------|--------------------------------------|
| GET    | `/api/model/status`      | Current model metadata               |
| GET    | `/api/model/versions`    | All model versions (V0–V4)           |
| GET    | `/api/model/comparison`  | Comparison across candidate models   |

### Adaptive Learning

| Method | Endpoint                | Description                                        |
|--------|-------------------------|----------------------------------------------------|
| GET    | `/api/adaptive/metrics` | Performance and feedback metrics across versions   |
| POST   | `/api/adaptive/update`  | Trigger adaptive model update (production only)    |

## Testing

With the server running:

```powershell
py test_backend.py
```

This exercises the health check, test cases, all five test predictions, invalid-case handling, and the test-mode adaptive update lock.

## Notes

- Model inference and explanations are currently **simulated** using fixed weights (`app/constants.py`); in production these would come from the trained Logistic Regression model and preprocessing pipeline.
- Feedback and predictions are stored **in memory** and reset on restart; replace with persistent storage for production.
- The Flask development server is used for research purposes only — use a production WSGI server (e.g. waitress, gunicorn) for deployment.

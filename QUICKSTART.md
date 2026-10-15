# Quick Start Guide

Get the ML platform running in 10 minutes!

> I removed Steps 3, 4 (API server) and 6 (data drift monitoring). May be good to include step 6 in later versions.

- [Prerequisites](#prerequisites)
- [Step 1: Setup (2 minutes)](#step-1-setup-2-minutes)
- [Step 2: Generate Data and Train Model (2 minutes)](#step-2-generate-data-and-train-model-2-minutes)
- [Step 5: Run MLflow Experiments (Optional, 3 minutes)](#step-5-run-mlflow-experiments-optional-3-minutes)
- [Common Commands](#common-commands)
- [What's Next?](#whats-next)
- [Troubleshooting](#troubleshooting)
- [Success Criteria](#success-criteria)

## Prerequisites

- Install [`uv` (Python package and project manager)](https://docs.astral.sh/uv/getting-started/installation/#installation-methods).
- Run `uv --version` to verify the installation.
- Run `python -V` to check the version of Python installed on your machine.

The rest of this guide assumes it is Python 3.12. If not, you should modify some commands below.

## Step 1: Setup (2 minutes)

Open terminal in this repo's root directory and run:

```bash
# Create virtual environment
uv venv --python 3.12

# Install dependencies
uv pip install -r requirements.txt --system-certs
```

If it fails, lower the python version:

```bash
uv venv --python 3.10 --allow-existing
uv pip install -r requirements.txt --system-certs
```

## Step 2: Generate Data and Train Model (2 minutes)

```bash
# Create directories for generated artifacts (scripts will also create these)
mkdir -p data models

# Generate synthetic fraud detection data
uv run python src/generate_data.py

# Train baseline model
uv run python src/train_naive.py
```

**Expected output:**

- `data/train.csv` (8,000 transactions)
- `data/test.csv` (2,000 transactions)
- `models/model.pkl` (trained model)
- Accuracy: ~98%, F1-score: ~0.5-0.7

<!-- ## Step 3: Start the API (1 minute)

```bash
# Start the validated API with data validation
uv run uvicorn src.serve_validated:app --reload --host 0.0.0.0 --port 8000
```

**Verify:** Open <http://localhost:8000/docs> in your browser

## Step 4: Test the API (1 minute)

```bash
# In a new terminal - Test valid transaction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"amount": 150.0, "hour": 14, "day_of_week": 3, "merchant_category": "online"}'

# Expected response:
# {"is_fraud": false, "fraud_probability": 0.15, "validation_passed": true}

# Test invalid transaction (should return 400 error)
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"amount": -100, "hour": 25, "day_of_week": 10, "merchant_category": "fake"}'

# Expected: HTTP 400 with detailed error messages
``` -->

## Step 5: Run MLflow Experiments (Optional, 3 minutes)

In a first terminal, run:

```bash
# Start MLflow server
uv run mlflow server --host 0.0.0.0 --port 5001 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

In a second terminal, run:

```bash
# Run experiment sweep
uv run python src/train_mlflow.py
```

Open <http://localhost:5001> to view experiments

<!-- ## Step 6: Run Drift Monitoring (Optional, 1 minute)

```bash
uv run python src/monitoring.py

# Open drift_report.html in your browser
``` -->

## Common Commands

```bash
# Run all tests
uv run pytest tests/ -v

# Build Docker image
docker build -t fraud-detection-api .

# Run Docker container
docker run -p 8000:8000 fraud-detection-api

# Check API health
curl http://localhost:8000/health

# View API documentation
open http://localhost:8000/docs
```

## What's Next?

1. **Explore the code:** Check `src/` directory for implementations
2. **Read the full README:** Detailed chapter-by-chapter guide
3. **Experiment:** Modify hyperparameters in training scripts
4. **Deploy:** Use Docker for containerized deployment
5. **Scale:** Follow production scaling guide in README

## Troubleshooting

**API won't start:**

```bash
# Check if port 8000 is already in use
lsof -i :8000
# Kill the process if needed
kill -9 <PID>
```

**Model not found:**

```bash
# Retrain the model
uv run python src/train_naive.py
```

**Import errors:**

```bash
# Make sure you're in the project root
pwd  # Should end with freecodecamp-local-ml-platform
# Reinstall dependencies
uv pip install -r requirements.txt
```

**MLflow connection error:**

```bash
# Make sure MLflow server is running on port 5001
curl http://localhost:5001/health
```

## Success Criteria

✅ API responds to health check  
✅ Valid predictions return 200 status  
✅ Invalid inputs return 400 with error details  
✅ Model accuracy > 90%  
✅ F1-score > 0.3  
✅ All tests pass  

---

**Time to complete:** ~10 minutes  
**Difficulty:** Beginner-friendly  
**Support:** Check README.md for detailed documentation

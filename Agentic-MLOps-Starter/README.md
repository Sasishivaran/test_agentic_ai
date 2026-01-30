# Agentic-MLOps-Starter

This repository contains a complete MLOps starter workflow including:
- Drift detection (PSI + KS)
- Feedback ingestion
- Retraining pipeline
- MLflow experiment tracking
- CI/CD workflow template
- Synthetic data generation

## Folder Structure
- data/
  - baseline_data.csv
  - inference_data.csv
  - feedback/
- monitor.py
- retrain.py
- data_generator.py
- feedback_generator.py
- .github/workflows/retrain.yml


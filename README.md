# Earthquake Disaster Prediction Agent

This repository has shifted from flood-risk experimentation to an **earthquake-focused disaster prediction and decision-support workflow**.

## Current Project Focus

The current notebooks build an end-to-end pipeline around historical earthquake records:

- Explore earthquake event distributions and data quality
- Clean and preprocess missing geographic/alert metadata
- Engineer risk-oriented features (for example `energy_proxy`, time features)
- Create a target class: `Disaster_Urgency`
- Prepare train/test splits and scaled features for downstream modeling

## Notebook Workflow

- `/tmp/workspace/adan-ayaz-stan/flood-prediction-agent/notebooks/01_data_exploration.ipynb`
  - Loads `earthquake_data.csv`
  - Performs overview, missing-value checks, time analysis, spatial analysis, and correlation analysis

- `/tmp/workspace/adan-ayaz-stan/flood-prediction-agent/notebooks/02_preprocessing.ipynb`
  - Cleans nulls in `alert`, `continent`, `country`, and `location`
  - Builds engineered features from `magnitude` and `date_time`
  - Derives `Disaster_Urgency` labels from hazard indicators (`magnitude`, `mmi`, `tsunami`)
  - Encodes categorical features and prepares scaled ML-ready datasets

## Repository Structure

- `/tmp/workspace/adan-ayaz-stan/flood-prediction-agent/notebooks` – data exploration and preprocessing notebooks
- `/tmp/workspace/adan-ayaz-stan/flood-prediction-agent/src` – project entrypoint code
- `/tmp/workspace/adan-ayaz-stan/flood-prediction-agent/requirements.txt` – Python dependencies

## Getting Started

1. Create and activate a Python virtual environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Open notebooks:
   - `jupyter notebook`
4. Run the application entrypoint (placeholder app):
   - `python /tmp/workspace/adan-ayaz-stan/flood-prediction-agent/src/main.py`

## Notes

- Existing notebooks and preprocessing logic are currently centered on **earthquake** analytics.
- README terminology has been updated to match this shifted scope.

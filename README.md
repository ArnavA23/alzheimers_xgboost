# Two-Stage Hierarchical XGBoost Ensemble for Three-Class Alzheimer's Disease Classification

Code accompanying the preprint *"Predicting diagnostic status from structural MRI using a
hierarchical XGBoost ensemble without cognitive tests"* by Arnav Acharya and Serena Qinyun Z. Shi
(eiRxiv, 2026).

The pipeline classifies patients as cognitively normal (CN), mild cognitive impairment (MCI),
or Alzheimer's disease (AD) from structural MRI and routinely available clinical variables,
using **no cognitive test scores as inputs**. A parallel regressor predicts CDR Sum of Boxes.

## Data availability

**No data is included in this repository.** All data derive from the National Alzheimer's
Coordinating Center (NACC) under a Data Use Agreement (Request ID #17018). Per that agreement,
NACC data may not be redistributed to any third party. To reproduce this work you must obtain
the data directly from NACC (https://naccdata.org) and supply it locally (see Setup). This
repository contains **code only**.

## Repository contents

```
src/alzheimers_xgboost_v6.py   Full pipeline: feature engineering, two-stage ensemble,
                               CDRSUM regressor, calibration (Method 6), evaluation
docs/feature_dictionary.md     Definitions for all 340 candidate features
requirements.txt               Python dependencies
```

## Setup

```bash
pip install -r requirements.txt
export ALZ_DATA_DIR=/path/to/your/nacc/data   # Windows: set ALZ_DATA_DIR=...
python src/alzheimers_xgboost_v6.py
```

`ALZ_DATA_DIR` must contain `Train_Data.xlsx`, `Validation_Data.xlsx`, `Test_Data.xlsx`,
and an `Imaging Data/` subfolder of extracted axial-slice PNGs. These files are not provided.

## Feature dictionary

All 340 candidate features (287 tabular + 53 image; 203 retained after gain-based selection)
are defined in [`docs/feature_dictionary.md`](docs/feature_dictionary.md), including the
FreeSurfer naming conventions, the hand-crafted image features, and the engineered
rate-of-change and interaction features.

## Model and calibration (Method 6)

The classifier is a two-stage hierarchy: Stage 1 separates CN from not-normal (NN); Stage 2
separates MCI from AD within the NN group. Each stage averages an ensemble of 10 XGBoost
models. Method 6 ("Adjusted") applies class-specific decision-threshold optimization under
nested cross-validation; threshold selection is defined in `src/alzheimers_xgboost_v6.py`.

## AI-use disclosure

AI assistants were used for code review, editing, and figure/document preparation. All
analyses were designed and run by the author, who reviewed and takes responsibility for all
content.

## License

Released under the MIT License (see `LICENSE`).

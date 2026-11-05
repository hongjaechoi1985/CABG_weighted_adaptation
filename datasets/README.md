# Restricted data (not distributed)

The patient-level SNUH and BRM datasets used in the publication are **not** included. Do not commit data files to this folder.

The script expects two separate input files by default:

- `BRM_subset/Total_TL_CABG_data_cleaned_BRM_holdout70.csv`: SNUH patients (`hospital == 1`) and fixed BRM evaluation patients (`hospital == 2`).
- `BRM_subset/BRM_CABG_adaptation_30pct.csv`: fixed BRM adaptation subset.

The two BRM partitions must contain disjoint patients. Since no identifiers are required for analysis, verify non-overlap locally before removing identifiers from the analytical data. The script only performs an optional ID-based overlap check when the ID columns exist in both frames.

All predictor names are listed in `feature_cols`, and the outcome column is `inhosp_death` (0/1). Details on coding and clinical units should be checked against the institutional data dictionary.

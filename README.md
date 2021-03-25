# german_credit_python
 
A Logistic Regression Python model to predict loan default.
Model was trained on the German Credit Data dataset.
logreg_classifier.pickle is the trained model artifact.
Sample inputs to the scoring function are included (`df_baseline.json`, `df_sample.json`).

Model code includes a metrics function used to compute Group and Bias metrics.
The metrics function expects a DataFrame with at lease the following three columns three columns: `score` (predicted), `label_value` (actual), and `gender` (protected attribute).
Sample inputs to the metrics function are included (`df_baseline_scored.json`, `df_sample_scored.json`).

The output of the metrics job when the input data is `df_sample_scored.json` should be 

{
    "attribute_name": "gender",
    "attribute_value": "female",
    "ppr_disparity": 0.5,
    "pprev_disparity": 0.888888888888889,
    "precision_disparity": 1.3599999999999999,
    "fdr_disparity": 0.7567567567567567,
    "for_disparity": 1.6097560975609757,
    "fpr_disparity": 0.7648073605520413,
    "fnr_disparity": 1.32,
    "tpr_disparity": 0.8976000000000001,
    "tnr_disparity": 1.1500366837857667,
    "npv_disparity": 0.9158957106812448
}

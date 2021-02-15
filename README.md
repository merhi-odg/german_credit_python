# german_credit_python
 
A Logistic Regression Python model to predict loan default.
Model was trained on the German Credit Data dataset.
logreg_classifier.pickle is the trained model artifact.
Sample inputs to the scoring function are included (`df_baseline.json`, `df_sample.json`)

Model code includes a metrics function used to compute Group and Bias metrics.
The metrics function expects a DataFrame with at lease the following three columns three columns: `score` (predicted), `label_value` (actual), and `gender` (protected attribute).
Sample inputs to the metrics function are included (`df_baseline_scored.json`, `df_sample_scored.json`)

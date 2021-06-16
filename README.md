# german_credit_python
 
A Logistic Regression Python model to predict loan default.
Model was trained on the German Credit Data dataset.
logreg_classifier.pickle is the trained model artifact.
Sample inputs to the scoring function are included (`df_baseline.json`, `df_sample.json`).

Model code includes a metrics function used to compute bias, drift, performance, and explainability metrics.
The metrics function expects two DataFrames with at lease the following four columns: `score` (predicted), `label_value` (actual), `predicted_probs`, and `gender` (protected attribute).
Sample inputs to the metrics function are included (`df_baseline_scored.json`, `df_sample_scored.json`).

The output of the scoring job when the input data is `df_baseline.json` is a JSONS file (one-line JSON records). Here are the first three output records
```json
[{"id": 687,"duration_months": 36,"credit_amount": 2862,"installment_rate": 4,"present_residence_since": 3,"age_years": 30,"number_existing_credits": 1,"checking_status": "A12","credit_history": "A33","purpose": "A40","savings_account": "A62","present_employment_since": "A75","debtors_guarantors": "A101","property": "A124","installment_plans": "A143","housing": "A153","job": "A173","number_people_liable": 1,"telephone": "A191","foreign_worker": "A201","gender": "male","label_value": 0,"score": 1,"predicted_probs": 0.78560664508261}]
[{"id": 500,"duration_months": 24,"credit_amount": 3123,"installment_rate": 4,"present_residence_since": 1,"age_years": 27,"number_existing_credits": 1,"checking_status": "A11","credit_history": "A32","purpose": "A40","savings_account": "A61","present_employment_since": "A72","debtors_guarantors": "A101","property": "A122","installment_plans": "A143","housing": "A152","job": "A173","number_people_liable": 1,"telephone": "A191","foreign_worker": "A201","gender": "female","label_value": 1,"score": 1,"predicted_probs": 0.8489434446913293}]
[{"id": 332,"duration_months": 60,"credit_amount": 7408,"installment_rate": 4,"present_residence_since": 2,"age_years": 24,"number_existing_credits": 1,"checking_status": "A12","credit_history": "A32","purpose": "A40","savings_account": "A62","present_employment_since": "A72","debtors_guarantors": "A101","property": "A122","installment_plans": "A143","housing": "A152","job": "A174","number_people_liable": 1,"telephone": "A191","foreign_worker": "A201","gender": "female","label_value": 1,"score": 1,"predicted_probs": 0.9402698129381786}]
```

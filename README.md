# german_credit_python
 
A Logistic Regression Python model to predict loan default.

To run the attached Jupyter notebook, create a Python 3.6.9  virtual environments with the libraries versioned in `requirements.txt`.

Model was trained on the German Credit Data dataset.
`logreg_classifier.pickle` is the trained model artifact.
Sample inputs to the scoring function are included (`df_baseline.json`, `df_sample.json`).

Model code includes a metrics function used to compute Group and Bias metrics.
The metrics function expects a DataFrame with at lease the following three columns three columns: `score` (predicted), `label_value` (actual), and `gender` (protected attribute).
Sample inputs to the metrics function are included (`df_baseline_scored.json`, `df_sample_scored.json`).

The output of the scoring job when the input data is `df_sample.json` is a JSONS file (one-line JSON records). Here are the first three output records:
```json
[{"ID": 993, "DURATION_MONTHS": 36, "CREDIT_AMOUNT": 3959, "INSTALLMENT_RATE": 4, "PRESENT_RESIDENCE_SINCE": 3, "AGE_YEARS": 30, "NUMBER_EXISTING_CREDITS": 1, "CHECKING_STATUS": "A11", "CREDIT_HISTORY": "A32", "PURPOSE": "A42", "SAVINGS_ACCOUNT": "A61", "PRESENT_EMPLOYMENT_SINCE": "A71", "DEBTORS_GUARANTORS": "A101", "PROPERTY": "A122", "INSTALLMENT_PLANS": "A143", "HOUSING": "A152", "JOB": "A174", "NUMBER_PEOPLE_LIABLE": 1, "TELEPHONE": "A192", "FOREIGN_WORKER": "A201", "GENDER": "MALE", "LABEL": 0, "PREDICTED_SCORE": 1}]
[{"ID": 859, "DURATION_MONTHS": 9, "CREDIT_AMOUNT": 3577, "INSTALLMENT_RATE": 1, "PRESENT_RESIDENCE_SINCE": 2, "AGE_YEARS": 26, "NUMBER_EXISTING_CREDITS": 1, "CHECKING_STATUS": "A14", "CREDIT_HISTORY": "A32", "PURPOSE": "A40", "SAVINGS_ACCOUNT": "A62", "PRESENT_EMPLOYMENT_SINCE": "A73", "DEBTORS_GUARANTORS": "A103", "PROPERTY": "A121", "INSTALLMENT_PLANS": "A143", "HOUSING": "A151", "JOB": "A173", "NUMBER_PEOPLE_LIABLE": 2, "TELEPHONE": "A191", "FOREIGN_WORKER": "A202", "GENDER": "MALE", "LABEL": 0, "PREDICTED_SCORE": 0}]
[{"ID": 298, "DURATION_MONTHS": 18, "CREDIT_AMOUNT": 2515, "INSTALLMENT_RATE": 3, "PRESENT_RESIDENCE_SINCE": 4, "AGE_YEARS": 43, "NUMBER_EXISTING_CREDITS": 1, "CHECKING_STATUS": "A14", "CREDIT_HISTORY": "A32", "PURPOSE": "A42", "SAVINGS_ACCOUNT": "A61", "PRESENT_EMPLOYMENT_SINCE": "A73", "DEBTORS_GUARANTORS": "A101", "PROPERTY": "A121", "INSTALLMENT_PLANS": "A143", "HOUSING": "A152", "JOB": "A173", "NUMBER_PEOPLE_LIABLE": 1, "TELEPHONE": "A192", "FOREIGN_WORKER": "A201", "GENDER": "MALE", "LABEL": 0, "PREDICTED_SCORE": 0}]
```

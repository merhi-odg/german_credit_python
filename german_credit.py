# modelop.schema.0: input_schema.avsc
# modelop.schema.1: output_schema.avsc

import pandas as pd
import pickle
import numpy as np

# Bias libraries
from aequitas.preprocessing import preprocess_input_df
from aequitas.group import Group
from aequitas.bias import Bias 


# modelop.init
def begin():
    
    global logreg_classifier
    
    # load pickled logistic regression model
    logreg_classifier = pickle.load(open("logreg_classifier.pickle", "rb"))

    
# modelop.score
def action(data):
    
    # Turn data into DataFrame
    data = pd.DataFrame([data])
    
    # There are only two unique values in data.number_people_liable.
    # Treat it as a categorical feature
    data.number_people_liable = data.number_people_liable.astype('object')

    predictive_features = [
        'duration_months', 'credit_amount', 'installment_rate',
        'present_residence_since', 'age_years', 'number_existing_credits',
        'checking_status', 'credit_history', 'purpose', 'savings_account',
        'present_employment_since', 'debtors_guarantors', 'property',
        'installment_plans', 'housing', 'job', 'number_people_liable',
        'telephone', 'foreign_worker'
    ]
    
    data["predicted_score"] = logreg_classifier.predict(data[predictive_features])
    
    # MOC expects the action function to be a *yield* function
    yield data.to_dict(orient="records")


# modelop.metrics
def metrics(data):
    
    data = pd.DataFrame(data)

    # To measure Bias towards gender, filter DataFrame
    # to "score", "label_value" (ground truth), and
    # "gender" (protected attribute)
    data_scored = data[["score", "label_value", "gender"]]

    # Process DataFrame
    data_scored_processed, _ = preprocess_input_df(data_scored)

    # Group Metrics
    g = Group()
    xtab, _ = g.get_crosstabs(data_scored_processed)

    # Absolute metrics, such as 'tpr', 'tnr','precision', etc.
    absolute_metrics = g.list_absolute_metrics(xtab)

    # DataFrame of calculated absolute metrics for each sample population group
    absolute_metrics_df = xtab[
        ['attribute_name', 'attribute_value'] + absolute_metrics].round(2)

    # For example:
    """
        attribute_name  attribute_value     tpr     tnr  ... precision
    0   gender          female              0.60    0.88 ... 0.75
    1   gender          male                0.49    0.90 ... 0.64
    """

    # Bias Metrics
    b = Bias()

    # Disparities calculated in relation gender for "male" and "female"
    bias_df = b.get_disparity_predefined_groups(
        xtab,
        original_df=data_scored_processed,
        ref_groups_dict={'gender': 'male'},
        alpha=0.05, mask_significance=True
    )

    # Disparity metrics added to bias DataFrame
    calculated_disparities = b.list_disparities(bias_df)

    disparity_metrics_df = bias_df[
        ['attribute_name', 'attribute_value'] + calculated_disparities]

    # For example:
    """
        attribute_name	attribute_value    ppr_disparity   precision_disparity
    0   gender          female             0.714286        1.41791
    1   gender          male               1.000000        1.000000
    """

    output_metrics_df = disparity_metrics_df # or absolute_metrics_df

    # Output a JSON object of calculated metrics
    yield output_metrics_df.to_dict(orient="records")

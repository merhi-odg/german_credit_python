# modelop.schema.0: input_schema.avsc
# modelop.schema.1: output_schema.avsc

import pandas as pd
import numpy as np
import pickle
import shap
from aequitas.preprocessing import preprocess_input_df
from aequitas.group import Group
from aequitas.bias import Bias
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, confusion_matrix
from scipy.spatial.distance import jensenshannon
from scipy.stats import epps_singleton_2samp, gaussian_kde, ks_2samp


# modelop.init
def begin():

    global logreg_classifier
    global predictive_features
    global explainer

    # load pickled logistic regression model
    logreg_classifier = pickle.load(open("logreg_classifier.pickle", "rb"))

    # load pickled predictive feature list
    predictive_features = pickle.load(open("predictive_features.pickle", "rb"))

    # load shap explainer
    explainer = pickle.load(open("explainer.pickle", "rb"))


def preprocess(data):
    # There are only two unique values in data.number_people_liable.
    # Treat it as a categorical feature
    data["number_people_liable"] = data["number_people_liable"].astype("object")

    # one-hot encode data with pd.get_dummies()
    data = pd.get_dummies(data)

    # in case features don't exist that are needed for the model (possible when dummying)
    # will create column of zeros for that feature
    for col in predictive_features:
        if col not in data.columns:
            data[col] = np.zeros(data.shape[0])

    return data


# modelop.score
def action(data):

    # Turn data into DataFrame
    data = pd.DataFrame(data)

    # preprocess data
    data = preprocess(data)

    # generate predictions
    data["predicted_probs"] = [
        x[1] for x in logreg_classifier.predict_proba(data[predictive_features])
    ]
    data["score"] = logreg_classifier.predict(data[predictive_features])

    # MOC expects the action function to be a *yield* function
    yield data.to_dict(orient="records")


# modelop.metrics
def metrics(df_baseline, data):
    # dictionary to hold final metrics
    metrics = {}

    # getting dummies for shap values
    data_processed = preprocess(data)[predictive_features]

    # calculate metrics
    f1 = f1_score(data["label_value"], data["score"])
    cm = confusion_matrix(data["label_value"], data["score"])
    labels = ["Default", "Pay Off"]
    cm = matrix_to_dicts(cm, labels)
    fpr, tpr, thres = roc_curve(data["label_value"], data["predicted_probs"])
    auc_val = roc_auc_score(data["label_value"], data["predicted_probs"])
    roc = [{"fpr": x[0], "tpr": x[1]} for x in list(zip(fpr, tpr))]

    # assigning metrics to output dictionary
    metrics["performance"] = [
        {
            "test_name": "Classification Metrics",
            "test_category": "performance",
            "test_type": "classification_metrics",
            "test_id": "performance_classification_metrics",
            "values": {"f1_score": f1, "auc": auc_val, "confusion_matrix": cm},
        }
    ]

    # top-level metrics
    metrics["confusion_matrix"] = cm
    metrics["roc"] = roc

    # categorical/numerical columns for drift
    categorical_features = [
        f
        for f in list(data.select_dtypes(include=["category", "object"]))
        if f in df_baseline.columns
    ]
    numerical_features = [
        f for f in df_baseline.columns if f not in categorical_features
    ]
    numerical_features = [
        x
        for x in numerical_features
        if x not in ["id", "score", "label_value", "predicted_probs"]
    ]

    # assigning metrics to output dictionary
    metrics["bias"] = get_bias_metrics(data)
    metrics["data_drift"] = get_data_drift_metrics(
        df_baseline, data, numerical_features, categorical_features
    )
    metrics["concept_drift"] = get_concept_drift_metrics(df_baseline, data)
    metrics["explainability"] = [get_shap_values(data_processed)]

    # MOC expects the action function to be a *yield* function
    yield metrics


def get_bias_metrics(data):
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
        ["attribute_name", "attribute_value"] + absolute_metrics
    ].round(2)

    # Bias Metrics
    b = Bias()

    # Disparities calculated in relation gender for "male" and "female"
    bias_df = b.get_disparity_predefined_groups(
        xtab,
        original_df=data_scored_processed,
        ref_groups_dict={"gender": "male"},
        alpha=0.05,
        mask_significance=True,
    )

    # Disparity metrics added to bias DataFrame
    calculated_disparities = b.list_disparities(bias_df)

    disparity_metrics_df = bias_df[
        ["attribute_name", "attribute_value"] + calculated_disparities
    ]

    # Output a JSON object of calculated metrics
    return {
        "test_name": "Aequitas Bias",
        "test_category": "bias",
        "test_type": "bias",
        "protected_class": "race",
        "test_id": "bias_bias_gender",
        "reference_group": "male",
        "thresholds": {"min": 0.8, "max": 1.25},
        "values": [disparity_metrics_df.to_dict(orient="records")],
    }


def matrix_to_dicts(matrix, labels):
    cm = []
    for idx, label in enumerate(labels):
        cm.append(dict(zip(labels, matrix[idx, :].tolist())))
    return cm


def get_data_drift_metrics(df_baseline, df_sample, numerical_cols, categorical_cols):
    data_drift_metrics = []
    data_drift_metrics.append(es_metric(df_baseline, df_sample, numerical_cols))
    data_drift_metrics.append(ks_metric(df_baseline, df_sample, numerical_cols))
    data_drift_metrics.append(
        js_metric(df_baseline, df_sample, numerical_cols, categorical_cols)
    )
    return data_drift_metrics


def get_concept_drift_metrics(df_baseline, df_sample):
    concept_drift_metrics = []
    concept_drift_metrics.append(es_metric(df_baseline, df_sample, ["score"]))
    concept_drift_metrics.append(ks_metric(df_baseline, df_sample, ["score"]))
    concept_drift_metrics.append(js_metric(df_baseline, df_sample, ["score"], []))
    return concept_drift_metrics


def ks_metric(df1, df2, numerical_columns):
    ks_tests = [
        ks_2samp(data1=df1.loc[:, col], data2=df2.loc[:, col])
        for col in numerical_columns
    ]
    p_values = [x[1] for x in ks_tests]
    ks_pvalues = dict(zip(numerical_columns, p_values))
    return {
        "test_name": "Kolmogorov-Smirnov",
        "test_category": "data_drift",
        "test_type": "kolmogorov_smirnov",
        "metric": "p_value",
        "test_id": "data_drift_kolmogorov_smirnov_p_value",
        "values": ks_pvalues,
    }


def es_metric(df1, df2, numerical_columns):
    es_tests = []
    for col in numerical_columns:
        try:
            es_test = epps_singleton_2samp(x=df1.loc[:, col], y=df2.loc[:, col])
        except np.linalg.LinAlgError:
            es_test = [None, None]
        es_tests.append(es_test)
    p_values = [x[1] for x in es_tests]
    es_pvalues = dict(zip(numerical_columns, p_values))
    return {
        "test_name": "Epps-Singleton",
        "test_category": "data_drift",
        "test_type": "epps_singleton",
        "metric": "p_value",
        "test_id": "data_drift_epps_singleton_p_value",
        "values": es_pvalues,
    }


def js_metric(df1, df2, numerical_columns, categorical_columns):
    res = {}
    STEPS = 100
    for col in categorical_columns:
        col_baseline = df1[col].to_frame()
        col_sample = df2[col].to_frame()
        col_baseline["source"] = "baseline"
        col_sample["source"] = "sample"

        col_ = pd.concat([col_baseline, col_sample], ignore_index=True)

        arr = (
            col_.groupby([col, "source"])
            .size()
            .to_frame()
            .reset_index()
            .pivot(index=col, columns="source")
            .droplevel(0, axis=1)
        )
        arr_ = arr.div(arr.sum(axis=0), axis=1)
        arr_.fillna(0, inplace=True)
        js_distance = jensenshannon(
            arr_["baseline"].to_numpy(), arr_["sample"].to_numpy()
        )
        res.update({col: js_distance})

    for col in numerical_columns:
        # fit guassian_kde
        col_baseline = df1[col]
        col_sample = df2[col]
        kde_baseline = gaussian_kde(col_baseline)
        kde_sample = gaussian_kde(col_sample)

        # get range of values
        min_ = min(col_baseline.min(), col_sample.min())
        max_ = max(col_baseline.max(), col_sample.max())
        range_ = np.linspace(start=min_, stop=max_, num=STEPS)

        # sample range from KDE
        arr_baseline_ = kde_baseline(range_)
        arr_sample_ = kde_sample(range_)

        arr_baseline = arr_baseline_ / np.sum(arr_baseline_)
        arr_sample = arr_sample_ / np.sum(arr_sample_)

        # calculate js distance
        js_distance = jensenshannon(arr_baseline, arr_sample)

        res.update({col: js_distance})

    list_output = sorted(res.items(), key=lambda x: x[1], reverse=True)
    dict_output = dict(list_output)
    return {
        "test_name": "Jensen-Shannon",
        "test_category": "data_drift",
        "test_type": "jensen_shannon",
        "metric": "distance",
        "test_id": "data_drift_jensen_shannon_distance",
        "values": dict_output,
    }


def get_shap_values(data):
    # getting shap values for the test data
    shap_values = explainer.shap_values(data)

    # re-organizing and sorting the data
    shap_values = np.mean(abs(shap_values), axis=0).tolist()
    shap_values = dict(zip(data.columns, shap_values))
    sorted_shap_values = {
        k: v for k, v in sorted(shap_values.items(), key=lambda x: x[1])
    }

    # show the values
    return {
        "test_name": "SHAP",
        "test_category": "interpretability",
        "test_type": "shap",
        "metric": "feature_importance",
        "test_id": "interpretability_shap_feature_importance",
        "values": sorted_shap_values,
    }

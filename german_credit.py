import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import copy
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
    global df_baseline
    global explainer

    # load pickled logistic regression model
    logreg_classifier = pickle.load(open("logreg_classifier.pickle", "rb"))

    # load pickled predictive feature list
    predictive_features = pickle.load(open("predictive_features.pickle", "rb"))

    # load df_baseline_scored for metrics (specifically drift calculations)
    df_baseline = pd.read_json("df_baseline_scored.json", orient="records", lines=True)

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
    data["predicted_score"] = logreg_classifier.predict(data[predictive_features])
    data["predicted_probs"] = [
        x[1] for x in logreg_classifier.predict_proba(data[predictive_features])
    ]

    # MOC expects the action function to be a *yield* function
    # return data.to_dict(orient="records")
    yield data.to_dict(orient="records")


# modelop.metrics
def metrics(data):
    # dictionary to hold final metrics
    metrics = {}

    # convert data into DataFrame
    data = pd.DataFrame(data)

    # getting dummies for shap values
    data_processed = preprocess(data)[predictive_features]

    # calculate metrics
    f1 = f1_score(data["label_value"], data["score"])
    cm = confusion_matrix(data["label_value"], data["score"])
    labels = ["Default", "Pay Off"]
    cm = matrix_to_dicts(cm, labels)
    fpr, tpr, thres = roc_curve(data["label_value"], data["predicted_probs"])
    auc_val = roc_auc_score(data["label_value"], data["predicted_probs"])
    rc = [{"fpr": x[0], "tpr": x[1]} for x in list(zip(fpr, tpr))]

    # categorical/numerical columns for drift
    categorical_features = [
        f
        #for f in list(metrics_sample.select_dtypes(include=["category", "object"]))
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
    metrics["f1_score"] = f1
    metrics["confusion_matrix"] = cm
    metrics["auc"] = auc_val
    metrics["ROC"] = rc
    metrics["bias"] = get_bias_metrics(data)
    metrics["drift_metrics"] = get_drift_metrics(
        df_baseline, data, numerical_features, categorical_features
    )
    metrics["shap"] = get_shap_values(data_processed)

    # MOC expects the action function to be a *yield* function
    yield metrics
    # return metrics


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

    output_metrics_df = disparity_metrics_df  # or absolute_metrics_df

    # Output a JSON object of calculated metrics
    return output_metrics_df.to_dict(orient="records")


def matrix_to_dicts(matrix, labels):
    cm = []
    for idx, label in enumerate(labels):
        cm.append(dict(zip(labels, matrix[idx, :].tolist())))
    return cm


def get_drift_metrics(df_baseline, df_sample, numerical_cols, categorical_cols):
    drift_metrics = {}
    drift_metrics["drift__es"] = es_metric(df_baseline, df_sample, numerical_cols)
    drift_metrics["drift__ks"] = ks_metric(df_baseline, df_sample, numerical_cols)
    drift_metrics["drift__js"] = js_metric(
        df_baseline, df_sample, numerical_cols, categorical_cols
    )
    drift_metrics["concept_drift__es"] = es_metric(df_baseline, df_sample, ["score"])
    drift_metrics["concept_drift__ks"] = ks_metric(df_baseline, df_sample, ["score"])
    drift_metrics["concept_drift__js"] = js_metric(
        df_baseline, df_sample, ["score"], []
    )
    return drift_metrics


def ks_metric(df1, df2, numerical_columns):
    ks_tests = [
        ks_2samp(data1=df1.loc[:, col], data2=df2.loc[:, col])
        for col in numerical_columns
    ]
    p_values = [x[1] for x in ks_tests]
    list_of_pval = [f"{col}_p-value" for col in numerical_columns]
    ks_pvalues = dict(zip(list_of_pval, p_values))
    return ks_pvalues


def es_metric(df1, df2, numerical_columns):
    es_tests = []
    for col in numerical_columns:
        try:
            es_test = epps_singleton_2samp(x=df1.loc[:, col], y=df2.loc[:, col])
        except np.linalg.LinAlgError:
            es_test = [None, None]
        es_tests.append(es_test)
    p_values = [x[1] for x in es_tests]
    list_of_pval = [f"{col}_p-value" for col in numerical_columns]
    es_pvalues = dict(zip(list_of_pval, p_values))
    return es_pvalues


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
    return dict_output


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
    return sorted_shap_values

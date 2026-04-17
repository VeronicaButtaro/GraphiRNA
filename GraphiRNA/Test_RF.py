from sklearn.metrics import classification_report, confusion_matrix
from joblib import load
import pandas as pd
import os
import json


def load_model_and_evaluate(
        df_test,
        df_prod_test,
        selected_views_test,
        fold_number="default",
        model_output_path="../data/models",
        save_path="../data/processed",
        view_name=None
):

    df_expr_test = df_test.drop(columns=['country', 'disease', 'age', 'sex', 'apoe4'], errors='ignore')
    df_meta_test = df_test[['age', 'sex', 'apoe4']].copy()
    df_meta_test['sex'] = df_meta_test['sex'].map({'female': 0, 'male': 1})
    df_diagnosis_test = df_test[['disease']]

    views_test = {
        "expr_test": df_expr_test,
        "meta_test": df_meta_test,
        "prod_test": df_prod_test,
        "diagnosis_test": df_diagnosis_test
    }

    dataset_test = views_test[selected_views_test[0]]
    for view in selected_views_test[1:]:
        dataset_test = dataset_test.merge(views_test[view], left_index=True, right_index=True)
    dataset_test = dataset_test.merge(df_diagnosis_test, left_index=True, right_index=True)

    X_test = dataset_test.drop(columns=['disease'])
    y_test = dataset_test['disease']


    if view_name is None:
        view_name = "_".join(view.replace("_test", "") for view in selected_views_test)

    model_path = os.path.join(model_output_path, f"rf_model_{view_name}_{fold_number}.joblib")
    features_path = os.path.join(model_output_path, f"rf_features_{view_name}_{fold_number}.joblib")

    model = load(model_path)
    feature_names_train = load(features_path)

    common_features = [f for f in feature_names_train if f in X_test.columns]
    print(f"🔍 Feature match: {len(common_features)}/{len(feature_names_train)}")

    X_test = X_test[common_features].fillna(0)

    y_pred = model.predict(X_test)
    print(f"🔍 Predictions: {pd.Series(y_pred).value_counts().to_dict()}")
    score = model.score(X_test, y_test)
    print(f"✅ Score calcolato: {score:.3f}")


    active_classes = sorted(y_test.unique())
    report_dict = classification_report(
        y_test, y_pred,

    )
    report_str = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm)


    pred_df = pd.DataFrame({
        'y_true': y_test,
        'y_pred': y_pred
    })




    test_save_path = os.path.join(save_path, "test_results_df23")
    os.makedirs(test_save_path, exist_ok=True)

    rep_filename = f"test_report_{view_name}_{fold_number}.json"
    cm_filename = f"test_conf_matrix_{view_name}_{fold_number}.csv"


    with open(os.path.join(test_save_path, rep_filename), 'w') as f:
        json.dump(report_dict, f, indent=4)


    cm_df.to_csv(os.path.join(test_save_path, cm_filename))

    print(f"Test saved: {cm_filename} | Acc: {score:.3f}")

    return score, report_str,cm_df


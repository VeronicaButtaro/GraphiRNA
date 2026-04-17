from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump
import pandas as pd
import os
import json

def train_model_and_save(
    df_train,
    df_prod_train,
    selected_views_train,
    fold_number="default",
    model_output_path="../data/models",
    save_path="../data/processed"
):


    df_expr_train = df_train.drop(columns=['country', 'disease', 'age', 'sex', 'apoe4'], errors='ignore')
    df_meta_train = df_train[['age', 'sex', 'apoe4']].copy()
    df_meta_train['sex'] = df_meta_train['sex'].map({'female': 0, 'male': 1})
    df_diagnosis_train = df_train[['disease']]

    views_train = {
        "expr": df_expr_train,
        "meta": df_meta_train,
        "prod": df_prod_train,
        "diagnosis": df_diagnosis_train
    }


    dataset = views_train[selected_views_train[0]]
    for view in selected_views_train[1:]:
        dataset = dataset.merge(views_train[view], left_index=True, right_index=True)
    dataset = dataset.merge(df_diagnosis_train, left_index=True, right_index=True)

    X_train = dataset.drop(columns=['disease'])
    y_train = dataset['disease']

    # === Training ===
    rfc = RandomForestClassifier(n_estimators=100, random_state=42)
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_train)

    # === Classification report ===
    report_dict = classification_report(y_train, y_pred, output_dict=True,digits=4)

    # === Confusion matrix ===
    labels = sorted(y_train.unique())
    cm = confusion_matrix(y_train, y_pred, labels=labels)  # [web:16]
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)


    view_name = "_".join(selected_views_train)
    model_filename = f"rf_model_{view_name}_{fold_number}.joblib"
    features_filename = f"rf_features_{view_name}_{fold_number}.joblib"
    importance_filename = f"feature_importance_{view_name}_{fold_number}.csv"
    metrics_filename = f"classification_report_{view_name}_{fold_number}.json"
    cm_filename = f"confusion_matrix_{view_name}_{fold_number}.csv"


    os.makedirs(model_output_path, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)


    dump(rfc, os.path.join(model_output_path, model_filename))
    dump(X_train.columns.tolist(), os.path.join(model_output_path, features_filename))


    with open(os.path.join(save_path, metrics_filename), 'w') as f:
        json.dump(report_dict, f, indent=4)

    cm_df.to_csv(os.path.join(save_path, cm_filename))

    print(
        f"Model, features, feature importance, report, and confusion matrix "
        f"saved for views: {selected_views_train}, fold: {fold_number}"
    )

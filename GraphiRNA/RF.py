from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from joblib import dump
import os
import json
import pandas as pd
import numpy as np
from scipy import stats

def execute_train_single_fold(
    df_train,
    df_test,
    df_prod_train,
    df_prod_test,
    selected_views_train,
    selected_views_test,
    fold_number="default",
    save_path="../data/processed",
    model_output_path="../data/models"
):

    main_view = "_".join(selected_views_train)


    df_expr_train = df_train.drop(columns=['country', 'disease', 'age', 'sex', 'apoe4'], errors='ignore')
    df_expr_test = df_test.drop(columns=['country', 'disease', 'age', 'sex', 'apoe4'], errors='ignore')
    df_meta_train = df_train[['age', 'sex', 'apoe4']].copy()
    df_meta_train['sex'] = df_meta_train['sex'].map({'female': 0, 'male': 1})
    df_diagnosis_train = df_train[['disease']]
    df_diagnosis_test = df_test[['disease']]
    df_meta_test = df_test[['age', 'sex', 'apoe4']].copy()
    df_meta_test['sex'] = df_meta_test['sex'].map({'female': 0, 'male': 1})


    views_train = {
        "expr": df_expr_train,
        "prod": df_prod_train,
        "meta": df_meta_train,
        "diagnosis": df_diagnosis_train
    }
    views_test = {
        "expr_test": df_expr_test,
        "prod_test": df_prod_test,
        "meta_test": df_meta_test,
        "diagnosis_test": df_diagnosis_test
    }


    dataset = views_train[selected_views_train[0]]
    for view in selected_views_train[1:]:
        dataset = dataset.merge(views_train[view], left_index=True, right_index=True)
    dataset = dataset.merge(df_diagnosis_train, left_index=True, right_index=True)
    X_train = dataset.drop(columns=['disease'])
    y_train = dataset['disease']


    dataset_test = views_test[selected_views_test[0]]
    for view in selected_views_test[1:]:
        dataset_test = dataset_test.merge(views_test[view], left_index=True, right_index=True)
    dataset_test = dataset_test.merge(df_diagnosis_test, left_index=True, right_index=True)
    X_test = dataset_test.drop(columns=['disease'])
    X_test = X_test[X_train.columns]  # Allineamento colonne
    y_test = dataset_test['disease']

    # Training
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Saving models
    model_path = os.path.join(model_output_path, f"rf_model_fold_{fold_number}_{main_view}.joblib")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    dump(model, model_path)
    dump(model, model_path)
    print(f"Model saved to {model_path}")

    # Evaluation
    y_pred = model.predict(X_test)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    score = model.score(X_test, y_test)

    #Saving Predicions
    pred_df = pd.DataFrame({
        'y_true': y_test.values,
        'y_pred': y_pred
    }, index=X_test.index)

    pred_path = os.path.join(save_path, f"predictions_fold_{fold_number}_{main_view}.csv")
    pred_df.to_csv(pred_path, index=True)
    print(f"Predictions saved to {pred_path}")

    # Salvataggio report JSON

    report_output_path = os.path.join(save_path, f"classification_report_fold_{fold_number}_{main_view}.json")
    os.makedirs(os.path.dirname(report_output_path), exist_ok=True)
    with open(report_output_path, 'w') as f:
        json.dump(report_dict, f, indent=4)
    print(f"Report saved to {report_output_path}")
    return report_dict,score



def aggregate_classification_reports(fold_ids, views, save_path):
    """
    Aggregate the classification reports by class, calculating the mean,
    standard deviation and 95% confidence interval for precision, recall and F1-score.


    """

    if isinstance(views, (list, tuple)):
        view_name = "_".join(views)
    else:
        view_name = views

    per_class_results = {}

    for fold in fold_ids:
        file_path = os.path.join(save_path, f"classification_report_fold_{fold}_{view_name}.json")
        if not os.path.exists(file_path):
            print(f"Missing report for fold {fold} - {views}")
            continue

        with open(file_path, 'r') as f:
            report = json.load(f)

        for class_name, metrics in report.items():
            if class_name in ['accuracy', 'macro avg', 'weighted avg']:
                continue
            if class_name not in per_class_results:
                per_class_results[class_name] = {k: [] for k in metrics.keys()}
            for k, v in metrics.items():
                per_class_results[class_name][k].append(v)


    if not per_class_results:
        print(f"No reports found for views {views}, skipping aggregation.")
        return


    def compute_stats(values):
        values = np.array(values)
        n = len(values)
        if n == 0:
            return np.nan, np.nan, np.nan, np.nan
        mean = np.mean(values)
        std = np.std(values, ddof=1) if n > 1 else 0.0
        if n > 1:
            sem = std / np.sqrt(n)
            ci_low, ci_high = stats.t.interval(0.95, n - 1, loc=mean, scale=sem)
        else:
            ci_low, ci_high = mean, mean
        return mean, std, ci_low, ci_high


    writer_path = os.path.join(save_path, f"aggregated_classification_report_{view_name}.xlsx")
    with pd.ExcelWriter(writer_path, engine='openpyxl') as writer:
        all_classes_stats = []

        for class_name, metrics_dict in per_class_results.items():
            row_data = {'Class': class_name}


            for metric_name, values in metrics_dict.items():
                mean, std, ci_low, ci_high = compute_stats(values)
                row_data[f'{metric_name}_mean'] = mean
                row_data[f'{metric_name}_std'] = std
                row_data[f'{metric_name}_CI_low'] = ci_low
                row_data[f'{metric_name}_CI_high'] = ci_high

            all_classes_stats.append(row_data)

        result_df = pd.DataFrame(all_classes_stats)


        metric_names = list(per_class_results[list(per_class_results.keys())[0]].keys())
        ordered_cols = ['Class']
        for metric in metric_names:
            ordered_cols.extend([f'{metric}_mean', f'{metric}_std', f'{metric}_CI_low', f'{metric}_CI_high'])

        result_df = result_df[ordered_cols]
        result_df.to_excel(writer, sheet_name="aggregated", index=False)

    print(f"Aggregated report saved to {writer_path}")

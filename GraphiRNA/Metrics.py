import os
import pandas as pd
import numpy as np
from scipy import stats

import os
import pandas as pd
import numpy as np
from scipy import stats


def aggregate_and_save_results(accuracies, reports, selected_views_train, save_path):

    # Stats accuracy
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies, ddof=1)
    ci_accuracy = stats.t.interval(0.95, len(accuracies) - 1, loc=mean_accuracy, scale=stats.sem(accuracies))
    print(f"Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f} [CI: {ci_accuracy[0]:.4f}-{ci_accuracy[1]:.4f}]")

    # Safe stats function (fix SciPy warnings + NaN)
    def safe_stats(values):
        values = np.asarray(values)
        values = values[~np.isnan(values)]  # Filtra NaN
        if len(values) < 2:
            return np.nan, np.nan, np.nan, np.nan
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        ci = stats.t.interval(0.95, len(values) - 1, loc=mean, scale=stats.sem(values))
        return mean, std, ci[0], ci[1]


    macro_p, macro_r, macro_f1 = [], [], []
    w_p, w_r, w_f1 = [], [], []

    for report in reports:
        macro = report.get('macro avg', {})
        weighted = report.get('weighted avg', {})
        macro_p.append(macro.get('precision', np.nan))
        macro_r.append(macro.get('recall', np.nan))
        macro_f1.append(macro.get('f1-score', np.nan))
        w_p.append(weighted.get('precision', np.nan))
        w_r.append(weighted.get('recall', np.nan))
        w_f1.append(weighted.get('f1-score', np.nan))


    stats_dict = {
        'macro_precision': safe_stats(macro_p),
        'macro_recall': safe_stats(macro_r),
        'macro_f1': safe_stats(macro_f1),
        'weighted_precision': safe_stats(w_p),
        'weighted_recall': safe_stats(w_r),
        'weighted_f1': safe_stats(w_f1)
    }


    flat_dict = {}
    for k, (mean, std, ci_l, ci_h) in stats_dict.items():
        flat_dict[f'{k}_mean'] = mean
        flat_dict[f'{k}_std'] = std
        flat_dict[f'{k}_CI_low'] = ci_l
        flat_dict[f'{k}_CI_high'] = ci_h

    results_df = pd.DataFrame([{
        **flat_dict,
        'Accuracy_mean': mean_accuracy,
        'Accuracy_std': std_accuracy,
        'Accuracy_CI_low': ci_accuracy[0],
        'Accuracy_CI_high': ci_accuracy[1],
        'view': '-'.join(selected_views_train)
    }])

    file_path = f"{save_path}/final_results.csv"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if os.path.exists(file_path):
        existing_df = pd.read_csv(file_path)
        results_df = pd.concat([existing_df, results_df], ignore_index=True)
        results_df.to_csv(file_path, index=False)
    else:
        results_df.to_csv(file_path, index=False)

    print(f"Results with stats saved to {file_path} (shape: {results_df.shape}, cols: {len(results_df.columns)})")

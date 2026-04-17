
import gc
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, false_discovery_control

def execute_corr_graph(df, alpha, output_file, output_file_correct):
    df_expr = df.drop(columns=['disease','age','sex','apoe4','country'], errors='ignore').astype(np.float32)
    cols = df_expr.columns.tolist()
    n = len(cols)
    vals = df_expr.values

    corr_arr = np.eye(n, dtype=np.float32)
    pval_arr = np.ones((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(i+1, n):
            mask = np.isfinite(vals[:, i]) & np.isfinite(vals[:, j])
            if mask.sum() >= 2:
                r, p = pearsonr(vals[mask, i], vals[mask, j])
                corr_arr[i, j] = corr_arr[j, i] = r
                pval_arr[i, j] = pval_arr[j, i] = p


    iu1 = np.triu_indices(n, k=1)
    pvals_flat = pval_arr[iu1]
    valid_mask = np.isfinite(pvals_flat)
    pvals_adj = np.ones_like(pvals_flat)
    if valid_mask.sum() > 0:
        pvals_adj[valid_mask] = false_discovery_control(pvals_flat[valid_mask], method='bh')

    p_adj_arr = np.ones((n, n), dtype=np.float32)
    p_adj_arr[iu1] = pvals_adj
    p_adj_arr[(iu1[1], iu1[0])] = pvals_adj


    corr_no_corr = corr_arr.copy()
    corr_no_corr[pval_arr > alpha] = 0
    pd.DataFrame(corr_no_corr, index=cols, columns=cols).to_csv(output_file)
    del corr_no_corr; gc.collect()

    corr_arr[p_adj_arr > alpha] = 0
    pd.DataFrame(corr_arr, index=cols, columns=cols).to_csv(output_file_correct)
    del corr_arr, pval_arr, p_adj_arr; gc.collect()




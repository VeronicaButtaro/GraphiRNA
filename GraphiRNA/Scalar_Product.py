import pandas as pd
import numpy as np
def scalar_product(df,emb,compl_emb,output):

    columns_to_drop = ['country', 'disease', 'age', 'sex', 'apoe4']
    columns_present = [col for col in columns_to_drop if col in df.columns]
    df = df.drop(columns=columns_present)
    df=df.fillna(0)
    row_sums = df.sum(axis=1)

    emb.index=compl_emb.index
    emb = emb[~emb.index.str.startswith('non')]

    df_normalized = df.div(row_sums, axis=0)
    prod=np.dot(df_normalized,emb)
    prod=pd.DataFrame(prod)
    prod.index=df.index
    prod.columns=emb.columns
    prod.to_csv(output)
    return prod





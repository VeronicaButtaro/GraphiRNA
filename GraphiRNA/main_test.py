
from Test_RF import load_model_and_evaluate
from dataset import MiRNADataProcessor
from Scalar_Product import scalar_product
import torch
import pandas as pd


processor = MiRNADataProcessor(
    raw_data_path="../data/raw",
    processed_data_path="../data/processed"
)

df1 = pd.read_csv(f"{processor.processed_data_path}/df_84_tot.csv", index_col=0)
df2 = pd.read_csv(f"{processor.processed_data_path}/df_93_tot.csv", index_col=0)
df3 = pd.read_csv(f"{processor.processed_data_path}/df_23_tot.csv", index_col=0)
train_df=pd.concat([df1,df2])
train_df=train_df[train_df['disease'] != "MCI-NC"]
train_df['disease'] = train_df['disease'].replace({'Alzheimer Disease': 'AD'})
train_df = train_df[~((train_df['age'] >= 85) & (train_df['disease'] == 'AD'))]
train_df.to_csv(f"{processor.processed_data_path}/train_df_84_93.csv")
test_df=df3
test_df = test_df[test_df['disease'] != "MCI-NC"]
test_df['disease'] = test_df['disease'].replace({'Alzheimer Disease': 'AD'})
#test_df = test_df[~((test_df['age'] >= 85) & (test_df['disease'] == 'AD'))]
test_df.to_csv(f"{processor.processed_data_path}/test_df_23.csv", index=True)



def main(
    df_test_path,
    fold_number="final",
    model_output_path="../models/23",
save_path="..data/processed/hts/results/23",
    views_to_test=None
):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = MiRNADataProcessor(
        raw_data_path="../data/raw",
        processed_data_path="../data/processed"
    )

    processed_path = processor.processed_data_path
    train_df=pd.read_csv(f"{processor.processed_data_path}/train_df_84_93.csv",index_col=0)
    train_df = train_df[~train_df['disease'].isin(["MCI-NC", "MCI"])]


    complexive_df_path = "../data/processedcomplexive_embeddings_default.csv"
    complexive_df = pd.read_csv(complexive_df_path, index_col=0)
    graph_conv_output_path = f"{processor.processed_data_path}/graph_embeddings_84_93_512_0.05.csv"
    node_embeddings = pd.read_csv(graph_conv_output_path, index_col=0)


    df_test= pd.read_csv(df_test_path, index_col=0)

    prod_test_path='../data/processed/prod_test_84_93_512_0.05.csv'

    df_test = df_test[df_test.columns.intersection(train_df.columns)]

    missing_columns = list(set(train_df.columns) - set(df_test.columns))


    if missing_columns:
        new_data = pd.DataFrame(columns=missing_columns, index=df_test.index)
        df_test = pd.concat([df_test, new_data], axis=1)

    scalar_product(df_test, node_embeddings, complexive_df,
                   '../data/processed/prod_test_84_93_512_0.05.csv')
    prod_test = pd.read_csv(prod_test_path, index_col=0)

    if views_to_test is None:
        views_to_test = [
            (['expr_test'], "expr"),
            (['expr_test', 'meta_test'], "expr_meta"),
            (['prod_test', 'meta_test'], "prod_meta"),
            (['expr_test', 'prod_test'], "expr_prod"),
            (['prod_test'], "prod"),
            (['expr_test', 'prod_test', 'meta_test'], "expr_prod_meta"),
            (['meta_test'], "meta")
        ]


    for selected_views, view_name in views_to_test:

        try:
            print(f"Model: {model_output_path} | fold: {fold_number} | view: {view_name}")
            score, report_str,cm_df = load_model_and_evaluate(
                df_test=df_test,
                df_prod_test=prod_test,
                selected_views_test=selected_views,
                fold_number=fold_number,
                model_output_path=model_output_path,
                save_path="..data/results/23",
                view_name=view_name
            )

            print(f"Accuracy: {score:.4f}")
            print(f"Classification Report (macro avg): {report_str}")
            print(f"Confusion Matrix: {cm_df}")
        except Exception as e:
            print(f"Failed to evaluate model for view {view_name}: {e}")


if __name__ == "__main__":
    custom_views = [
        (['prod_test',], "prod")

    ]

    main(
        df_test_path=f"{processor.processed_data_path}/test_df_23.csv",
        fold_number="final",
        model_output_path="../models/23",
        save_path="..data/processed/results/23",

        views_to_test=custom_views
    )






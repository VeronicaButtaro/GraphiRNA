from dataset import MiRNADataProcessor
import pandas as pd
from RnaBERT import execute_rna_bert
from Correlation import execute_corr_graph
from P_value_matrix_neg import normalize_p_value
from Graph_Conv import execute_graph_conv
from Scalar_Product import  scalar_product
from RF import execute_train_single_fold, aggregate_classification_reports
from Metrics import aggregate_and_save_results
import torch

if __name__ == "__main__":

    """
    Data Preprocessing
    """

    processor = MiRNADataProcessor(
        normalized_data_path="../data/normalized",
        processed_data_path="../data/processed"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor.load_data()
    processor.preprocess()
    processor.save_processed_data()

    #Kfold
    processor.create_folds(n_splits=5)

    """
    MiRNA embedding initialization with pre trained RNABERT
    """

    execute_rna_bert()
    dim_emb=512
    for fold in range(5):

        train_df = pd.read_csv(f"{processor.processed_data_path}/train_fold_{fold}.csv", index_col=0)
        test_df = pd.read_csv(f"{processor.processed_data_path}/test_fold_{fold}.csv", index_col=0)

        df_diagnosis_train = train_df[['disease']]
        df_diagnosis_test = test_df[['disease']]

        """
        Correlation matrix calculation, without correction(correlation_output_path) 
        and with correction(correlation_output_correct_path)

        """
        correlation_output_path = f"{processor.processed_data_path}/correlation_matrix_train_05_fold_{fold}.csv"
        correlation_output_correct_path = f"{processor.processed_data_path}/correlation_matrix_correct_train_05_fold_{fold}.csv"

        execute_corr_graph(train_df, 0.05, correlation_output_path,correlation_output_path)

        corr_matrix = pd.read_csv(correlation_output_path, index_col=0)



        """
        The normalize p value` function handles the insertion of artificial nodes 
        
        """
        graph_output_path = f"{processor.processed_data_path}/graph_edges_train_05_{dim_emb}_fold_{fold}.csv"
        file_suffix = f"fold_{fold}"
        complexive_df = normalize_p_value(
            p_value_matrix=corr_matrix,
            output1=graph_output_path,
            file_suffix=file_suffix,
            path=processor.processed_data_path
        )


        edge_df = pd.read_csv(graph_output_path)

        graph_conv_output_path = f"{processor.processed_data_path}/graph_embeddings_05_{dim_emb}_fold_{fold}_.csv"
        execute_graph_conv(edge_df, complexive_df,{dim_emb}, graph_conv_output_path)
        node_embeddings=pd.read_csv(graph_conv_output_path,index_col=0)

        prod_train_path=f"{processor.processed_data_path}/subj_embeddings_train_05_fold_{fold}_{dim_emb}.csv"
        prod_test_path=f"{processor.processed_data_path}/subj_embeddings_test_05_fold_{fold}_{dim_emb}.csv"

        """
        Scalar_product" to obtain the subject embedding
        
        """
        scalar_product(train_df,node_embeddings , complexive_df,prod_train_path)
        scalar_product(test_df,node_embeddings ,complexive_df,prod_test_path)
        prod_train=pd.read_csv(f"{processor.processed_data_path}/subj_embeddings_train_05_fold_{fold}_{dim_emb}.csv" ,index_col=0)
        prod_test=pd.read_csv(f"{processor.processed_data_path}/subj_embeddings_test_05_fold_{fold}_{dim_emb}.csv",index_col=0 )
    """
    Multimodal Classification Model
    
    """
    view_combinations = [
        (['expr'], ['expr_test']),
        (['meta'], ['meta_test']),
        (['expr', 'meta'], ['expr_test', 'meta_test']),
        (['prod', 'meta'], ['prod_test', 'meta_test']),
        (['expr', 'prod'], ['expr_test', 'prod_test']),
        (['prod'], ['prod_test']),
        (['expr', 'prod', 'meta'], ['expr_test', 'prod_test', 'meta_test'])
    ]


    for selected_views_train, selected_views_test in view_combinations:
        accuracies = []
        reports = []
        view_name = "_".join(selected_views_train)


        folds= []

        for fold in range(5):
            print(f"==== Training Fold {fold} for views {selected_views_train} ====")


            train_df = pd.read_csv(f"{processor.processed_data_path}/train_fold_{fold}.csv", index_col=0)
            test_df = pd.read_csv(f"{processor.processed_data_path}/test_fold_{fold}.csv", index_col=0)
            prod_train = pd.read_csv(f"{processor.processed_data_path}/subj_embeddings_train_05_fold_{fold}_{dim_emb}.csv",
                                     index_col=0)
            prod_test = pd.read_csv(f"{processor.processed_data_path}/subj_embeddings_test_05_fold_{fold}_{dim_emb}.csv",
                                    index_col=0)
            folds.append((train_df, test_df, prod_train, prod_test))


            report,score = execute_train_single_fold(
                df_train=train_df,
                df_test=test_df,
               df_prod_train=prod_train,
                df_prod_test=prod_test,
                selected_views_train=selected_views_train,
                selected_views_test=selected_views_test,
                fold_number=fold,
                save_path=f"../data/processed/results/no_corr/{dim_emb}/05",
                model_output_path=f"../models/no_corr/{dim_emb}/05"
            )

            reports.append(report)
            accuracies.append(score)


        aggregate_classification_reports(
            fold_ids=["0", "1", "2", "3", "4"],
            views=[view_name],
            save_path=f"../data/processed/results/no_corr/{dim_emb}/05"
        )

        aggregate_and_save_results(
            accuracies,
            reports,
            selected_views_train,
            save_path=f"../data/processed/results/no_corr/{dim_emb}/05"
        )



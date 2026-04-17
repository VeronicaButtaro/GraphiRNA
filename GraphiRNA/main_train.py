from dataset import MiRNADataProcessor
from RnaBERT import execute_rna_bert
from Correlation import *
from P_value_matrix_neg import normalize_p_value
from Graph_Conv import execute_graph_conv
from Scalar_Product import  scalar_product
from Train_RF import train_model_and_save
import torch


processor = MiRNADataProcessor(
   processed_data_path="../data/processed"
)

processor.load_data()
processor.preprocess()
processor.save_processed_data()
df1=pd.read_csv(f"{processor.processed_data_path}/df_84_tot.csv",index_col=0)
df2=pd.read_csv(f"{processor.processed_data_path}/df_93_tot.csv",index_col=0)
df3=pd.read_csv(f"{processor.processed_data_path}/df_23_tot.csv",index_col=0)
train_df=pd.concat([df1,df2])
print(train_df['disease'].unique())
train_df = train_df[~train_df['disease'].isin(["MCI-NC", "MCI"])]
print(train_df['disease'].unique())
train_df['disease'] = train_df['disease'].replace({'Alzheimer Disease': 'AD'})
train_df = train_df[~((train_df['age'] >= 85) & (train_df['disease'] == 'AD'))]
test_df=df3
test_df = test_df[test_df['disease'] != "MCI-NC"]
test_df['disease'] = test_df['disease'].replace({'Alzheimer Disease': 'AD'})
execute_rna_bert()



correlation_output_path = f"{processor.processed_data_path}/correlation_matrix_train_84_93_0.05.csv"
correlation_output_path_correct=f"{processor.processed_data_path}/correlation_matrix_train_84_93_0.05_corr.csv"

execute_corr_graph(train_df,0.05, correlation_output_path, correlation_output_path_correct)

corr_matrix = pd.read_csv(correlation_output_path, index_col=0)


graph_output_path = f"{processor.processed_data_path}/graph_edges_train_fold_84_93_512_0.05.csv"

complexive_df = normalize_p_value(
    p_value_matrix=corr_matrix,
 output1=graph_output_path,
   path=processor.processed_data_path
)


edge_df = pd.read_csv(graph_output_path)

graph_conv_output_path = f"{processor.processed_data_path}/graph_embeddings_84_93_512_0.05.csv"

execute_graph_conv(edge_df, complexive_df,512, graph_conv_output_path)



node_embeddings=pd.read_csv(graph_conv_output_path,index_col=0)

prod_train_path=f"{processor.processed_data_path}/subj_embeddings_train_84_93_512_0.05.csv"
train_df = train_df[~train_df['disease'].isin(["MCI-NC", "MCI"])]
scalar_product(train_df,node_embeddings , complexive_df,prod_train_path)

prod_train=pd.read_csv(f"{processor.processed_data_path}/subj_embeddings_train_84_93_512_0.05.csv" ,index_col=0)


view_combinations = [
    (['expr'], ['expr_test']),
    (['expr', 'meta'], ['expr_test', 'meta_test']),
    (['prod', 'meta'], ['prod_test', 'meta_test']),
    (['expr', 'prod'], ['expr_test', 'prod_test']),
    (['prod'], ['prod_test']),
    (['expr', 'prod', 'meta'], ['expr_test', 'prod_test', 'meta_test']),
    (['meta'], ['meta_test'])
]


for selected_views_train, selected_views_test in view_combinations:
    accuracies = []
    reports = []

    train_model_and_save(
        df_train=train_df,
        df_prod_train=prod_train,
        selected_views_train=selected_views_train,
        fold_number="final",
        model_output_path="../models/23",
        save_path="../data/processed/results/23"
    )
device = "cuda" if torch.cuda.is_available() else "cpu"

import pandas as pd
import numpy as np

import torch
import pickle


def normalize_p_value(p_value_matrix, output1,file_suffix="default",path="../data/processed/"):
    def normalize_embeddings(embeddings_tensor):
        """
        Normalises the embeddings to values between 0 and 1.
        """
        min_vals = embeddings_tensor.min(dim=0, keepdim=True).values
        max_vals = embeddings_tensor.max(dim=0, keepdim=True).values
        return (embeddings_tensor - min_vals) / (max_vals - min_vals)

    """
    Management of negative correlations.
    
    """

    # Node Mapping
    mapping_nodes = {k: v for k, v in enumerate(p_value_matrix.columns)}
    with open("../data/processed/mapping_miRNA", 'wb') as file:
        pickle.dump(mapping_nodes, file)
    print("Contenuto di mapping_nodes:", mapping_nodes)

    """
    
    
    Load the embeddings and filter them to include only the nodes present in p_value_matrix.columns
    
    
    """
    embeddings = torch.load('../data/processed/miRNA.pt',weights_only=False)
    embeddings = pd.DataFrame.from_dict(embeddings, orient='index')

    embeddings = embeddings.loc[p_value_matrix.columns]
    embeddings_tensor = torch.tensor(embeddings.values, dtype=torch.float)

    embeddings_normalized = normalize_embeddings(embeddings_tensor)
    embeddings_normalized = pd.DataFrame(embeddings_normalized.numpy(), index=embeddings.index)


    # Creating a dataframe for a graph
    upper_tri = p_value_matrix.where(np.triu(np.ones(p_value_matrix.shape), k=1).astype(bool))
    stacked = upper_tri.stack()
    stacked = stacked[stacked != 0]


    sources = []
    targets = []
    weights = []
    artificial_nodes = {}

    new_node_counter = len(p_value_matrix)  # Indice per i nodi artificiali

    for (source, target), weight in stacked.items():
        abs_weight = abs(weight)

        if weight < 0:  # if the correlation is negative, it creates artificial nodes
            non_source = f"non_{source}"
            non_target = f"non_{target}"

            # Add the artificial nodes to the dictionary
            if non_source not in artificial_nodes:

                artificial_nodes[non_source] = (1 - embeddings_normalized.loc[source].values).tolist()
            if non_target not in artificial_nodes:

                artificial_nodes[non_target] = (1 - embeddings_normalized.loc[target].values).tolist()

            # Create a link between the original node and the artificial node
            sources.extend([non_source, source])
            targets.extend([target, non_target])
            weights.extend([abs_weight, abs_weight])

        else:  # If the correlation is positive, add it as usual
            sources.append(source)
            targets.append(target)
            weights.append(abs_weight)


    result = {
        'source': sources,
        'target': targets,
        'weight': weights
    }
    df_graph = pd.DataFrame(result)

    print(f"✓ edge_df int: {df_graph.shape}")

    df_graph.to_csv(output1, index=False)

    # Creating and saving artificial node embeddings


    artificial_embeddings = pd.DataFrame.from_dict(artificial_nodes, orient='index')
    artificial_embeddings.to_csv(f"{path}artificial_embeddings_{file_suffix}.csv")
    complexive_embedding = pd.concat([embeddings_normalized, artificial_embeddings])
    complexive_embedding.to_csv(f"{path}complexive_embeddings_{file_suffix}.csv")
    print(artificial_embeddings.shape)
    return complexive_embedding


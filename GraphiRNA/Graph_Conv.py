import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv
from torch_geometric.utils import negative_sampling, to_undirected
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
import gc

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Set seed
set_seed(42)


def execute_graph_conv(df, complexive_embeddings, dim_emb,output):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class GCNModel(nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels):
            super(GCNModel, self).__init__()
            self.conv1 = GraphConv(in_channels, hidden_channels)
            self.conv2 = GraphConv(hidden_channels, out_channels)

        def forward(self, x, edge_index, edge_weight=None):
            x = self.conv1(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, edge_index, edge_weight=edge_weight)
            return x

    # Setup
    in_channels = complexive_embeddings.shape[1]
    hidden_channels = 32
    out_channels = dim_emb
    x = torch.tensor(complexive_embeddings.values, dtype=torch.float).to(device)

    # Map nodes to indices
    node_mapping = {node: i for i, node in enumerate(pd.concat([df['source'], df['target']]).unique())}
    df['source'] = df['source'].map(node_mapping)
    df['target'] = df['target'].map(node_mapping)

    edge_index = torch.tensor(df[['source', 'target']].values.T, dtype=torch.long).to(device)
    edge_weight = torch.tensor(df['weight'].values, dtype=torch.float).to(device)
    edge_index, edge_weight = to_undirected(edge_index, edge_weight)

    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight).to(device)

    train_loader = NeighborLoader(data, num_neighbors=[150, 50], batch_size=256, shuffle=True) #default 256
    test_loader = NeighborLoader(data, num_neighbors=[150, 50], batch_size=256, shuffle=False) #default vicini 150,50

    model = GCNModel(in_channels, hidden_channels, out_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    best_loss = float('inf')


    def train():
        model.train()
        total_loss = 0
        total_samples = 0
        for batch in train_loader:
            optimizer.zero_grad()
            batch = batch.to(device)
            z = model(batch.x, batch.edge_index, batch.edge_weight)
            batch_mask = batch.edge_index[0] < batch.batch_size
            pos_edge_index = batch.edge_index[:, batch_mask]
            neg_edge_index = negative_sampling(pos_edge_index, num_nodes=batch.num_nodes, num_neg_samples=pos_edge_index.size(1))
            neg_edge_index = neg_edge_index.to(torch.long)
            pos_score = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=-1)
            neg_score = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=-1)
            pos_loss = F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score, dtype=torch.float))
            neg_loss = F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score, dtype=torch.float))
            loss = pos_loss + neg_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.batch_size
            total_samples += batch.batch_size
        return total_loss / total_samples

    def save_node_embeddings():
        model.eval()
        all_embeddings = []
        for batch in test_loader:
            batch = batch.to(device)
            with torch.no_grad():
                z = model(batch.x, batch.edge_index, batch.edge_weight)
            all_embeddings.append(z[:batch.batch_size].cpu())
        embeddings = torch.cat(all_embeddings, dim=0).numpy()
        pd.DataFrame(embeddings, index=complexive_embeddings.index).to_csv(output, index=True)

    for epoch in range(1, 250):
        loss = train()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), 'best_model.pth')


    model.load_state_dict(torch.load('best_model.pth'))
    save_node_embeddings()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

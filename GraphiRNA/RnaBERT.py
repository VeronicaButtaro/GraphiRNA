import pandas as pd
import torch
from collections import OrderedDict
from RNABERT.bert import get_config, BertModel, BertForMaskedLM

def execute_rna_bert():
    config = get_config('RNABERT/RNA_bert_config.json')
    config.hidden_size = config.num_attention_heads * config.multiple

    model = BertModel(config)
    model = BertForMaskedLM(config, model).eval()

    state_dict = torch.load('RNABERT/bert_mul_2.pth', map_location='cpu')
    model.load_state_dict(OrderedDict((key[7:], value) for key, value in state_dict.items()))
    mirna_df = pd.read_csv('../data/processed/miRNA_seq_tot', index_col=0)
    nan_indices = mirna_df.index.isna()

    #mirna_df.loc[nan_indices, 'ID'] = range(1, nan_indices.sum() + 1)
    mirna_df.loc[nan_indices, 'Sequence'] = range(1, nan_indices.sum() + 1)
    mirna = dict(zip(mirna_df.index, mirna_df['Sequence']))

    mapping = {'A': 2, 'U': 3, 'G': 4, 'C': 5}
    default_value = 0
    for name, sequence in mirna.items():
        try:
            input_ids = [mapping.get(c, default_value) for c in sequence]

            input_ids += [0] * (config.max_position_embeddings - len(sequence))
            input_tensor = torch.tensor([input_ids])


            encoded_layers = model(input_tensor)[-1]

            mirna[name] = torch.mean(encoded_layers[0], dim=0).detach().numpy()
        except KeyError as e:
            print(f"Character '{e.args[0]}' in sequence '{sequence}' is not in the mapping dictionary")


    torch.save(mirna, '../data/processed/miRNA.pt')
execute_rna_bert()
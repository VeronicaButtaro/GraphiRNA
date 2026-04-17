import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


mirna_df = pd.read_csv('../data/processed/miRNA_seq_tot', index_col=0)
mirna = dict(zip(mirna_df.index, mirna_df['Sequence']))

# === K-mer ===
def k_mers(k, seq):
    seq = str(seq).upper().strip()
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]

mi_names = list(mirna.keys())
mi_seq_mers = [k_mers(3, seq) for seq in mirna.values()]

# ===  Train Doc2Vec ===
corpus = [TaggedDocument(words=mers, tags=[name]) for name, mers in zip(mi_names, mi_seq_mers)]
model = Doc2Vec(vector_size=120, window=5, min_count=1, dm=1, workers=4, epochs=100)
model.build_vocab(corpus)
model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)

# === Embedding extraction ===
embeddings = pd.DataFrame(
    {name: model.infer_vector(mers) for name, mers in zip(mi_names, mi_seq_mers)}
).T
embeddings.index.name = 'mirna_id'

embeddings.to_csv('../data/processed/doc2vec_node_embeddings.csv')


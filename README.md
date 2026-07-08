# A novel graph-based patient embedding method for the diagnosis of Alzheimer's disease from microRNA expression data

**The repository contains code referred to the work:**
*Veronica Buttaro, Antonio Pellicani, Gianvito Pio, Domenica D'Elia, Cristina Pizzulli, Michelangelo Ceci*
[*A novel graph-based patient embedding method for the diagnosis of Alzheimer's disease from microRNA expression data*](https://)

Please cite our work if you find it useful for your research and work.

```bibtex
@ARTICLE{GraphiRNA,
  author={Buttaro, Veronica and Pellicani, Antonio and Pio, Gianvito and D'Elia, Domenica and Pizzulli, Cristina and Ceci, Michelangelo},
  journal={Data Mining and Knowledge Discovery},
  title={A novel graph-based patient embedding method for the diagnosis of Alzheimer's disease from microRNA expression data},
  year={??},
  volume={??},
  number={??},
  pages={??},
  doi={??}}
```

## Overview
GraphiRNA builds a patient embedding for Alzheimer's disease diagnosis from miRNA expression data by:
1. building a miRNA correlation graph from the training set, retaining only statistically significant Pearson correlations (the method is evaluated in two variants: **GraphiRNA⁻**, using uncorrected p-values, and **GraphiRNA⁺**, using Benjamini–Hochberg FDR-corrected p-values);
2. handling negative correlations by introducing synthetic ("non_X") nodes, so that the sign of each correlation is preserved rather than discarded or replaced by its absolute value;
3. initializing node features with RNA sequence embeddings extracted using the pretrained RNABERT model;
4. learning graph-based miRNA node embeddings with a GraphConv model (self-supervised, link-prediction objective, weighted-mean aggregation);
5. projecting each patient's (normalized) expression profile onto those node embeddings to obtain a patient-level graph embedding;
6. training/evaluating Random Forest classifiers on this embedding alone or combined with raw expression and clinical metadata.

## Data
In our experiments, we assessed GraphiRNA's performance using real-world miRNA expression data of patients affected by Alzheimer's disease (AD) and Mild Cognitive Impairment (MCI) obtained from the GEO repository. Specifically, we employed [GSE120584](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE120584), [GSE150693](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE150693), and [GSE242923](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE242923).

The three series were used in two settings:
- **Cross-validation:** all three series are merged and evaluated via stratified 5-fold cross-validation.
- **External test:** GSE120584 + GSE150693 serve as the discovery/training cohort, while GSE242923 is held out as an independent external test cohort — a transfer-learning setting, since the platforms and normalization protocols differ between cohorts.

The normalized expression matrices and sample metadata for all three series are already provided under [data/normalized/](data/normalized/).

## Repository structure
```
GraphiRNA/
├── data/
│   ├── normalized/          # Input data (provided): normalized expression + metadata for the 3 GEO series
│   │   ├── GSE120584.csv / .metadata.csv
│   │   ├── GSE150693.csv / .metadata.csv
│   │   ├── GSE242923.csv / .metadata.csv
│   │   └── df.na.csv        # miRNA ID -> sequence (miRBase) lookup table
│   └── processed/           # miRNA_seq_tot is provided; everything else is generated at runtime
│       └── miRNA_seq_tot    # miRNA ID -> sequence table consumed by RnaBERT.py
└── GraphiRNA/                # Source code (run scripts from inside this folder)
    ├── dataset.py            # MiRNADataProcessor: loads/merges the 3 datasets, harmonizes disease labels,
    │                         # builds the stratified 5-fold train/test splits
    ├── RNABERT/               # Pretrained RNABERT weights (bert_mul_2.pth) + config used to embed miRNA sequences
    ├── RnaBERT.py             # Embeds every miRNA sequence with RNABERT -> data/processed/miRNA.pt
    ├── Correlation.py         # Pairwise Pearson correlation + p-values between miRNAs (co-expression graph),
    │                         # with optional Benjamini-Hochberg correction (GraphiRNA+ variant)
    ├── P_value_matrix_neg.py  # Correlation matrix -> graph edge list; negative correlations get artificial "non_X" nodes
    ├── Graph_Conv.py          # GraphConv (torch_geometric)  model -> final miRNA node embeddings
    ├── Scalar_Product.py      # Projects each patient's expression profile onto the miRNA graph embeddings ("prod" view)
    ├── RF.py                  # Random Forest training/evaluation per cross-validation fold and per view combination
    ├── Train_RF.py            # Trains the final Random Forest on the full cohort (GSE120584 + GSE150693)
    ├── Test_RF.py             # Evaluates a trained model on the external test cohort (GSE242923)
    ├── Metrics.py             # Aggregates accuracy/precision/recall/F1 across folds with 95% confidence intervals
    ├── doc2vec.py             # Optional alternative miRNA sequence embedding (not used by the default pipeline)
    ├── iLearn-master/         # Bundled third-party feature-extraction toolkit (not invoked by the pipeline scripts)
    ├── main.py                # Entry point: 5-fold stratified cross-validation on the combined dataset
    ├── main_train.py          # Entry point: trains the final model on the full cohort (GSE120584 + GSE150693)
    └── main_test.py           # Entry point: evaluates the final model on the GSE242923 external cohort
```

## How to reproduce
### 1. Setup
```bash
git clone <this repository>
cd GraphiRNA
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```
The pinned `torch==2.6.0+cu118` build requires a CUDA 11.8 GPU; if you don't have one, install a CPU (or matching CUDA) build of PyTorch 2.6 instead — the code automatically falls back to CPU when `torch.cuda.is_available()` is `False`. A GPU is strongly recommended for the `Graph_Conv.py` training step.

### 2. Run the pipeline
All commands below are run from inside the `GraphiRNA/` source folder, since scripts use paths relative to it:
```bash
cd GraphiRNA
```

**a. Cross-validation experiment** (5-fold stratified CV over the combined dataset, for every view combination — expression only, metadata only, graph embedding only, and their combinations):
```bash
python main.py
```
Outputs: per-fold correlation matrices, graph embeddings, patient embeddings, trained RF models (`../models/no_corr/512/05/`) and per-fold/aggregated classification reports (`../data/processed/results/no_corr/512/05/`). The `no_corr` path corresponds to the GraphiRNA⁻ variant (uncorrected p-values); `512` and `05` denote the embedding dimensionality `q` and the significance level `α`, respectively.

**b. Train the final model** on the discovery cohort (GSE120584 + GSE150693):
```bash
python main_train.py
```
Outputs: trained RF models per view combination in `../models/23/` and training reports in `../data/processed/results/23/`.

**c. Evaluate on the external test cohort** (GSE242923):
```bash
python main_test.py
```
Outputs: predictions, classification reports and confusion matrices in `../data/processed/results/23/test_results_df23/`.

Each step reuses artifacts produced by the previous one (processed dataframes, RNABERT embeddings, graph embeddings), so they should be run in order on a first run.

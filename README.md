# PriMPT: Prior-Informed Multi-Scale Paired-Token Representation Learning for CRISPR–Cas9 Off-Target Activity Classification
PriMPT is a prior-informed deep learning framework for CRISPR–Cas9 off-target activity classification. The model represents aligned sgRNA–DNA sequence pairs as multi-scale paired tokens, including 1-gram, 2-gram, and 3-gram guide–target paired tokens. Mechanism-related pair-prior descriptors are injected into token representations to improve the modeling of guide–target pairing status, mismatch chemistry, local mismatch topology, PAM-related context, and sequence-level mismatch burden.  
## Data Sources
Five public CRISPR/Cas9 guide–target activity datasets were used in this project to evaluate the classification performance of PriMPT, including K562, HEK293T, II4, II5, and CRISPOR.

The datasets were obtained from the supplementary data of the CRISPR/Cas9 benchmarking study by Zhang et al. in Briefings in Bioinformatics:
Zhang, G.; Luo, Y.; Dai, X.; Dai, Z. Benchmarking deep learning methods for predicting CRISPR/Cas9 sgRNA on- and off-target activities. Briefings in Bioinformatics 2023, 24(6), bbad333. DOI: 10.1093/bib/bbad333.
For detailed data descriptions, and citation information, see [`data/README.md`](data/README.md).
## Environment Requirements
The code was developed using Python and PyTorch. GPU acceleration is recommended for model training and Integrated Gradients-based interpretation.

A minimal environment can be installed with:
pip install -r requirements.txt
The recommended requirements.txt includes:
numpy==2.1.2
pandas==2.3.1
scipy==1.16.1
scikit-learn==1.7.2
PyYAML==6.0.2
torch==2.7.1+cu118
tqdm==4.67.1
matplotlib==3.10.7
openpyxl==3.1.5
tensorboard==2.20.0
The original experiments were conducted with PyTorch 2.7.1 and CUDA 11.8.
## Data Preprocessing
Data preprocessing is controlled by YAML files under:
configs/data/
The preprocessing pipeline performs:

1. Loading of raw CRISPR/Cas9 guide–target activity datasets.
2. Sequence normalization and uppercase conversion.
3. Sequence length checking.
4. Binary label validation.
5. Duplicate guide–target pair handling.
6. Guide-disjoint train/validation/test splitting.
7. External benchmark construction.

By default, duplicate guide–target pairs with conflicting labels are removed, while duplicate pairs with the same label are collapsed.
Run the following commands from the project root directory:

python scripts/prepare_data.py --config configs/split/II4.yaml
python scripts/prepare_data.py --config configs/split/II5.yaml
python scripts/prepare_data.py --config configs/split/HEK293T.yaml
python scripts/prepare_data.py --config configs/split/K562.yaml
python scripts/prepare_data.py --config configs/split/CRISPOR.yaml
python scripts/prepare_data.py --config configs/split/external_benchmark.yaml

The processed datasets are saved under:

data/processed/
## Model Overview
PriMPT represents each aligned guide–target sequence pair as multi-scale paired tokens:

1-gram paired tokens represent single aligned guide–target paired columns.
2-gram paired tokens represent adjacent paired-column relationships.
3-gram paired tokens represent short-range local mismatch configurations.

For each scale, PriMPT constructs scale-aligned pair-prior descriptors and injects them into paired-token embeddings. The encoded representations are processed by a Transformer encoder and a local CNN residual module, followed by a classification head for binary activity prediction.
The main implementation files are:
primpt/priors.py
primpt/datasets.py
primpt/model.py
primpt/training.py
primpt/experiments.py
## Model Training
Model training is controlled by:

configs/train/full_model.yaml

This configuration file controls:

selected datasets;
fold indices;
random seeds;
CUDA device;
model hyperparameters;
batch size;
learning rate;
weight decay;
early stopping;
checkpoint and result output paths.

Run training with:

python scripts/train.py --config configs/train/train_model.yaml
The main evaluation metric is AUPRC. AUROC and other classification metrics are also reported.

Training outputs are saved under:

results/full_model/

## Model Interpretation
PriMPT provides Integrated Gradients-based interpretation for pair-prior descriptors.

The interpretation pipeline computes attribution with respect to pair-prior tensors only, while keeping paired-token identities fixed. The attribution target is the active-class logit. This allows the analysis to identify which prior descriptors contribute positively or negatively to predicted cleavage activity.

The interpretation configuration file is:

configs/explain/activity_relevance.yaml

Run interpretation with:

python scripts/explain.py --config configs/explain/activity_relevance.yaml

The final interpretation results are saved under:

results/explain/activity_relevance_development_macro_average/

The main output files are:

global_activity_relevance.csv
local_activity_relevance.csv
local_position_activity_relevance.csv

## Citation

If you use this repository, please cite the PriMPT study:
A full citation will be added after publication.









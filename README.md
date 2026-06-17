# PriMPT: Prior-Informed Multi-Scale Paired-Token Representation Learning for CRISPR–Cas9 Off-Target Activity Classification
PriMPT is a prior-informed deep learning framework for CRISPR–Cas9 off-target activity classification. The model represents aligned sgRNA–DNA sequence pairs as multi-scale paired tokens, including 1-gram, 2-gram, and 3-gram guide–target paired tokens. Mechanism-related pair-prior descriptors are injected into token representations to improve the modeling of guide–target pairing status, mismatch chemistry, local mismatch topology, PAM-related context, and sequence-level mismatch burden.  
## Data Sources
Five public CRISPR/Cas9 guide–target activity datasets were used in this project to evaluate the classification performance of PriMPT, including K562, HEK293T, II4, II5, and CRISPOR.

The datasets were obtained from the supplementary data of the CRISPR/Cas9 benchmarking study by Zhang et al. in Briefings in Bioinformatics:
Zhang, G.; Luo, Y.; Dai, X.; Dai, Z. Benchmarking deep learning methods for predicting CRISPR/Cas9 sgRNA on- and off-target activities. Briefings in Bioinformatics 2023, 24(6), bbad333. DOI: 10.1093/bib/bbad333.
For detailed data descriptions, and citation information, see [`data/README.md`](data/README.md).

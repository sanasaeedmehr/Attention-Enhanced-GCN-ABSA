# The Impact of Attention Mechanisms on Aspect‑Based Sentiment Analysis Performance Using Affective‑Knowledge‑Enhanced GCNs  

---

## 1. Introduction  

Aspect‑Based Sentiment Analysis (ABSA) is the task of detecting the sentiment polarity (positive, negative, or neutral) expressed toward a specific aspect within a text. This repository implements the “Proposed Model” from the paper **“The Impact of Attention Mechanisms on Aspect‑Based Sentiment Analysis Performance Using Affective‑Knowledge‑Enhanced GCNs”**.  

Key contributions:  

| # | Feature | Description |
|---|---------|-------------|
| 1 | **Attention‑Enhanced GCN** | Graph Convolutional Network that incorporates both dependency and sentic (affective) graphs with multi‑head attention. |
| 2 | **Aspect‑aware BERT integration** | The model can be used with or without BERT embeddings; BERT‑based variants use the same attention & GCN pipeline. |
| 3 | **Modular data pipeline** | Separate scripts for dependency graph, sentic graph, and combined graphs. |
| 4 | **Extensible training framework** | Supports numerous baseline models (CNN, LSTM, Bi‑LSTM‑GCN, etc.) and hyper‑parameter sweeps. |

---

## 2. Installation  

### 2.1 Clone the repo  

```bash
git clone https://github.com/sanasaeedmehr/Attention-Enhanced-GCN-ABSA.git
cd AE‑GCN
```

### 2.2 Create a virtual environment (recommended)  

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 2.3 Install Python dependencies  


```bash
pip install -r requirements.txt
```

| Library | Version |
|---------|---------|
| torch | 2.7.0+cu128 |
| transformers | 4.57.1 |
| numpy | 2.2.4 |
| pandas | 2.2.3 |
| scikit-learn | 1.6.1 |
| scipy | 1.15.2 |
| matplotlib | 3.10.1 |
| tqdm | 4.67.1 |
| spacy | 3.8.14 *(install separately, see below)* |
| spacy‑model‑en | 3.7.4 |

> **Optional** – for GPU acceleration you may install a CUDA‑enabled CUDA toolkit matching your PyTorch version.  

### 2.4 Install GloVe embeddings  

```bash
wget https://nlp.stanford.edu/data/wordvecs/glove.42B.300d.zip
unzip glove.42B.300d.zip
# Move the file into the repo root
mv glove.42B.300d.txt .
```

### 2.5 Install spaCy and English model  

For tokenization, lemmatization, and dependency parsing, the spaCy library is utilized.

```bash
python -m spacy download en_core_web_sm==3.7.4
```

### 2.6 Verify installation  

```bash
python -c "import torch, transformers, spacy; print('All good!')"
```

---

## 3. Datasets

The repository supports the following benchmark datasets:  

| Dataset | Link | 
|---------|------|
| SemEval‑2014 Task 4:|https://alt.qcri.org/semeval2014/task4/|
|SemEval‑2015 Task 12:|https://alt.qcri.org/semeval2015/task12/|
|SemEval‑2016 Task 5:|https://alt.qcri.org/semeval2016/task5/|



**Generating graphs**  

Run the scripts in the following order:

```bash
python generate_dependency_graph.py
python generate_sentic_graph.py
python generate_sentic_dependency_graph.py
```


---

## 4. Training  

### 4.1 Non‑BERT Models  

```bash
python train.py --model_name=lstm --dataset=acl-14 --epochs=20 --device=cpu --lr=0.001 --batch_size=64
```

Available `--model_name` options (see `train.py`):

- `lstm`, `bi_lstm`, `cnn`, `gcn`, `gcn_attention`, etc.

### 4.2 BERT‑based Models  

```bash
python train_bert.py --model_name=bi_lstm_gcn_att_sentic_aspect_bert --dataset=rest14 --seed=42 --epochs=10 --device=cuda
```

Available `--model_name` options (see `train_bert.py`):

- `lstm_bert`, `bi_lstm_bert`, `cnn_bert`, `gcn_bert`, `bi_lstm_gcn_att_sentic_aspect_bert`, etc.

### 4.3 Hyper‑parameter Tuning  

The training scripts accept standard flags (`--lr`, `--batch_size`, `--epochs`, `--kernel_size`, `--dropout_rate`, `--num_filters`, etc.). Adjust them according to your GPU memory or desired accuracy.

---


## 5. Random seeds used in the experiments:
[42, 100, 101, 123, 2021, 2022, 2023, 2024, 2025, 2026]

## 6. Dataset Integrity Verification

SHA256 checksums for all datasets and processed splits are provided in `CHECKSUMS.sha256`.

To verify integrity:

```bash
sha256sum -c CHECKSUMS.sha256


## 7. Results & Discussion  

The *Proposed Model* (attention‑enhanced GCN with sentic knowledge) results:

| Dataset | Lap14 | Rest14 | Rest15 | Rest16 |
|---------|-------|--------|--------|--------|
| Acc. | 81.09% | 87.34% | 85.06% | 92.21% |
| F1 | 77.52% | 81.36% | 73.05% | 79.43% |

---

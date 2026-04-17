# NLP Assignment 2

This repository contains the notebook and supporting data for the NLP Assignment 2 pipeline:

- Part 1: word embeddings (TF-IDF, PPMI, Skip-gram Word2Vec)
- Part 2: sequence labeling (POS tagging and NER with BiLSTM/CRF)
- Part 3: transformer encoder for topic classification

## Repository Layout

```text
.
├── ASSIGNMENT_REPORT.md
├── i23_Assignment2_NLP.ipynb
├── data/
├── embeddings/
├── models/
├── ner_train.conll
├── ner_test.conll
├── pos_train.conll
├── pos_test.conll
```

## Requirements

Use Python 3.10 or newer. The notebook depends on the following packages:

- numpy
- pandas
- torch
- scikit-learn
- matplotlib
- seaborn
- tqdm

If you are working on macOS, the notebook will run on CPU unless PyTorch detects a compatible GPU backend.

## Environment Setup

Create and activate a virtual environment, then install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy pandas torch scikit-learn matplotlib seaborn tqdm jupyter
```

If you prefer Conda, create an equivalent environment and install the same packages.

## How to Reproduce

1. Open `i23_Assignment2_NLP.ipynb` in Jupyter Notebook or VS Code.
2. Run the cells from top to bottom without skipping cells.
3. The notebook expects the dataset files and folders already present in the repository root.
4. Re-run the import/setup cells whenever you restart the kernel.

### Recommended run order

1. Data loading and preprocessing
2. Part 1: embedding experiments
3. Part 2: POS tagging and NER
4. Part 3: transformer classification
5. Evaluation and report-generation cells

## Input Data

The notebook uses the following files:

- `pos_train.conll`
- `pos_test.conll`
- `ner_train.conll`
- `ner_test.conll`

Preprocessed artifacts and intermediate files are stored in `data/` and `embeddings/`.

## Output Files

The notebook writes trained models and analysis outputs to these locations:

- `embeddings/`
  - `tfidf_matrix.npy`
  - `ppmi_matrix.npy`
  - `embeddings_w2v.npy`
  - `embeddings_w2v_c2_raw.npy`
  - `embeddings_w2v_c4_d200.npy`
  - `word2idx.json.npy`
- `models/`
  - `bilstm_pos.pt`
  - `bilstm_ner.pt`
  - `bilstm_ner_crf_best.pt`
  - `bilstm_pos_frozen_best.pt`
  - `bilstm_pos_finetuned_best.pt`
  - `transformer_cls.pt`

Figures and plots are also saved from the notebook cells into `embeddings/`.

## Expected Runtime

Runtime depends on hardware and whether model checkpoints already exist.

- With existing checkpoints: many cells finish quickly.
- Full run on CPU: expect several hours for the complete notebook.
- Full run on a faster machine or GPU: substantially less time.

The longest sections are the sequence labeling and transformer training cells.

## Notes

- Run the notebook in order so that variables and imports are defined before later cells use them.
- If you restart the kernel, re-run the import and preprocessing cells before evaluation cells.
- Some cells save plots and model weights automatically, so make sure the `embeddings/` and `models/` folders are writable.
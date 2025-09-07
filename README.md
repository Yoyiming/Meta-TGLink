# Meta-TGLink: Structure-Enhanced Graph Meta Learning for Few-Shot Gene Regulatory Network Inference

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-red.svg)](https://pytorch.org)

> **Meta-TGLink** is a novel structure-enhanced graph meta-learning framework designed for **few-shot gene regulatory network (GRN) inference**. By integrating prior biological network structures with meta-learning paradigms, our method enables rapid adaptation to new regulatory contexts with limited labeled data, advancing the robustness and generalizability of GRN prediction in low-data regimes.

---

## 📌 TODO List

A roadmap of current and future development goals:

- [x] Upload core Meta-TGLink model codes
- [x] Upload training and evaluating codes for three kinds of cross-domain few-shot GRN inference
- [x] Upload curated datasets and cell line (or cell type) expression matrix
- [ ] Write detailed documentation and tutorials for cell line data generation
- [ ] Release inference API for easy deployment

---

## 🛠 Dependencies Used in This Project

This project relies on the following core packages:

| Package         | Version   | Purpose |
|-----------------|-----------|--------|
| Python          | >= 3.8    | Core language |
| PyTorch         | >= 1.12   | Deep learning framework |
| PyTorch Geometric | >= 2.3.1| Graph neural networks |
| NumPy           | >= 1.24   | Numerical computing |

For a complete list, see `environments.yml`.

---

## 🐍 Set Up Conda Environment

We recommend using `conda` to manage dependencies. Follow these steps to set up your environment:

```bash
# Create a conda environment
conda env create -f environment.yml
```

---

## 🚀 Usage
### Project Structure

```text
Meta-TGLink/
├── cell_line_dataset/          # Training and Testing sets
├── expression_matrix/          # Expression matrix for cell line or cell type datasets
├── model.py                    # Meta-TGLink architecture
├── dataset.py                  # Meta Dataset for constructing meta-tasks
├── train_cross_cell_line.py    # Training codes for cross-cell line GRN inference
├── test_cross_cell_line.py     # Evaluating codes for cross-cell line GRN inference (also including benchmark GRN inference)
├── train_test_cross_tf.py      # Training and Evaluating codes for cross-TF GRN inference (including cold-start inference)
├── train_cross_species.py      # Training codes for cross-species GRN inference
├── test_cross_species.py       # Evaluating codes for cross-species GRN inference
├── utils.py                    # Helper functions and evaluation metrics
└── train.py                    # Main training script
```

### Split Datasets

We provide three dataset splitting scripts for different scenarios:

- `limma_dataset_generate.py` — For human cell line datasets  
- `specific_dataset_generate.py` — For mouse cell-type-specific datasets  
- `tf_dataset_split.py` — For human cell line cross-TF GRN inference (ensures **no overlapping TFs** among train/validation/test sets)

Run each script to generate the corresponding dataset splits.


### Training

To train Meta-TGLink for cross-cell line GRN inference or benchmark GRN inference using default configuration:

```bash
python train_cross_cell_line.py --train_cell PC3
```

Here, PC3 represents the source cell line. You can replace this with other source cell lines as needed (e.g. A375, A549).

💡 You can train Meta-TGLink to your own datasets by modifying the data paths in the script.

### Evaluating

To evaluate Meta-TGLink for cross-cell line GRN inference or benchmark GRN inference:

```bash
python test_cross_cell_line.py --train_cell PC3
```

💡 Also you can use your dataset and exchange configs in codes.

> 🔄 For other cross-domain GRN inference tasks (e.g., cross-TF, cross-species), usage follows the same pattern — just replace the scripts.


## 📮 Contact
For any inquiries, feel free to raise issues or contact me via email at yoyiming7@gmail.com.

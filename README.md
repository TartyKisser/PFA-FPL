# Multi-Label Few-Shot Image Classification via Pairwise Feature Augmentation and Flexible Prompt Learning

This repository contains the unofficial PyTorch implementation for [**PFA+FPL**](https://ojs.aaai.org/index.php/AAAI/article/view/32578/34733), a framework designed for **Multi-label Few-shot Learning**.

The method leverages **CLIP (Contrastive Language-Image Pre-training)** as a powerful visual backbone and introduces a **Pairwise Feature Augmentation (PFA)** mechanism to enrich the support set. It combines image prototypes with text prototypes to enhance classification performance in low-data regimes.

## 📌 Key Features

* **CLIP Backbone**: Utilizes a pre-trained CLIP (ResNet50) model to extract robust visual and textual features, leveraging its zero-shot transfer capabilities.
* **Pairwise Feature Augmentation (PFA)**: Implements a mechanism to concatenate support set image features in pairs, generating synthetic samples and union labels to alleviate data sparsity.
* **Dual Prototype Fusion**: Dynamically fuses **Image Prototypes** (visual features) and **Text Prototypes** (label semantics) using learnable alpha parameters.
* **Multi-Task Loss Strategy**: The training optimizes a weighted combination of multiple losses:
    * **BCE Loss**: Standard classification loss.
    * **Symmetry Loss**: Ensures consistency in feature concatenation order (A+B vs. B+A).
    * **Count Loss**: Predicts the number of objects/labels in an image.
    * **Uncertainty Weighting**: Uses learnable `sigma` parameters to balance loss components automatically.

## 🛠️ Requirements

Ensure your environment meets the following dependencies:

* Python >= 3.8
* PyTorch >= 1.8.0 (CUDA support recommended)
* Torchvision
* NumPy
* SciPy
* Pillow
* [CLIP](https://github.com/openai/CLIP)

To install CLIP:
```bash
pip install git+[https://github.com/openai/CLIP.git](https://github.com/openai/CLIP.git)
```

## 📂 Data Preparation

This project supports **COCO2014** and **Visual Genome (VG)** datasets.

### 1. Directory Structure

Please organize your data directory as follows:

```text
Project_Root/
├── data/
│   ├── COCO2014.json      # Metadata containing label text
│   ├── COCO2014_idx/      # Pre-processed .npy and .npz index files
│   │   ├── COCO2014_train_images.npy
│   │   ├── COCO2014_train_labels.npz
│   │   └── ...
│   ├── VG.json
│   ├── VG_idx/
│   └── readme.md
├── dataset/               # Raw image storage
│   ├── COCO2014/
│   │   ├── COCO_train2014_000000000009.jpg
│   │   └── ...
│   └── VG/
└── ...
```

> **Note**: The image paths are defined in `data/dataset.py` (default points to `../dataset/COCO2014`). Please adjust the `path_to_images_dict` variable if your images are stored elsewhere. The path to the dataset split JSON file (e.g., `data/COCO2014.json`) is hardcoded in `run.py`. You must manually update the file path in the `__main__` block of `run.py` if you are using a different metadata file.

### 2. Index File Description

The project uses pre-processed files for efficient data loading:

* **`images.npy`**: A Numpy array containing the filenames of the images.
* **`labels.npz`**: A CSR sparse matrix storing the one-hot encoded labels for the corresponding images.

## 🚀 Usage

### Training & Testing

Use `run.py` to train and evaluate the model. The script automatically loads the pre-trained CLIP model.

**Basic Command:**

```bash
python run.py --dataset_name COCO2014 --n_way 10 --n_shot 1 --device cuda:0
```

### Arguments

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--dataset_name` | str | `VG` | Dataset name: `['COCO2014', 'CUB', 'NUS', 'VG']` |
| `--n_way` | int | `10` | N-way: Number of classes per episode |
| `--n_shot` | int | `1` | K-shot: Number of support samples per class |
| `--n_query` | int | `auto` | Number of query samples per class (defaults to `n_way // 2` in code) |
| `--max_epoch` | int | `500` | Maximum number of training epochs |
| `--hidden_dim` | int | `100` | Hidden dimension size (used in specific modules) |
| `--num_workers` | int | `8` | Number of DataLoader workers |
| `--seed` | int | `0` | Random seed for reproducibility |

### Example Configuration

**Run 10-way 1-shot training on COCO2014:**

```bash
python run.py \
  --dataset_name COCO2014 \
  --n_way 10 \
  --n_shot 1 \
  --max_epoch 500 \
  --num_workers 8 \
  --seed 0
```

## 🧠 Code Structure

* **`run.py`**: Entry point. Handles argument parsing, DataLoader construction, training loops (`_train`), and testing loops (`_test`). Saves the best model based on mAP.
* **`methods/model.py`**: Core model definition.
* `Model`: Main class containing the CLIP encoder, MLP layers, and loss logic.
* `create_pairwise_samples`: Implements the feature concatenation logic.
* `set_forward_loss`: Computes the combined loss (Symmetry + CE + Count).


* **`data/dataset.py`**: Data loader implementation.
* `MLLDataset`: Standard dataset class.
* `EpisodeSampler`: Sampler for Meta-Learning episodes (N-way K-shot).
* `RandAugment`: Custom augmentation implementation.



## ⚠️ Notes

1. **HuggingFace Mirror**: The `run.py` sets `os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"` to facilitate model downloads in regions with restricted access. Comment this out if not needed.
2. **GPU Memory**: Ensure sufficient GPU memory as the model uses ResNet50-CLIP and expands batch sizes via pairwise feature augmentation.
3. **Frozen Parameters**: The CLIP parameters are frozen (`param.requires_grad = False`) by default; only the subsequent MLP and adaptation layers are trained.

## 📝 Citation

If you use this code in your research, please cite the relevant paper.
```bash
@inproceedings{liu2025multi,
  title={Multi-Label Few-Shot Image Classification via Pairwise Feature Augmentation and Flexible Prompt Learning},
  author={Liu, Han and Wang, Yuanyuan and Zhang, Xiaotong and Zhang, Feng and Wang, Wei and Ma, Fenglong and Yu, Hong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={5},
  pages={5433--5441},
  year={2025}
}
```

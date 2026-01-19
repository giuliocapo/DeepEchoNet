
# DeepEchoNet: A Lightweight Architecture for Extreme Low-Resolution Monocular Depth Estimation

This repository contains the official implementation of **DeepEchoNet**, a lightweight hybrid CNN–Transformer model for **monocular depth estimation (MDE)** in an **extreme low-resolution** regime.  
The project investigates accuracy–efficiency trade-offs for depth-aware perception on resource-constrained platforms and reproduces the experimental protocol and results reported in the associated thesis/paper.

---

## Features
- End-to-end supervised training on **NYU Depth v2**
- Support for **96×96** input resolution
- Training pipeline with a **strong-to-weak augmentation schedule**
- Evaluation on the standard NYU Depth v2 test split with common MDE metrics (e.g., AbsRel, RMSE, δ-threshold accuracies)
- Inference-time benchmarking (single-image throughput)

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/giuliocapo/DeepEchoNet
   cd DeepEchoNet
   ```


2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## Dataset: NYU Depth v2

This project uses the **NYU Depth v2** indoor RGB-D dataset.

* Official dataset page:
  [https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)

### Train/Test CSV Splits

The exact image lists used for training and testing are provided as CSV files (RGB/depth paths).
Download links:

* **CSV splits (train/test)**: `https://drive.google.com/drive/folders/1jH8qPoz0fUu2VeNzkXPUVOKliGhADL2r?usp=sharing`

> Note: You are responsible for obtaining NYU Depth v2 from the official sources and ensuring compliance with the dataset license/terms.

---

## Pretrained Weights

Pretrained model weights for the main DeepEchoNet configurations can be downloaded here:

* **Pretrained weights**: `https://drive.google.com/drive/folders/1jH8qPoz0fUu2VeNzkXPUVOKliGhADL2r?usp=sharing`

After downloading, place the checkpoints in a local directory of your choice (e.g., `checkpoints/`) and reference that path when running evaluation.

---

## Usage

### Evaluation

Evaluate a trained checkpoint on the NYU Depth v2 test set using the **nyu2_test** image set:

- **nyu2_test (images)**: https://drive.google.com/drive/folders/1jH8qPoz0fUu2VeNzkXPUVOKliGhADL2r?usp=sharing

```bash
python test.py
```

## Reproducibility Notes

* Experiments were conducted using a consistent hardware/software stack (PyTorch, single GPU).
* Inference-time measurements are computed with **batch size = 1** and include warm-up iterations before timing.

---

## Citation

If you use this code in academic work, please cite the associated thesis/paper.
This is a provisional citation entry. A final BibTeX entry will be provided once the thesis/paper is officially published.

```bibtex
@misc{deepechonet2026,
  title        = {DeepEchoNet: A Lightweight Architecture for Extreme Low-Resolution Monocular Depth Estimation},
  author       = {Giulio Caporro and Paolo Russo},
  year         = {2026},
  note         = {Code available at: https://github.com/giuliocapo/DeepEchoNet}
}
```

---

## License

Specify the license for this repository (e.g., MIT, Apache-2.0) in `LICENSE`.
If you are using NYU Depth v2 data, ensure compliance with its original terms.

---

## Acknowledgements

This work builds on prior research in monocular depth estimation and mobile vision transformers, including NYU Depth v2 and MobileViT-based baselines.

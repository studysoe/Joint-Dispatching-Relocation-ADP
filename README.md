# Joint Dispatching and Relocation with ADP

This repository contains the implementation for the paper:

**Approximate Dynamic Programming for Joint Dispatching and Relocation in Heterogeneous Ride-Hailing Systems**

---

## 📌 Overview

This project studies the joint dispatching and relocation problem in ride-hailing systems with heterogeneous vehicles. We propose an Approximate Dynamic Programming (ADP) framework under a Non-Homogeneous Poisson Process (NHPP) demand model and compare it with a Fluid Approximation baseline under a Homogeneous Poisson Process (HPP).

The repository includes:

* Implementation of the proposed **ADP-NHPP** method
* Implementation of the **FA-HPP** baseline
* Training and testing pipelines
* Pretrained neural network models
* Data processing and visualization scripts

---

## ⚙️ Installation

All dependencies are specified in `pyproject.toml`.

We recommend using **uv** for environment setup:

```bash
uv sync
```

Alternatively, you may install dependencies manually using:

```bash
pip install -r requirements.txt
```

---

## 📁 Repository Structure

```
.
├── adp_nhpp/                     # ADP-NHPP implementation
│   ├── train_*.py               # Training scripts
│   ├── test_*.py                # Testing scripts
│
├── fluid_approximation_hpp/     # FA-HPP baseline implementation
│   ├── train_*.py
│   ├── test_*.py
│
├── data/                        # Dataset and preprocessing scripts
├── figure/                      # Plotting scripts
│
├── *.pth                        # Trained models and checkpoints
├── pyproject.toml               # Dependency configuration
└── README.md
```

---

## 🚀 Methods

### 1. ADP-NHPP (Proposed Method)

* Located in: `adp_nhpp/`
* Based on Approximate Dynamic Programming with neural network value function approximation
* Demand modeled using Non-Homogeneous Poisson Process (NHPP)
* Includes:

  * Training scripts
  * Testing scripts

---

### 2. FA-HPP (Baseline)

* Located in: `fluid_approximation_hpp/`
* Fluid approximation under Homogeneous Poisson Process (HPP)
* Used as a benchmark for comparison

---

## 🧠 Pretrained Models

The repository provides pretrained neural networks for **ADP-NHPP testing**.

### Morning Period (6–8)

#### Different service ratios (α)

* `adp_6_8_model_alpha0.6_20260327_174817.pth`
* `adp_6_8_model_alpha0.7_20260327_190328.pth`
* `adp_6_8_model_alpha0.8_20260327_201554.pth`
* `adp_6_8_model_alpha0.9_20260327_212641.pth`

#### Different fleet sizes (N)

* `adp_6_8_model_N50_20260327_155831.pth`
* `adp_6_8_model_N150_20260327_185808.pth`
* `adp_6_8_model_N200_20260327_210825.pth`
* `adp_6_8_model_N250_20260327_232519.pth`
* `adp_6_8_model_N300_20260328_013412.pth`

---

### Evening Period (17–19)

#### Different fleet sizes (N)

* `adp_17_19_model_N250_20260326_225629.pth`
* `adp_17_19_model_N300_20260327_001919.pth`
* `adp_17_19_model_N350_20260327_014900.pth`
* `adp_17_19_model_N400_20260327_032508.pth`
* `adp_17_19_model_N450_20260327_050537.pth`
* `adp_17_19_model_N500_20260327_065043.pth`

#### Different service ratios

* `adp_17_19_model_ratio_0_6_20260327_182112.pth`
* `adp_17_19_model_ratio_0_7_20260327_204230.pth`
* `adp_17_19_model_ratio_0_8_20260327_225158.pth`
* `adp_17_19_model_ratio_0_9_20260328_004957.pth`

---

## 💾 Training Checkpoints

Intermediate checkpoints saved during training:

* `checkpoint_ep10.pth`
* `checkpoint_ep20.pth`
* ...
* `checkpoint_ep100.pth`

These can be used for:

* Resuming training
* Monitoring convergence behavior

---

## 📊 Data

The `data/` directory contains:

* Raw datasets used in the study
* Data preprocessing scripts

---

## 📈 Visualization

The `figure/` directory includes scripts for:

* Generating plots in the paper
* Performance comparison visualization

---

## 🧪 Usage

### Train ADP-NHPP

```bash
python adp_nhpp/train_*.py
```

### Test ADP-NHPP

```bash
python adp_nhpp/test_*.py
```

### Run FA-HPP baseline

```bash
python fluid_approximation_hpp/test_*.py
```

---

## 📌 Notes

* Large files such as logs and `wandb/` outputs are excluded via `.gitignore`
* Pretrained models are provided for reproducibility
* Experiments cover both **morning peak (6–8)** and **evening peak (17–19)** scenarios

---

## 📜 License

This project is intended for academic research purposes.

If you use this code, please cite the corresponding paper.

---

## ✉️ Contact

For questions or collaborations, please open an issue on GitHub.

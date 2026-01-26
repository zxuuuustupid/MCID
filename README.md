# Collaborative Teacher-Student Learning (CTSL)

## Overview

This repository contains the official implementation of **Collaborative Teacher-Student Learning (CTSL)** (also referred to as **MCID** in the codebase), a novel framework for Multi-Domain Generalized Fault Diagnosis.

**Paper Title:** Collaborative Teacher-Student Learning (CTSL): Simulated Domain Attacks for Class-Intrinsic Feature Learning in Multi-Domain Generalized Fault Diagnosis

Existing data-driven fault diagnosis methods often suffer from performance degradation under varying operating conditions due to domain shift. While Domain Generalization (DG) attempts to mitigate this via distribution alignment, such approaches can damage the intrinsic data manifold structure.

**CTSL** addresses these challenges by reframing condition variations as adversarial **"Simulated Domain Attacks"**. Instead of passive alignment, CTSL employs an active defense mechanism within a Teacher-Student paradigm to extract **Class-Intrinsic Features** that remain invariant across domains.

### Key Contributions

- **Simulated Domain Attacks**: A new perspective that models complex condition fluctuations as attacks, enhancing robustness through active defense.
- **Collaborative Framework**: leverages a frozen teacher network to provide stable manifold priors to a student encoder, avoiding the structural damage often caused by forced feature alignment.
- **Multi-Consistency Distillation**: Introduces a protocol enforcing Domain Consistency (DC), Cycle Consistency (CC), and Label Consistency (LC) to ensure the learned features are both domain-invariant and distinctive.

## Repository Structure

```
.
├── configs/      # Configuration files for Teacher and CTSL (MCID) training
├── data/         # Dataset directory (organized by working conditions)
├── ckpts/        # Checkpoints for models and training artifacts
├── scripts/      # Scripts for training, evaluation, and visualization
├── src/          # Source code including models, dataloaders, and core logic
└── README.md
```

## Getting Started

### Prerequisites

Ensure you have the necessary dependencies installed:

```bash
pip install -r requirements.txt
```

### Step 1: Train the Teacher Network

The teacher network is pre-trained on a benign source working condition (e.g., WC1) to establish a stable feature manifold.

```bash
# Example: Train teacher on Paderborn University (PU) dataset, Single Condition
python scripts/train_teacher.py --config configs/teacher_<dataset>_train_<wc>.yaml
```

### Step 2: Collaborative Student Learning (CTSL/MCID)

Train the student network using the CTSL framework. This process involves the two-stage simulated optimization strategy to generalize to unseen target domains.

_Note: The codebase uses `mcid` prefixes for configuration files corresponding to the CTSL method._

```bash
# Example: Train CTSL on PU dataset, Source: Included in config, Targets: Meta-learning targets
python scripts/train_main.py --config configs/mcid_<dataset>_train_<source>_meta_<targets>.yaml
```

### Visualization

To visualize the feature distributions and compare the Teacher vs. Student (CTSL) performance:

```bash
python experiments/compare_PU_tsne.py --save_name compare_results
```

## Citation

If you find this code or method useful for your research, please cite our work:

```bibtex
@article{ctsl2026,
  title={Collaborative Teacher-Student Learning (CTSL): Simulated Domain Attacks for Class-Intrinsic Feature Learning in Multi-Domain Generalized Fault Diagnosis},
  author={Zhixu Duan and Zuoyi Chen},
  journal={IEEE Transactions on Industrial Informatics},
  year={2026}
}
```

## License

This project is licensed for academic research purposes.

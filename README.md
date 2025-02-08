# A Graph Bottleneck and Masked Feature Interaction Framework for Multi-Objective Optimization in E-commerce Recommendation Systems

**Paper**: [ACM MSCE 2025](https://dl.acm.org/doi/10.1145/3760023.3760094) | Published September 2025

## Abstract

This study aims to enhance the accuracy of multi-objective prediction in e-commerce recommendation systems, specifically focusing on user behaviors such as clicks, cart additions, and purchases. Utilizing the OTTO dataset, we developed a recommendation system that integrates advanced feature engineering, the ESM2 framework for multi-task learning, and MaskNet for capturing high-order feature interactions. A power-weighted fusion mechanism combines click-through rate and conversion rate predictions for superior performance.

## Dataset

[OTTO Multi-Objective Recommender System](https://www.kaggle.com/competitions/otto-recommender-system) — Real-world e-commerce session data with clicks, carts, and orders.

## Methods

- **ESM2 (Embedding Shared Multi-task Model)** — Shared embedding layer with task-specific towers for concurrent multi-objective prediction
- **MaskNet** — Dynamic feature masking for high-order feature interactions
- **FM Feature Cross** — Factorization Machines for pairwise feature interactions
- **Power-Weighted Fusion** — Learned combination of CTR and CTCVR predictions
- **LightGBM Recall** — Fast candidate retrieval with gradient boosting

## Repository Structure

```
├── lightgbm-fast-recall-20.ipynb          # LightGBM-based fast recall (top-20 candidates)
├── EasyRec/                               # Multi-task recommendation framework
│   ├── model/                             # Model implementations (ESM2, DBMTL, DeepFM, etc.)
│   ├── layers/                            # Neural network layers
│   ├── configs/                           # Sample model configurations
│   ├── README.md                          # EasyRec documentation
│   └── LICENSE                            # Apache 2.0
├── logs/                                  # Training logs
└── README.md
```

## Results

| Model | AUC | F1-score | NDCG |
|-------|-----|----------|------|
| LightGBM | 0.728 | 0.671 | 0.5123 |
| ESM2 | 0.734 | 0.682 | 0.5015 |
| RecBole GRU4Rec | 0.751 | 0.694 | 0.5191 |
| FM + ESM2 | 0.802 | 0.741 | 0.5789 |
| **FM + ESM2 + MaskBlock** | **0.892** | **0.792** | **0.6692** |

## Citation

```bibtex
@inproceedings{jin2025graph,
  title={A Graph Bottleneck and Masked Feature Interaction Framework for Multi-Objective Optimization in E-commerce Recommendation Systems},
  author={Jin, Tian},
  booktitle={Proceedings of the 2025 International Conference on Management Science and Computer Engineering (MSCE)},
  pages={429--432},
  year={2025},
  organization={ACM},
  doi={10.1145/3760023.3760094}
}
```

## Requirements

```
numpy
pandas
scikit-learn
lightgbm
tensorflow>=2.3
```

## Acknowledgments

This project uses [EasyRec](https://github.com/alibaba/EasyRec), an open-source recommendation framework by Alibaba.

## License

MIT (project code) | Apache 2.0 (EasyRec framework)

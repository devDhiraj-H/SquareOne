# Physics-Informed Kinematic & Aerodynamic Consistency Benchmarks

| Model                                 | Methodology                                   |   FDI AUC (%) |   FDI Recall at 5% FAR (%) |   Overall Anomaly AUC (%) | Macro F1 (%)         |   Latency (us/sample) |
|:--------------------------------------|:----------------------------------------------|--------------:|---------------------------:|--------------------------:|:---------------------|----------------------:|
| Pure Chi-Square Physics Detector      | Mahalanobis Kinematic Invariant (Zero ML)     |         99.31 |                      99.88 |                     86.62 | N/A (Binary Anomaly) |                  0.85 |
| Physics-Informed XGBoost (PI-XGBoost) | Tree Boosting + 9 Kinematic Residuals         |         99.98 |                      99.59 |                     95.8  | 76.36                |                131.06 |
| Physics-Informed Multimodal XGBoost   | Unified Cyber Packets + Aerodynamic Residuals |        100    |                     100    |                     99.1  | 87.69                |                 79.95 |


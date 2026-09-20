# Hardened Benchmark: Performance Without Artificial Crash Codes

| Suite                         | Model               | Domain   |   Accuracy (%) |   Macro F1 (%) |   F1: Benign (%) |   F1: DoS (%) |   F1: Evil_Twin (%) |   F1: FDI (%) |   F1: Replay (%) |
|:------------------------------|:--------------------|:---------|---------------:|---------------:|-----------------:|--------------:|--------------------:|--------------:|-----------------:|
| Hardened (No Crash Sentinels) | Physical XGBoost    | Physical |          47.42 |          55.01 |            15.47 |         43.65 |               55.62 |         99.79 |            60.53 |
| Hardened (No Crash Sentinels) | Multimodal XGBoost  | Fused    |          94.6  |          88.17 |            95.77 |         75.94 |              100    |        100    |            69.14 |
| Hardened (No Crash Sentinels) | Multimodal LightGBM | Fused    |          94.36 |          87.46 |            95.83 |         74.69 |              100    |        100    |            66.8  |


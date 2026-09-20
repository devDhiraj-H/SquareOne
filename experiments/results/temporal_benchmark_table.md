# Temporal Sequence Modeling Benchmark: Overcoming DoS & Replay Bottlenecks

| Model                                | Paradigm                              |   Accuracy (%) |   Macro F1 (%) |   Latency (us/sample) |   Model Size (KB) |   F1: Benign (%) |   F1: DoS (%) |   F1: Evil_Twin (%) |   F1: FDI (%) |   F1: Replay (%) |
|:-------------------------------------|:--------------------------------------|---------------:|---------------:|----------------------:|------------------:|-----------------:|--------------:|--------------------:|--------------:|-----------------:|
| Static Multimodal XGBoost (Baseline) | Static Tabular (1 Row)                |          94.41 |          88    |                135.89 |               980 |            95.49 |         75.84 |               99.97 |           100 |            68.73 |
| Temporal Rolling XGBoost             | Temporal Rolling (W=10 Stats)         |          98.16 |          95.41 |                147    |              1850 |            99.84 |         88.93 |               99.91 |           100 |            88.35 |
| Temporal 1D-CNN                      | Deep Convolutional Window (W=10)      |          97.09 |          92.41 |                 41.75 |               120 |            99.96 |         83.18 |              100    |           100 |            78.92 |
| Temporal GRU                         | Recurrent Gated Unit (W=10)           |          97.25 |          92.92 |                 69.68 |                95 |            99.96 |         83    |              100    |           100 |            81.64 |
| Hybrid 1D-CNN + GRU                  | Convolutional-Recurrent Hybrid (W=10) |          97.44 |          93.42 |                 76.54 |               135 |            99.92 |         83.9  |              100    |           100 |            83.28 |


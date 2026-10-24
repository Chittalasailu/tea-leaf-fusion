# Analysis numbers

Seeds: [42, 7, 123]

## Pairwise diversity on teaLeafBD test (mean over seeds)

| pair                |   disagreement |   double_fault |   yule_q |   error_correlation |
|:--------------------|---------------:|---------------:|---------:|--------------------:|
| CNN / DenseNet121   |         0.1326 |         0.0312 |   0.7696 |              0.2735 |
| CNN / VGG16         |         0.1657 |         0.041  |   0.6533 |              0.2378 |
| DenseNet121 / VGG16 |         0.107  |         0.0385 |   0.8765 |              0.3845 |

## Fusion vs oracle on teaLeafBD test (mean over seeds)

| config                    |   accuracy |   oracle_accuracy |   errors_fixed_vs_best_single |   errors_added_vs_best_single |
|:--------------------------|-----------:|------------------:|------------------------------:|------------------------------:|
| CNN                       |     0.8706 |            0.8706 |                       36.3333 |                      103.667  |
| DenseNet121               |     0.9343 |            0.9343 |                        0      |                        0      |
| VGG16                     |     0.8816 |            0.8816 |                       28.6667 |                       84.3333 |
| CNN + DenseNet121         |     0.9413 |            0.9688 |                       12.6667 |                        5.3333 |
| CNN + VGG16               |     0.922  |            0.959  |                       32.3333 |                       45.3333 |
| DenseNet121 + VGG16       |     0.9343 |            0.9615 |                       12.3333 |                       12.3333 |
| CNN + DenseNet121 + VGG16 |     0.9451 |            0.9782 |                       20.6667 |                        9.3333 |

## Cross-dataset (trained on teaLeafBD, tested on TeaLeafDiseaseBD, no retraining)

Images: 10954; shared classes: ['1. Tea algal leaf spot', '2. Brown Blight', '3. Gray Blight', '4. Helopeltis', '5. Red spider', '7. Healthy leaf']; skipped: {'Red Rust': 2131}

| config                    |   ('external_accuracy', 'mean') |   ('external_accuracy', 'std') |   ('external_macro_f1', 'mean') |   ('external_macro_f1', 'std') |   ('teaLeafBD_accuracy', 'mean') |   ('teaLeafBD_accuracy', 'std') |
|:--------------------------|--------------------------------:|-------------------------------:|--------------------------------:|-------------------------------:|---------------------------------:|--------------------------------:|
| CNN                       |                           40.96 |                           2.84 |                           36.4  |                           2.53 |                            87.06 |                            1.06 |
| DenseNet121               |                           53.17 |                           0.64 |                           50.2  |                           0.8  |                            93.43 |                            0.29 |
| VGG16                     |                           43.55 |                           0.87 |                           41.08 |                           1.31 |                            88.16 |                            0.53 |
| CNN + DenseNet121         |                           53.82 |                           0.63 |                           50.24 |                           1.02 |                            94.13 |                            0.28 |
| CNN + VGG16               |                           46.48 |                           0.73 |                           43.36 |                           1.33 |                            92.2  |                            0.44 |
| DenseNet121 + VGG16       |                           52.09 |                           0.26 |                           49.57 |                           0.55 |                            93.43 |                            0.54 |
| CNN + DenseNet121 + VGG16 |                           52.75 |                           0.51 |                           49.84 |                           0.54 |                            94.51 |                            0.78 |

Per-class recall of the triple ensemble (first seed):

| class                  |   recall_triple |    n |
|:-----------------------|----------------:|-----:|
| 1. Tea algal leaf spot |          0.2457 | 1799 |
| 2. Brown Blight        |          0.0317 | 1957 |
| 3. Gray Blight         |          0.6839 | 1949 |
| 4. Helopeltis          |          0.6498 | 1879 |
| 5. Red spider          |          0.5586 | 1588 |
| 7. Healthy leaf        |          0.9972 | 1782 |


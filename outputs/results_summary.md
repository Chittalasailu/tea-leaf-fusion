# Results summary

Device: mps, seeds: [42, 7, 123]

## Table II: images per class

| cls                    |   count |
|:-----------------------|--------:|
| 1. Tea algal leaf spot |     418 |
| 2. Brown Blight        |     508 |
| 3. Gray Blight         |    1013 |
| 4. Helopeltis          |     607 |
| 5. Red spider          |     515 |
| 6. Green mirid bug     |    1282 |
| 7. Healthy leaf        |     935 |

Split (train/val/test): {'train': 3694, 'val': 528, 'test': 1056}

## Table III: test-set results (%)

| config                    | accuracy     | precision_w   | recall_w     | f1_w         | f1_macro     | accuracy_equal_weights   |
|:--------------------------|:-------------|:--------------|:-------------|:-------------|:-------------|:-------------------------|
| CNN                       | 87.06 ± 1.06 | 86.99 ± 1.16  | 87.06 ± 1.06 | 86.98 ± 1.1  | 84.69 ± 1.16 | 87.06 ± 1.06             |
| DenseNet121               | 93.43 ± 0.29 | 93.6 ± 0.16   | 93.43 ± 0.29 | 93.47 ± 0.26 | 92.35 ± 0.35 | 93.43 ± 0.29             |
| VGG16                     | 88.16 ± 0.53 | 88.22 ± 0.46  | 88.16 ± 0.53 | 88.09 ± 0.5  | 86.37 ± 0.67 | 88.16 ± 0.53             |
| CNN + DenseNet121         | 94.13 ± 0.28 | 94.22 ± 0.22  | 94.13 ± 0.28 | 94.14 ± 0.27 | 93.12 ± 0.42 | 94.16 ± 0.29             |
| CNN + VGG16               | 92.2 ± 0.44  | 92.2 ± 0.47   | 92.2 ± 0.44  | 92.14 ± 0.44 | 90.61 ± 0.6  | 92.2 ± 0.44              |
| DenseNet121 + VGG16       | 93.43 ± 0.54 | 93.5 ± 0.53   | 93.43 ± 0.54 | 93.44 ± 0.53 | 92.21 ± 0.8  | 93.53 ± 0.61             |
| CNN + DenseNet121 + VGG16 | 94.51 ± 0.78 | 94.55 ± 0.74  | 94.51 ± 0.78 | 94.51 ± 0.76 | 93.39 ± 0.98 | 94.48 ± 0.55             |

## Fusion weights (first seed)

- CNN: CNN 1.000
- DenseNet121: DenseNet121 1.000
- VGG16: VGG16 1.000
- CNN + DenseNet121: CNN 0.487, DenseNet121 0.513
- CNN + VGG16: CNN 0.498, VGG16 0.502
- DenseNet121 + VGG16: DenseNet121 0.510, VGG16 0.490
- CNN + DenseNet121 + VGG16: CNN 0.327, DenseNet121 0.344, VGG16 0.330

## McNemar exact test (first seed)

- CNN + DenseNet121 + VGG16 vs CNN + DenseNet121: 11 vs 13 discordant, p = 0.8388
- CNN + DenseNet121 + VGG16 vs DenseNet121: 15 vs 10 discordant, p = 0.4244

## Models

- CNN seed 42: 1.2M params, 40 epochs, 12.3 min, 1.3 ms/image
- DenseNet121 seed 42: 7.2M params, 25 epochs, 11.8 min, 2.0 ms/image
- VGG16 seed 42: 14.7M params, 22 epochs, 11.3 min, 3.3 ms/image
- CNN seed 7: 1.2M params, 40 epochs, 12.3 min, 1.3 ms/image
- DenseNet121 seed 7: 7.2M params, 24 epochs, 11.3 min, 2.1 ms/image
- VGG16 seed 7: 14.7M params, 23 epochs, 8.3 min, 3.4 ms/image
- CNN seed 123: 1.2M params, 40 epochs, 14.4 min, 1.4 ms/image
- DenseNet121 seed 123: 7.2M params, 25 epochs, 12.2 min, 2.0 ms/image
- VGG16 seed 123: 14.7M params, 19 epochs, 6.9 min, 3.4 ms/image

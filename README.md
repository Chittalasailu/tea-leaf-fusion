# Tea leaf disease classification with multi-model fusion

Code, data split and results for the paper *Multi-Model Deep Learning Fusion for Tea Leaf Pathology Classification: Diversity, Calibration and Cross-Dataset Evaluation* (Putta Durga, Chittala Sailu, VIT-AP University). The paper is under review.

We train a small CNN, DenseNet121 and VGG16 on teaLeafBD, combine them with accuracy-weighted soft voting, and test every single, pair and triple combination on one fixed split over 3 seeds. The models are then tested without retraining on a second tea dataset, TeaLeafDiseaseBD.

## Results

teaLeafBD test set (1,056 images), mean ± std over seeds 42, 7 and 123:

| Model | Accuracy (%) | Macro-F1 (%) | TeaLeafDiseaseBD accuracy (%) |
|---|---|---|---|
| CNN | 87.06 ± 1.06 | 84.69 ± 1.16 | 40.96 ± 2.84 |
| DenseNet121 | 93.43 ± 0.29 | 92.35 ± 0.35 | 53.17 ± 0.64 |
| VGG16 | 88.16 ± 0.53 | 86.37 ± 0.67 | 43.55 ± 0.87 |
| CNN + DenseNet121 | 94.13 ± 0.28 | 93.12 ± 0.42 | 53.82 ± 0.63 |
| CNN + VGG16 | 92.20 ± 0.44 | 90.61 ± 0.60 | 46.48 ± 0.73 |
| DenseNet121 + VGG16 | 93.43 ± 0.54 | 92.21 ± 0.80 | 52.09 ± 0.26 |
| CNN + DenseNet121 + VGG16 | 94.51 ± 0.78 | 93.39 ± 0.98 | 52.75 ± 0.51 |

The last column is the 6 classes the two datasets share (10,954 images). Accuracy drops by about 40 points on the second dataset, and it drops the other way too: DenseNet121 trained on TeaLeafDiseaseBD gets 97.76% on its own test set but 68.19% on teaLeafBD.

All numbers are in `outputs/` (`table3.csv`, `results_summary.md`, `summary.json`) and `outputs/analysis/` (diversity, cross-dataset, calibration, agreement gate and remedies).

## Data

The datasets are not included here. Download them from Mendeley Data:

- teaLeafBD, version 4: https://doi.org/10.17632/744vznw5k2.4 (CC BY 4.0)
- TeaLeafDiseaseBD, version 3: https://doi.org/10.17632/jcvxm4gh7d.3 (CC BY 4.0)

Put them in `data/teaLeafBD/` (one folder per class) and `data/TeaLeafDiseaseBD/`.

The train/val/test split we used is in `outputs/split.csv` (3,694 / 528 / 1,056 images, stratified, seed 42). `tea_fusion.py` reuses it when it is there, so you get exactly the same test set.

## Running

Python 3.10+ and PyTorch 2.5+.

```bash
pip install -r requirements.txt

TEA_SEEDS=42,7,123 python tea_fusion.py   # train the 3 models, evaluate all 7 combinations
python analysis.py                        # diversity + test on TeaLeafDiseaseBD
python calibration.py                     # ECE, temperature scaling, AUC, 0.80 threshold
python gate.py                            # agreement gate
python remedies.py                        # background removal and AdaBN
python reverse.py                         # train on TeaLeafDiseaseBD, test on teaLeafBD
python sample_figure.py samples.png       # example images from both datasets
```

`TEA_SMOKE=1 python tea_fusion.py` does a quick 1-epoch run to check that everything works.

The extra analyses of the revised paper:

```bash
python revision.py          # E1-E13: rules, calibration, risk-coverage, cross-dataset, remedies, ablation (saved outputs only)
python robustness_eval.py   # E16-E18: regenerate saved predictions, rerun the reverse model, six corruptions (needs images + weights)
python benchmark.py         # E15: parameters, GFLOPs, CPU latency
python -m pytest tests      # E14: 158 tests
streamlit run app.py        # prototype: upload a leaf photo, get advise / refer
python inference.py leaf.jpg
```

Results go to `outputs/revision/`. The TeaLeafDiseaseBD labels, in the order the scripts read the images, are in `outputs/analysis/external_labels.npy`.

The saved predictions (`probs_*.npz`, `ext_probs_*.npy`) are in the repo. `tea_fusion.py` skips training for any model whose predictions already exist, so the tables can be rebuilt without a GPU. Delete them to train from scratch.

Training took about 12 minutes per model per seed on an Apple M4 Max (MPS). It also runs on CUDA or CPU.

## Trained weights

The weights for all 9 models (3 models x 3 seeds) and the reverse-direction DenseNet121 are on the [Releases](../../releases) page. Put them in `outputs/` (and `outputs_reverse/`) to use them.

## Files

| File | What it does |
|---|---|
| `tea_fusion.py` | data split, training, soft-voting fusion, McNemar test, figures |
| `tea_models.py` | CNN, DenseNet121 and VGG16 definitions |
| `analysis.py` | diversity measures and the cross-dataset test |
| `calibration.py` | calibration and confidence threshold |
| `gate.py` | agreement gate |
| `remedies.py` | test-time background removal and AdaBN |
| `leafmask.py` | leaf segmentation used for background removal |
| `reverse.py` | reverse-direction experiment |
| `sample_figure.py` | sample image grid |
| `fusionlib.py` | fusion rules, calibration, diversity, significance tests, selective prediction (numpy only) |
| `revision.py` | experiments E1-E13 from the saved predictions |
| `corruptions.py` | six image corruptions at two levels |
| `robustness_eval.py` | reruns with the released weights (E16-E18) |
| `benchmark.py` | model size, GFLOPs, CPU latency (E15) |
| `inference.py` | single-image prediction with the confidence-gated advice rule |
| `app.py` | Streamlit prototype |
| `tests/` | pytest suite (unit, data, regression, model, corruption, integration, acceptance) |

## License

Code is under the MIT License. The datasets keep their own CC BY 4.0 licenses.

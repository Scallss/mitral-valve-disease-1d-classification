# Mitral Valve Disease Classification from Digitized 12-Lead ECG

3-class classification of mitral valve disease (Control / Significant MR / Significant MS) using 1D deep learning on digitized 12-lead ECG signals.

## Latest model: 1-D ResNet on extraction v2 / v3 (2026-10)

The newest model is in [`ribeiro_resnet_v3/`](ribeiro_resnet_v3/): one flat 1-D ResNet (five fold networks per extraction version, plain probabilities, no blend weights or thresholds) with its weights, `predict.py` and the training recipe. On the 3,626 held-out scans: AUC 0.840 / macro-F1 0.606 (extraction v2) and 0.834 / 0.595 (v3, with overlap tracking), against 0.826 AUC for the `rib_ext` + `multitask` ensemble below on the 3,617 scans both were scored on. The weights are also on Kaggle: [pascalhafidzfajri/mitral-resnet1d-v2-v3](https://www.kaggle.com/datasets/pascalhafidzfajri/mitral-resnet1d-v2-v3). The earlier version of this repository (the `rib_ext` + `multitask` ensemble) is kept in the branch `archive/2026-06-thesis-ensemble` and the tag `archive-2026-06-thesis-ensemble`. The extraction code is in the [`ecg-1d-digitization`](https://github.com/Scallss/ecg-1d-digitization) repository (branch `release-v2-v3-2026-10`). The table and the notebooks below describe the earlier ensemble.

## Results Summary

| Method | Macro F1 | AUC | Ctrl Recall | MR Recall | MS Recall |
|---|---|---|---|---|---|
| Best single (rib_ext) | 0.5700 | 0.823 | 0.716 | 0.569 | 0.738 |
| Hard ensemble (rib_ext+cnn_iv) | 0.5980 | 0.822 | 0.796 | 0.532 | 0.617 |
| **Soft+threshold (rib_ext+multitask)** | **0.6049** | **0.826** | **0.744** | **0.623** | **0.638** |
| Baseline Tiffany (EfficientNet 2D) | 0.5884 | 0.841 | 0.826 | 0.584 | 0.653 |

## Model Variants

7 model configurations reported in the technical report:

| ID | Model | Architecture | Features | Params |
|---|---|---|---|---|
| R3 | `rib_ptbxl_iv` | Ribeiro ResNet + PTB-XL pretrain | 6 interval | ~8M |
| R4 | `rib_ext` | Ribeiro ResNet + PTB-XL pretrain | 12 extended | ~8M |
| I2 | `incep_iv` | InceptionTime nf=64 | 6 interval (input fusion) | ~2.1M |
| I3 | `incep_ext` | InceptionTime nf=64 | 12 extended (input fusion) | ~2.1M |
| I4 | `multitask` | InceptionTime nf=64 | 6 interval (aux regression) | ~2.1M |
| C1 | `cnn_bilstm` | CNN-BiLSTM-Attention | none | ~880K |
| C2 | `cnn_iv` | CNN-BiLSTM-Attention | 6 interval (input fusion) | ~880K |

## Dataset

### Data Sources (Kaggle)

| Dataset | Access | Link |
|---|---|---|
| Mitral ECG (NPZ files, split CSV, interval CSV) | **Private** | https://kaggle.com/datasets/b51641590c817242d62c271cf5e1dff081a505644056c70b2731bad71773199d |
| PTB-XL ECG Database (for pretraining) | Public | https://www.kaggle.com/datasets/sidalikhelilcherfi/ptbxl-ecg-database |

### Dataset Description

- **Mitral ECG**: Digital 12-lead ECG signals from mitral valve patients at RSJPD Harapan Kita. Each sample is an NPZ file containing `signal_matrix` (5000×12, 500Hz), `overlap_mask`, and `interpolated_mask`. The dataset also includes `data_split_3level_exp10.csv` (Tiffany's reference split) and `ecg_intervals.csv` / `ecg_features_extended.csv` (extracted ECG features).
- **PTB-XL**: ~21,800 digital 12-lead ECG recordings from PhysioNet, used for supervised pretraining of the Ribeiro ResNet backbone on 5 superclasses (NORM/MI/STTC/CD/HYP).

### Class Distribution

| Split | Control | MR | MS | Total |
|---|---|---|---|---|
| CV Pool (5-fold training) | 799 | 797 | 399 | 1,995 |
| Heldout (test) | 2,535 | 933 | 149 | 3,617 |
| Total | 3,334 | 1,730 | 548 | 5,612 |

- 14 samples were excluded due to poor digitization quality (NPZ not available)
- Control class includes: mild MR (label A), mild MS (label C), and other conditions (label E)

## Notebooks

### Environment

- **Notebook 00–08**: Run on **Kaggle** (GPU P100/T4, dataset mounted from Kaggle Datasets)
- **Notebook 09–10**: Run **locally**, using output (`dl_artifacts/`) downloaded from Kaggle

### Notebook Description

| # | Notebook | Env | Description | Required Input |
|---|---|---|---|---|
| 00 | `00_ptbxl_pretrain_ribeiro.ipynb` | Kaggle | Pretrain Ribeiro ResNet (36-ch) backbone on PTB-XL 5-superclass. Output: `ptbxl_pretrained_backbone.pt` | PTB-XL dataset |
| 01a | `01a_ecg_interval_extraction.ipynb` | Kaggle | Extract 6 ECG interval features (RR, RR-std, P-dur, QRS-dur, QTc, Sokolow) from leads II and V1/V5 using NeuroKit2. Output: `ecg_intervals.csv` | NPZ files, `data_split_3level_exp10.csv` |
| 01b | `01b_ecg_morphology_extraction.ipynb` | Kaggle | Extract 6 ECG morphology features (P/T/S amplitude, QRS axis) from multi-lead. Output: `ecg_morphology_features.csv` → merged with `ecg_intervals.csv` into `ecg_features_extended.csv` | NPZ files, `ecg_intervals.csv` |
| 02 | `02_train_ribeiro_ext.ipynb` | Kaggle | Train Ribeiro ResNet + PTB-XL pretrained + 12 extended features (dual-branch fusion). **Best single model** (F1=0.5700) | NPZ files, split CSV, `ecg_features_extended.csv`, pretrained weights |
| 03 | `03_train_ribeiro_ptbxl_iv.ipynb` | Kaggle | Train Ribeiro ResNet + PTB-XL pretrained + 6 interval features (dual-branch fusion) | NPZ files, split CSV, `ecg_intervals.csv`, pretrained weights |
| 04 | `04_train_inceptiontime_iv.ipynb` | Kaggle | Train InceptionTime nf=64 + 6 interval features (dual-branch fusion). 12-ch overlap-zeroed input | NPZ files, split CSV, `ecg_intervals.csv` |
| 05 | `05_train_inceptiontime_ext.ipynb` | Kaggle | Train InceptionTime nf=64 + 12 extended features (dual-branch fusion) | NPZ files, split CSV, `ecg_features_extended.csv` |
| 06 | `06_train_inceptiontime_multitask.ipynb` | Kaggle | Train InceptionTime nf=64 with auxiliary regression head — 6 ECG intervals as regression targets (not input). Loss = CE + 0.1×MSE | NPZ files, split CSV, `ecg_intervals.csv` |
| 07 | `07_train_cnn_bilstm.ipynb` | Kaggle | Train CNN-BiLSTM-Attention (multi-scale CNN k=7/15/31 → BiLSTM(256) → soft attention). Most lightweight architecture (~880K params) | NPZ files, split CSV |
| 08 | `08_train_cnn_bilstm_iv.ipynb` | Kaggle | Train CNN-BiLSTM-Attention + 6 interval features (dual-branch fusion) | NPZ files, split CSV, `ecg_intervals.csv` |
| 09 | `09_cross_arch_ensemble.ipynb` | Local | Evaluate cross-architecture ensemble: hard voting, soft voting, and comparison of cross-arch vs within-arch. No GPU needed — loads saved probabilities only | `dl_artifacts/*/fold_*/heldout_logits.npz` |
| 10 | `10_threshold_optimization.ipynb` | Local | Threshold optimization with logit-offset δ on OOF val predictions. Requires val inference (GPU) for models without existing `val_logits.npz` | `dl_artifacts/*/fold_*/heldout_logits.npz`, `dl_artifacts/*/fold_*/val_logits.npz`, model weights + NPZ files (if val inference is needed) |

### Output Structure (`dl_artifacts/`)

Each training notebook (02–08) produces outputs in `dl_artifacts/<model_name>/` with the following structure:

```
dl_artifacts/
└── ribeiro_resnet_ecg_features_ext_tiffany_flat/    # example: rib_ext
    ├── heldout_ensemble_metrics.csv                  # heldout ensemble metrics
    ├── heldout_confusion_matrix.csv
    ├── heldout_confusion_matrix_normalized.csv
    ├── heldout_fold_metrics.csv                      # per-fold metrics
    ├── fold_0/
    │   ├── best_model.pt                             # best checkpoint (~31MB Ribeiro, ~7MB Incep, ~3MB CNN)
    │   ├── heldout_logits.npz                        # heldout probabilities (probs, y_true, stems)
    │   ├── val_logits.npz                            # OOF val probabilities (for threshold tuning)
    │   ├── heldout_metrics.csv
    │   ├── training_history.csv                      # loss & metrics per epoch
    │   ├── train_split.csv                           # train fold stem list
    │   ├── val_split.csv                             # validation fold stem list
    │   └── interval_stats.npy                        # mean/std of interval features (if model uses features)
    ├── fold_1/ ...
    ├── fold_2/ ...
    ├── fold_3/ ...
    └── fold_4/ ...
```

Notebooks 09–10 (local) require the entire `dl_artifacts/` folder with at least `heldout_logits.npz` per fold per model. Download the Kaggle output and place it in the project root.

## Model Weights & Artifacts

Pretrained weights, inference logits, and metrics are available on Google Drive:

**[Download dl_artifacts.zip (475 MB)](https://drive.google.com/file/d/1hKTGekIDI-rFYrs_z9H8j_BuzJGFr99H/view?usp=sharing)**

After downloading, extract it in the project root:

```bash
unzip dl_artifacts.zip
```

This will generate the `dl_artifacts/` folder with the structure required by notebooks 09–10 (ensemble & threshold optimization). Zip contents:

| Folder | Contents | Size |
|---|---|---|
| `ptbxl_pretrain/` | Ribeiro PTB-XL pretrained backbone weights | 62 MB |
| 7 model folders (×5 folds) | `best_model.pt`, `heldout_logits.npz`, `val_logits.npz`, metrics CSV, split CSV | ~471 MB |

## Pipeline Overview

```
ECG NPZ (5000x12) → Overlap-zeroing → Per-sample normalization
    → Backbone (Ribeiro/InceptionTime/CNN-BiLSTM)
    → [Optional: Feature fusion / Multitask head]
    → 5-fold CV training (400/class balanced)
    → Within-arch ensemble (5-fold soft avg)
    → Cross-arch ensemble (soft voting)
    → Threshold optimization (logit-offset δ, tuned on OOF val)
    → Final prediction: Control / MR / MS
```

## Requirements

- Python 3.11+
- PyTorch 2.x (CUDA recommended for notebooks 00–08)
- neurokit2, scikit-learn, numpy, pandas, matplotlib, tqdm

## References

### 1. Deep Learning Architectures — ECG Domain

#### Ribeiro et al. (2020)
- **Title:** Automatic diagnosis of the 12-lead ECG using a deep neural network
- **Authors:** Ribeiro, A. H., Ribeiro, M. H., Paixão, G. M. M., Oliveira, D. M., Gomes, P. R., Canazart, J. A., Ferreira, M. P. S., Andersson, C. R., Macfarlane, P. W., Meira Jr., W., Schön, T. B., & Ribeiro, A. L. P.
- **Journal:** *Nature Communications*, 11, 1760

#### Ismail Fawaz et al. (2020)
- **Title:** InceptionTime: Finding AlexNet for Time Series Classification
- **Authors:** Ismail Fawaz, H., Lucas, B., Forestier, G., Pelletier, C., Schmidt, D. F., Weber, J., Webb, G. I., Idoumghar, L., Muller, P.-A., & Petitjean, F.
- **Journal:** *Data Mining and Knowledge Discovery*, 34(6), 1936–1962

#### Mechichi & Benzarti (2025)
- **Title:** An Enhanced Hybrid Model Combining CNN, BiLSTM, and Attention Mechanism for ECG Segment Classification
- **Authors:** Mechichi, N., & Benzarti, F.
- **Journal:** *Biomedical Engineering and Computational Biology*, 16

#### Thota et al. (2025)
- **Title:** A Lightweight CNN-Attention-BiLSTM Architecture for Multi-Class Arrhythmia Classification on Standard and Wearable ECGs
- **Authors:** Thota, V., Prajapati, H., Joshi, Y., & Rathi, S.
- **Source:** *arXiv:2511.08650*

---

### 2. Deep Learning Architectures — General

#### He et al. (2016)
- **Title:** Deep Residual Learning for Image Recognition
- **Authors:** He, K., Zhang, X., Ren, S., & Sun, J.
- **Conference:** *CVPR 2016*, 770–778

#### Huang et al. (2017)
- **Title:** Densely Connected Convolutional Networks
- **Authors:** Huang, G., Liu, Z., van der Maaten, L., & Weinberger, K. Q.
- **Conference:** *CVPR 2017*, 2261–2269 (Best Paper Award)

#### Vaswani et al. (2017)
- **Title:** Attention Is All You Need
- **Authors:** Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I.
- **Conference:** *NeurIPS 2017*, 5998–6008

#### Hu et al. (2018)
- **Title:** Squeeze-and-Excitation Networks
- **Authors:** Hu, J., Shen, L., & Sun, G.
- **Conference:** *CVPR 2018*, 7132–7141

#### Bai et al. (2018)
- **Title:** An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling
- **Authors:** Bai, S., Kolter, J. Z., & Koltun, V.
- **Source:** *arXiv:1803.01271*

---

### 3. Training & Regularization Techniques

#### Lin et al. (2017)
- **Title:** Focal Loss for Dense Object Detection
- **Authors:** Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P.
- **Conference:** *ICCV 2017*, 2999–3007

#### Szegedy et al. (2016)
- **Title:** Rethinking the Inception Architecture for Computer Vision
- **Authors:** Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z.
- **Conference:** *CVPR 2016*, 2818–2826

#### Zhang et al. (2018)
- **Title:** mixup: Beyond Empirical Risk Minimization
- **Authors:** Zhang, H., Cissé, M., Dauphin, Y. N., & Lopez-Paz, D.
- **Conference:** *ICLR 2018*

#### Ruder (2017)
- **Title:** An Overview of Multi-Task Learning in Deep Neural Networks
- **Authors:** Ruder, S.
- **Source:** *arXiv:1706.05098*

---

### 4. Feature Fusion

#### Qiu et al. (2024)
- **Title:** Deep learning-based multimodal fusion of the surface ECG and clinical features in prediction of atrial fibrillation recurrence following catheter ablation
- **Authors:** Qiu, Y., Guo, H., Wang, S., Yang, S., Peng, X., Xiayao, D., Chen, R., Yang, J., Liu, J., Li, M., Li, Z., Chen, H., & Chen, M.
- **Journal:** *BMC Medical Informatics and Decision Making*, 24

---

### 5. Dataset & Tools

#### Wagner et al. (2020)
- **Title:** PTB-XL, a large publicly available electrocardiography dataset
- **Authors:** Wagner, P., Strodthoff, N., Bousseljot, R.-D., Kreiseler, D., Lunze, F. I., Samek, W., & Schaeffter, T.
- **Journal:** *Scientific Data*, 7, 154

#### Strodthoff et al. (2021)
- **Title:** Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL
- **Authors:** Strodthoff, N., Wagner, P., Schaeffter, T., & Samek, W.
- **Journal:** *IEEE Journal of Biomedical and Health Informatics*, 25(5), 1519–1528

#### Makowski et al. (2021)
- **Title:** NeuroKit2: A Python Toolbox for Neurophysiological Signal Processing
- **Authors:** Makowski, D., Pham, T., Lau, Z. J., Brammer, J. C., Lespinasse, F., Pham, H., Schölzel, C., & Chen, S. H. A.
- **Journal:** *Behavior Research Methods*, 53, 1689–1696

---

### 6. Baselines & Statistics

#### Tiffany, A. (2026)
- **Title:** Klasifikasi Penyakit Katup Mitral dari Citra ECG 2D Menggunakan Convolutional Neural Network

#### Breiman (2001)
- **Title:** Random Forests
- **Authors:** Breiman, L.
- **Journal:** *Machine Learning*, 45(1), 5–32

#### Cohen (1988)
- **Title:** Statistical Power Analysis for the Behavioral Sciences (2nd ed.)
- **Authors:** Cohen, J.
- **Publisher:** Lawrence Erlbaum Associates

---

### 7. Scaling Laws — ECG

#### Li et al. (2026)
- **Title:** How Do Electrocardiogram Models Scale?
- **Authors:** Li, J., Bonassi, F., Jin, M., Gustafsson, S., Sundström, J., Schön, T. B., & Ribeiro, A. H.
- **Source:** *arXiv:2605.17276*

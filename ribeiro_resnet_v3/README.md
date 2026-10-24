# 1-D ResNet on extraction v2 / v3 (latest model)

Three-class classification (Control / significant MR / significant MS) from a digitized 12-lead ECG. One flat 1-D ResNet (7.9 M parameters, from scratch), five fold networks per extraction version, class probability = mean of the five softmax outputs, class = argmax. No blend weights, no thresholds, no clinical-feature step.

| Version | Digitizer (repo `ecg-1d-digitization`, branch `release-v2-v3-2026-10`) | Weights |
|---|---|---|
| v2 | `pipeline/bfss`, preset `black165_stable_sep_repair` (ink filter + stability-based lead segmentation with repair) | `weights/v2_bfss/fold0..4.pt` |
| v3 | `pipeline/bfss_tracker`, preset `black165_stable_sep_repair_tracker` (v2 + overlap tracking) | `weights/v3_tracker/fold0..4.pt` |

Use the weights with the extraction they were trained on.

## Use

```bash
pip install torch numpy opencv-python        # numba is optional (faster tracker, identical output)
python predict.py --version v3 --digitizer ../../ecg-1d-digitization scan1.png scan2.png     # images -> probabilities
python predict.py --version v3 --npz export.npz                                              # exported signal_matrix (5000 x 12)
```

Output: `name,p_Control,p_MR,p_MS,predicted`. The probabilities follow the training mix (40% Control, 40% MR, 20% MS), not the clinical prevalence; no prior shift or threshold is applied.

`model.py` holds the network, the per-lead normalisation and the ensemble loader; `train.py` documents the exact training recipe (it needs the private clinical data, which is not in this repository).

## Results (one run of five networks, seed 0)

Held-out set of 3,626 scans (natural class mix, never used for any choice):

| | AUC (macro OvR) | PR-AUC | macro-F1 |
|---|---|---|---|
| v2 | 0.840 | 0.660 | 0.606 |
| v3 | 0.834 | 0.665 | 0.595 |

On the 3,617 scans that the earlier production ensemble (`rib_ext` + `multitask`) was also scored on: AUC 0.826 for the production ensemble; paired difference to v2 +0.014 [+0.006, +0.022], to v3 +0.008 [-0.002, +0.018] (95% bootstrap over scans, test-sample noise only); macro-F1 differences are within noise.

Cost per scan on a free Kaggle machine (Tesla T4, 4 vCPU, batch 1, PyTorch fp32, image file to probabilities): v2 225 ms / v3 234 ms with the GPU, 664 / 671 ms on the CPU only; peak RAM about 1.1-1.2 GB.

## Limits
- One seed. The seed-to-seed spread of the full recipe is about 0.002 AUC (out-of-fold) and about 0.03 macro-F1, so v2 and v3 cannot be told apart from these numbers.
- Pool and held-out set come from the same acquisition batches: this is not an external validation, and it is not a clinical validation.
- The digitizer assumes the 3x4 + rhythm strip layout in standard lead order (an optional adaptive layout exists in the digitizer repository, validated on synthetic sheets only).
- Weights were trained on the private RSJPD Harapan Kita data set; the data itself is not included. `weights/SHA256SUMS` lists the checkpoint hashes (fp16 weights of the weight-averaged model at epoch 20).

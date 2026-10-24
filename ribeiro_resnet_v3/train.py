"""Training recipe of the released networks: 5-fold cross-validation on the labelled pool, one network per fold, from scratch.

Recipe (fixed, nothing tuned on the held-out set): 12 normalised lead channels, AdamW lr 3e-4 weight decay 1e-4, batch 32, cosine schedule over 20 epochs (floor 1e-5),
gradient clipping 5, weight averaging (EMA, decay 0.995; the EMA model is what is released), lead dropout (with probability 0.5 per scan, 1 to 3 random leads set to zero),
unweighted cross-entropy plus 2 x masked MSE of an auxiliary head that predicts six standardised interval measurements (RR, RR spread, P duration, QRS duration, QTc, Sokolow-Lyon;
used in training only, clipped to +-5, standardised with the training fold's median and standard deviation).

Inputs (private clinical data, not part of this repository):
  --signal   .npy of shape (N, 5000, 12), the digitized signals in the row order of --split (NaN outside a lead's own window)
  --split    CSV with columns filename, flat_label (Kontrol / MR / MS) and isHeldOut (true / false); the pool is the rows with isHeldOut false
  --targets  CSV with the six auxiliary columns, same row order as --split (NaN allowed)
Outputs in --out: fold{k}.pt (fp16 EMA weights, the format `model.py` loads), val_idx_fold{k}.npy, val_logits_fold{k}.npy (EMA), held_logits_fold{k}.npy (EMA).

    python train.py --signal tracker_signal.npy --split split.csv --targets features_tracker_raw.csv --out runs/v3 --seed 0
"""
import argparse, copy, random, time
from pathlib import Path

import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold

from model import PlainRibeiro, normalise

TARGET_COLUMNS = ["old_rr_ms", "old_rr_std_ms", "old_p_dur_ms", "old_qrs_dur_ms", "old_qtc_ms", "old_sokolow"]
EPOCHS, LR, WD, MIN_LR, BATCH, CLIP, EMA_DECAY, LAM, REG_CLIP, FOLD_SEED = 20, 3e-4, 1e-4, 1e-5, 32, 5.0, 0.995, 2.0, 5.0, 42
LDROP_P, LDROP_K = 0.5, 3
CLS = {"Kontrol": 0, "MR": 1, "MS": 2}


def lead_dropout(x, g):
    """With probability LDROP_P per scan, zero 1..LDROP_K randomly chosen leads."""
    B, C, _ = x.shape; dev = x.device
    on = torch.rand(B, device=dev, generator=g) < LDROP_P; k = torch.randint(1, LDROP_K + 1, (B,), device=dev, generator=g)
    rank = torch.rand(B, C, device=dev, generator=g).argsort(1).argsort(1); drop = (rank < k[:, None]) & on[:, None]
    return x * (~drop)[:, :, None].float()


@torch.no_grad()
def ema_update(ema, model, step):
    d = min(EMA_DECAY, (1 + step) / (10 + step))
    for (_, v), (_, w) in zip(ema.state_dict().items(), model.state_dict().items()):
        if v.is_floating_point(): v.mul_(d).add_(w.detach(), alpha=1 - d)
        else: v.copy_(w)


@torch.no_grad()
def predict(model, X, idx, device, bs=128):
    model.eval(); out = []
    for a in range(0, len(idx), bs):
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"): out.append(model(X[idx[a:a + bs]].to(device)).float().cpu())
    return torch.cat(out).numpy().astype(np.float32)


def masked_mse(pred, target):
    m = torch.isfinite(target)
    return F.mse_loss(pred[m].float(), target[m]) if m.any() else pred.sum() * 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--signal", required=True); ap.add_argument("--split", required=True); ap.add_argument("--targets", required=True); ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sp = pd.read_csv(a.split); y_all = sp.flat_label.map(CLS).values.astype(int)
    held = sp.isHeldOut.astype(str).str.lower().isin(["true", "1"]).values; pool_idx, held_idx = np.where(~held)[0], np.where(held)[0]
    X = torch.from_numpy(normalise(np.load(a.signal, mmap_mode="r"))); Y = torch.from_numpy(y_all)
    T = torch.tensor(pd.read_csv(a.targets)[TARGET_COLUMNS].values, dtype=torch.float32); assert len(T) == len(sp)
    folds = [(pool_idx[t], pool_idx[v]) for t, v in StratifiedKFold(5, shuffle=True, random_state=FOLD_SEED).split(pool_idx, y_all[pool_idx])]
    for fold, (tr, va) in enumerate(folds):
        tt = T[tr].numpy(); mu = torch.tensor(np.nanmedian(tt, 0)); sd = torch.tensor(np.maximum(np.nanstd(tt, 0), 1e-6))
        Te = ((T - mu) / sd).clamp(-REG_CLIP, REG_CLIP)                                        # NaN stays NaN (masked in the loss); training-fold statistics only
        s = a.seed * 1000 + fold; random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        model = PlainRibeiro(n_reg=len(TARGET_COLUMNS)).to(device); ema = copy.deepcopy(model).eval()
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=MIN_LR)
        scaler = torch.amp.GradScaler(enabled=device.type == "cuda"); ce = nn.CrossEntropyLoss()
        g = torch.Generator().manual_seed(s); ga = torch.Generator(device=device).manual_seed(s + 7); step, t0 = 0, time.time()
        for ep in range(1, EPOCHS + 1):
            model.train(); perm = tr[torch.randperm(len(tr), generator=g).numpy()]
            for b0 in range(0, len(perm), BATCH):
                b = perm[b0:b0 + BATCH]
                if len(b) < 2: continue
                xb, yb = lead_dropout(X[b].to(device), ga), Y[b].to(device)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    logits, reg = model(xb, aux=True); loss = ce(logits, yb) + LAM * masked_mse(reg, Te[b].to(device))
                opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), CLIP); scaler.step(opt); scaler.update()
                step += 1; ema_update(ema, model, step)
            sched.step()
        torch.save({k: (v.half() if v.is_floating_point() else v) for k, v in ema.state_dict().items()}, out / f"fold{fold}.pt")
        np.save(out / f"val_idx_fold{fold}.npy", va); np.save(out / f"val_logits_fold{fold}.npy", predict(ema, X, va, device)); np.save(out / f"held_logits_fold{fold}.npy", predict(ema, X, held_idx, device))
        print(f"fold {fold} done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()

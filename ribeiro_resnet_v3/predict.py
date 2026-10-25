"""Class probabilities (Control / MR / MS) for ECG scan images or digitized signals with the five fold networks of one extraction version.

    # from images (needs the digitizer repository, https://github.com/Scallss/ecg-1d-digitization, branch release-v2-v3-2026-10)
    python predict.py --version v3 --digitizer ../../ecg-1d-digitization scan1.png scan2.png
    # from an exported .npz of the digitizer (key signal_matrix, 5000 x 12)
    python predict.py --version v3 --npz export.npz

--version v3: extraction with overlap tracking (preset black165_stable_sep_repair_tracker, weights in <weights>/v3_tracker).
--version v2: extraction without it (preset black165_stable_sep_repair, weights in <weights>/v2_bfss).
The weights are not part of this repository; download the Kaggle dataset pascalhafidzfajri/mitral-resnet1d-v2-v3 into ./weights (or pass --weights DIR):
    kaggle datasets download -d pascalhafidzfajri/mitral-resnet1d-v2-v3 --unzip -p .
Output: one CSV line per input: name, p_Control, p_MR, p_MS, predicted class. The probabilities follow the training mix (40% Control, 40% MR, 20% MS), not the clinical prevalence.
"""
import argparse, os, sys
from pathlib import Path

import numpy as np
import torch

from model import CLASSES, load_ensemble, normalise, predict_proba

HERE = Path(__file__).resolve().parent
VERSIONS = {"v2": dict(weights="v2_bfss", folder="bfss", preset="black165_stable_sep_repair"),
            "v3": dict(weights="v3_tracker", folder="bfss_tracker", preset="black165_stable_sep_repair_tracker")}


def get_digitizer(root, version):
    """Import the digitizer of the version and return a function image path -> (5000, 12) signal."""
    v = VERSIONS[version]; pipe = Path(root) / "pipeline" / v["folder"]
    if not pipe.is_dir(): sys.exit(f"digitizer folder not found: {pipe}")
    try:
        import numba  # noqa: F401  (optional compiled tracker, identical output, much faster)
        os.environ.setdefault("OVERLAP_TRACKER_FAST", "1")
    except ImportError:
        pass
    sys.path[:0] = [str(pipe)]
    import ecg_digitizer as ed, extract_all_npz as X
    params = ed.get_params(v["preset"]); run = {**params, "build_debug_views": False}     # the debug images are not needed to get the signal

    def digitize(path):
        res = ed.digitize_ecg_image(Path(path), params=run)
        return np.asarray(X.build_canonical_export(res, params, image_path_text="x")[1], np.float32)
    return digitize


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="*", help="scan images (.png / .jpg)"); ap.add_argument("--npz", nargs="*", default=[], help="digitizer exports with signal_matrix")
    ap.add_argument("--version", choices=list(VERSIONS), default="v3"); ap.add_argument("--digitizer", default=str(HERE.parents[1] / "ecg-1d-digitization"), help="root of the digitizer repository")
    ap.add_argument("--weights", default=str(HERE / "weights"), help="folder with v2_bfss/ and v3_tracker/ (from the Kaggle dataset)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    if not a.inputs and not a.npz: ap.error("give scan images or --npz files")
    wdir = Path(a.weights) / VERSIONS[a.version]["weights"]
    if not (wdir / "fold0.pt").exists(): sys.exit(f"weights not found in {wdir}; download the Kaggle dataset pascalhafidzfajri/mitral-resnet1d-v2-v3 (see the top of this file)")
    nets = load_ensemble(wdir, a.device)
    digitize = get_digitizer(a.digitizer, a.version) if a.inputs else None
    print("name,p_Control,p_MR,p_MS,predicted")
    for name in a.inputs + a.npz:
        sig = digitize(name) if name in a.inputs else np.load(name)["signal_matrix"]
        p = predict_proba(nets, torch.from_numpy(normalise(sig[None])).to(a.device))[0]
        print(f"{Path(name).name},{p[0]:.4f},{p[1]:.4f},{p[2]:.4f},{CLASSES[int(p.argmax())]}")


if __name__ == "__main__":
    main()

"""
Emotion EEG preprocessing for CBraMod

Input
- train/HC/*.mat
- train/DEP/*.mat
- test/*.mat (optional, unlabeled)

Output
- LMDB database with train/val/test splits
- Each sample shape: (30, 10, 200)
"""

import argparse
import hashlib
import logging
import math
import pickle
from pathlib import Path

import h5py
import lmdb
import numpy as np
import scipy.io as sio
from scipy import signal

try:
    import mne
    from mne.preprocessing import ICA

    _HAS_MNE = True
except Exception:  # pragma: no cover
    mne = None
    ICA = None
    _HAS_MNE = False


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ===== FACED-like minimal preprocessing =====
# Notes:
# - This is intentionally minimal and robust: bandpass + MAD bad channel interpolation + ICA(EOG proxy).
# - We run it once per *continuous recording* (e.g., EEG_data_pos full length), then slice trials/windows.
EEG_MINPREP_DEFAULTS = {
    "enabled": True,
    "notch_freqs": [50.0],  # China mains
    "bp_l": 0.05,
    "bp_h": 47.0,
    "filter_method": "fir",  # MNE default is zero-phase FIR when phase="zero"
    "filter_phase": "zero",
    "mad_k": 3.0,
    "bad_ratio": 0.30,  # >30% outliers => bad channel
    "eps": 1e-12,
    "ica_n_components": 0.99,  # keep 99% variance
    "ica_method": "fastica",
    "ica_random_state": 97,
    "ica_fit_l": 1.0,  # stabilize ICA with higher HP
    "ica_fit_h": 47.0,
    "ica_eog_threshold": 3.0,  # MNE default
}

# Channel order (given 30 channels)
CHS_ORIG = [
    "FP1",
    "FP2",
    "F7",
    "F3",
    "FZ",
    "F4",
    "F8",
    "FT7",
    "FC3",
    "FCZ",
    "FC4",
    "FT8",
    "T3",
    "C3",
    "CZ",
    "C4",
    "T4",
    "TP7",
    "CP3",
    "CPZ",
    "CP4",
    "TP8",
    "T5",
    "P3",
    "PZ",
    "P4",
    "T6",
    "O1",
    "OZ",
    "O2",
]

# old -> modern 10-20 names (for montage)
RENAME_FOR_MONTAGE = {
    "FP1": "Fp1",
    "FP2": "Fp2",
    "FZ": "Fz",
    "FCZ": "FCz",
    "CZ": "Cz",
    "CPZ": "CPz",
    "PZ": "Pz",
    "OZ": "Oz",
    "T3": "T7",
    "T4": "T8",
    "T5": "P7",
    "T6": "P8",
}


def _to_mne_name(ch: str) -> str:
    return RENAME_FOR_MONTAGE.get(ch, ch)


CHS_MNE = [_to_mne_name(c) for c in CHS_ORIG]


def _make_raw_eeg(data_samp_ch: np.ndarray, sfreq: float) -> "mne.io.Raw":
    """
    data_samp_ch: (n_samples, n_channels)
    """
    if not _HAS_MNE:
        raise RuntimeError("MNE is not available; cannot build Raw.")
    if data_samp_ch.ndim != 2:
        raise ValueError(f"Expected 2D EEG array, got shape={data_samp_ch.shape}")
    if data_samp_ch.shape[1] != len(CHS_MNE):
        raise ValueError(f"Expected {len(CHS_MNE)} channels, got {data_samp_ch.shape[1]}")

    info = mne.create_info(ch_names=CHS_MNE, sfreq=float(sfreq), ch_types=["eeg"] * len(CHS_MNE))
    raw = mne.io.RawArray(data_samp_ch.T, info, verbose=False)  # MNE: (n_ch, n_times)

    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, on_missing="warn", verbose=False)
    return raw


def _mad_outlier_ratio(x: np.ndarray, k: float, eps: float) -> float:
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + eps
    out = np.abs(x - med) > (k * mad)
    return float(out.mean())


def detect_bads_mad_union(raw: "mne.io.Raw", seg_pts: int, n_seg: int, cfg: dict) -> list[str]:
    """
    Compute MAD outlier ratio per channel on each segment, mark channel bad if any segment exceeds threshold.
    """
    data = raw.get_data(picks="eeg")  # (n_ch, n_times)
    bads: set[str] = set()
    for si in range(int(n_seg)):
        s = int(si) * int(seg_pts)
        e = s + int(seg_pts)
        if e > data.shape[1]:
            break
        seg = data[:, s:e]
        for ci, ch in enumerate(raw.ch_names):
            ratio = _mad_outlier_ratio(seg[ci], k=float(cfg["mad_k"]), eps=float(cfg["eps"]))
            if ratio > float(cfg["bad_ratio"]):
                bads.add(ch)
    return sorted(bads)


def add_eog_proxy_channel(
    raw: "mne.io.Raw", fp1: str = "Fp1", fp2: str = "Fp2", eog_name: str = "EOG"
) -> "mne.io.Raw":
    """
    Add an EOG proxy channel as mean(Fp1,Fp2), type='eog', for ICA.find_bads_eog.
    """
    picks = []
    if fp1 in raw.ch_names:
        picks.append(fp1)
    if fp2 in raw.ch_names:
        picks.append(fp2)
    if len(picks) == 0:
        raise RuntimeError("Cannot build EOG proxy: neither Fp1 nor Fp2 is present.")

    eeg = raw.get_data(picks=picks)  # (len(picks), n_times)
    eog = np.mean(eeg, axis=0, keepdims=True)  # (1, n_times)

    eog_info = mne.create_info([eog_name], sfreq=float(raw.info["sfreq"]), ch_types=["eog"])
    raw_eog = mne.io.RawArray(eog, eog_info, verbose=False)
    raw2 = raw.copy()
    raw2.add_channels([raw_eog], force_update_info=True)
    return raw2


def run_ica_eog(raw: "mne.io.Raw", cfg: dict) -> "mne.io.Raw":
    """
    Fit ICA on a filtered copy (1-47 Hz) and apply to current raw (typically already 0.05-47 Hz).
    """
    raw_w_eog = add_eog_proxy_channel(raw, fp1="Fp1", fp2="Fp2", eog_name="EOG")

    raw_fit = raw_w_eog.copy().filter(
        l_freq=float(cfg["ica_fit_l"]),
        h_freq=float(cfg["ica_fit_h"]),
        method=str(cfg["filter_method"]),
        phase=str(cfg["filter_phase"]),
        picks="eeg",
        verbose=False,
    )

    ica = ICA(
        n_components=cfg["ica_n_components"],
        method=str(cfg["ica_method"]),
        random_state=int(cfg["ica_random_state"]),
        max_iter="auto",
        verbose=False,
    )
    ica.fit(raw_fit, picks="eeg")

    eog_inds, _scores = ica.find_bads_eog(
        raw_fit, ch_name="EOG", threshold=float(cfg["ica_eog_threshold"]), verbose=False
    )
    ica.exclude = eog_inds

    raw_clean = raw_w_eog.copy()
    ica.apply(raw_clean, verbose=False)
    raw_clean.drop_channels(["EOG"])
    return raw_clean


def preprocess_recording(data_samp_ch: np.ndarray, fs: int, seg_pts: int, n_seg: int, cfg: dict | None = None) -> np.ndarray:
    """
    Minimal preprocessing on the full continuous recording.
    Returns cleaned array with shape (n_samples, n_channels).
    """
    cfg = dict(EEG_MINPREP_DEFAULTS if cfg is None else cfg)
    if not cfg.get("enabled", True):
        return data_samp_ch.astype(np.float32, copy=False)

    if not _HAS_MNE:
        log.warning("MNE is not installed; skip minimal preprocessing (bandpass/bad-interp/ICA).")
        return data_samp_ch.astype(np.float32, copy=False)

    raw = _make_raw_eeg(data_samp_ch.astype(np.float64, copy=False), sfreq=float(fs))

    if cfg.get("notch_freqs"):
        raw.notch_filter(freqs=cfg["notch_freqs"], picks="eeg", verbose=False)

    raw.filter(
        l_freq=float(cfg["bp_l"]),
        h_freq=float(cfg["bp_h"]),
        method=str(cfg["filter_method"]),
        phase=str(cfg["filter_phase"]),
        picks="eeg",
        verbose=False,
    )

    bads = detect_bads_mad_union(raw, seg_pts=int(seg_pts), n_seg=int(n_seg), cfg=cfg)
    if bads:
        log.info(
            "   bad channels detected (MAD>%.2f & ratio>%.0f%%): %s",
            float(cfg["mad_k"]),
            float(cfg["bad_ratio"]) * 100.0,
            bads,
        )
        raw.info["bads"] = bads
        try:
            raw.interpolate_bads(reset_bads=True, verbose=False)
        except Exception as e:  # pragma: no cover
            log.warning("   interpolate_bads failed; continue without interpolation. err=%r", e)
            raw.info["bads"] = []

    try:
        raw = run_ica_eog(raw, cfg=cfg)
        log.info("   ICA(EOG) applied.")
    except Exception as e:  # pragma: no cover
        log.warning("   ICA failed; continue with filter + bads only. err=%r", e)

    return raw.get_data(picks="eeg").T.astype(np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess emotion EEG mats into CBraMod LMDB.")
    parser.add_argument("--project_root", type=str, required=True, help="Dataset root containing train/HC, train/DEP and optional test")
    parser.add_argument("--output_dir", type=str, required=True, help="LMDB output directory")
    parser.add_argument("--fs", type=int, default=250, help="Original sampling rate")
    parser.add_argument("--target_fs", type=int, default=200, help="Target sampling rate to match CBraMod patch size")
    parser.add_argument("--train_seg_pts", type=int, default=12500, help="Points per training trial")
    parser.add_argument("--train_n_seg", type=int, default=4, help="Trials per emotion in each training mat")
    parser.add_argument("--test_seg_pts", type=int, default=2500, help="Points per testing trial")
    parser.add_argument("--test_n_seg", type=int, default=8, help="Trials per public test mat")
    parser.add_argument("--sample_sec", type=int, default=4, help="Seconds per CBraMod sample")
    parser.add_argument("--sample_stride_sec", type=int, default=2, help="Stride for train sliding windows")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Subject-level validation ratio")
    parser.add_argument("--test_ratio", type=float, default=0.05, help="Subject-level held-out test ratio")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed for reproducible split")
    parser.add_argument("--map_size_gb", type=int, default=16, help="LMDB map size in GB")
    return parser.parse_args()


def _ensure_samples_first(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D EEG array, got shape={arr.shape}")
    if arr.shape[0] <= arr.shape[1]:
        arr = arr.T
    return arr


def load_mat_variable(mat_path: Path, var_name: str) -> np.ndarray:
    try:
        with h5py.File(str(mat_path), "r") as f:
            data = f[var_name][()]
            return _ensure_samples_first(np.asarray(data)).astype(np.float32)
    except Exception:
        mat = sio.loadmat(str(mat_path))
        if var_name not in mat:
            raise KeyError(f"{var_name} not found in {mat_path}")
        return _ensure_samples_first(np.asarray(mat[var_name])).astype(np.float32)


def resample_segment(x: np.ndarray, src_fs: int, dst_fs: int) -> np.ndarray:
    if src_fs == dst_fs:
        return x.astype(np.float32)
    dst_points = int(round(x.shape[0] * dst_fs / src_fs))
    return signal.resample(x, dst_points, axis=0).astype(np.float32)


def iter_sliding_windows(trial: np.ndarray, fs: int, sample_sec: int, stride_sec: int):
    sample_points = fs * sample_sec
    stride_points = fs * stride_sec
    if stride_points <= 0:
        raise ValueError(f"sample_stride_sec must be positive, got {stride_sec}")
    if trial.shape[0] < sample_points:
        return
    for win_start in range(0, trial.shape[0] - sample_points + 1, stride_points):
        yield win_start, trial[win_start:win_start + sample_points, :]


def to_cbramod_sample(segment: np.ndarray, target_fs: int, sample_sec: int) -> np.ndarray:
    sample_points = target_fs * sample_sec
    if segment.shape[0] != sample_points:
        raise ValueError(f"Expected {sample_points} points, got {segment.shape[0]}")
    sample = segment.reshape(sample_sec, target_fs, segment.shape[1]).transpose(2, 0, 1)
    return sample.astype(np.float32)


def stable_subject_split(subject_ids, val_ratio, test_ratio, seed):
    ordered = []
    for subject_id in sorted(subject_ids):
        key = hashlib.md5(f"{seed}:{subject_id}".encode()).hexdigest()
        ordered.append((key, subject_id))
    ordered.sort()
    ordered = [subject_id for _, subject_id in ordered]

    n_subjects = len(ordered)
    n_test = max(1, int(round(n_subjects * test_ratio))) if n_subjects >= 3 and test_ratio > 0 else 0
    n_val = max(1, int(round(n_subjects * val_ratio))) if n_subjects >= 3 and val_ratio > 0 else 0
    if n_test + n_val >= n_subjects:
        n_test = 1 if n_subjects >= 3 else 0
        n_val = 1 if n_subjects >= 3 else 0

    test_subjects = set(ordered[:n_test])
    val_subjects = set(ordered[n_test:n_test + n_val])
    train_subjects = set(ordered[n_test + n_val:])
    return train_subjects, val_subjects, test_subjects


def iter_train_samples(mat_path: Path, group: int, args):
    user_id = mat_path.stem
    emotion_map = {
        "EEG_data_neu": 0,
        "EEG_data_pos": 1,
    }

    for var_name, label in emotion_map.items():
        data = load_mat_variable(mat_path, var_name)
        # Minimal preprocessing on the whole continuous recording first.
        # Train mats are shaped as (train_n_seg * train_seg_pts, 30).
        data = preprocess_recording(data, fs=args.fs, seg_pts=args.train_seg_pts, n_seg=args.train_n_seg)
        for seg_idx in range(args.train_n_seg):
            seg_start = seg_idx * args.train_seg_pts
            seg_end = seg_start + args.train_seg_pts
            trial = data[seg_start:seg_end, :]
            if trial.shape[0] != args.train_seg_pts:
                continue
            window_id = 0
            for _, chunk in iter_sliding_windows(trial, args.fs, args.sample_sec, args.sample_stride_sec):
                chunk = resample_segment(chunk, args.fs, args.target_fs)
                sample = to_cbramod_sample(chunk, args.target_fs, args.sample_sec)
                key = f"{user_id}_{'dep' if group == 1 else 'hc'}_{var_name}_{seg_idx}_{window_id}"
                yield key, {
                    "sample": sample,
                    "label": float(label),
                    "subject_id": user_id,
                    "group": int(group),
                    "trial_id": int(seg_idx),
                    "window_id": int(window_id),
                }
                window_id += 1


def iter_public_test_samples(mat_path: Path, args):
    user_id = mat_path.stem
    try:
        data = load_mat_variable(mat_path, "test_eeg_c")
    except KeyError:
        log.warning("skip %s because test_eeg_c is missing", mat_path)
        return

    # Minimal preprocessing on the whole continuous recording first.
    data = preprocess_recording(data, fs=args.fs, seg_pts=args.test_seg_pts, n_seg=args.test_n_seg)

    for seg_idx in range(args.test_n_seg):
        seg_start = seg_idx * args.test_seg_pts
        seg_end = seg_start + args.test_seg_pts
        segment = data[seg_start:seg_end, :]
        if segment.shape[0] != args.test_seg_pts:
            continue
        window_id = 0
        for _, chunk in iter_sliding_windows(segment, args.fs, args.sample_sec, args.sample_stride_sec):
            chunk = resample_segment(chunk, args.fs, args.target_fs)
            sample = to_cbramod_sample(chunk, args.target_fs, args.sample_sec)
            yield sample, {
                "subject_id": user_id,
                "trial_id": seg_idx,
                "window_id": window_id,
                "sample_key": f"{user_id}_trial_{seg_idx}_{window_id}",
            }
            window_id += 1


def write_lmdb(db, key, value):
    with db.begin(write=True) as txn:
        txn.put(key=key.encode(), value=pickle.dumps(value))


def build_train_lmdb(args):
    root = Path(args.project_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    db = lmdb.open(str(output_dir), map_size=args.map_size_gb * (1024 ** 3))

    subject_entries = {}
    for group_dir, group_label in [("train/HC", 0), ("train/DEP", 1)]:
        for mat_path in sorted((root / group_dir).glob("*.mat")):
            log.info(f"processing {mat_path}")
            for key, record in iter_train_samples(mat_path, group_label, args):
                write_lmdb(db, key, record)
                subject_entries.setdefault(record["subject_id"], []).append(key)

    train_subjects, val_subjects, test_subjects = stable_subject_split(
        subject_entries.keys(), args.val_ratio, args.test_ratio, args.seed
    )

    split_keys = {"train": [], "val": [], "test": []}
    for subject_id, keys in subject_entries.items():
        if subject_id in train_subjects:
            split_keys["train"].extend(keys)
        elif subject_id in val_subjects:
            split_keys["val"].extend(keys)
        else:
            split_keys["test"].extend(keys)

    write_lmdb(db, "__keys__", split_keys)
    db.close()

    log.info(
        "LMDB built: train=%d val=%d test=%d",
        len(split_keys["train"]),
        len(split_keys["val"]),
        len(split_keys["test"]),
    )


def export_public_test_preview(args):
    root = Path(args.project_root)
    test_dir = root / "test"
    if not test_dir.exists():
        return

    output_path = Path(args.output_dir).parent / "public_test_samples.npy"
    manifest_path = Path(args.output_dir).parent / "public_test_manifest.pkl"

    sample_list = []
    meta_list = []
    for mat_path in sorted(test_dir.glob("*.mat")):
        for sample, meta in iter_public_test_samples(mat_path, args):
            sample_list.append(sample)
            meta_list.append(meta)

    if sample_list:
        np.save(output_path, np.stack(sample_list).astype(np.float32))
        with open(manifest_path, "wb") as f:
            pickle.dump(meta_list, f)
        log.info("public test preview saved: %s and %s", output_path, manifest_path)


def main():
    args = parse_args()
    log.info("building CBraMod emotion LMDB from %s", args.project_root)
    build_train_lmdb(args)
    export_public_test_preview(args)
    log.info("done")


if __name__ == "__main__":
    main()

"""
Emotion EEG preprocessing for CBraMod

Input
- train/HC/*.mat
- train/DEP/*.mat
- test/*.mat (optional, unlabeled)

Output
- LMDB database with train/val/test splits
- Each sample shape: (30, 10, 200)

Preprocessing notes
-------------------
* The raw .mat files are 4 (or 8) concatenated but *discontinuous* video-trial
  segments.  Filtering across segment boundaries would cause edge artifacts
  (cross-contamination between adjacent trials).  We therefore
      1. split the raw array into individual trials FIRST,
      2. build a separate mne.io.RawArray for each trial, and
      3. run the full preprocessing pipeline (bandpass + bad-channel
         interpolation + ICA) on each trial independently.
* The dataset documentation states that baseline drift (0.01 Hz HP) and mains
  noise (50 Hz notch) have already been removed during acquisition.  We
  therefore do NOT apply a 50 Hz notch filter here.
"""

import argparse
import hashlib
import logging
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
except Exception:
    mne = None
    ICA = None
    _HAS_MNE = False


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Preprocessing defaults
# ---------------------------------------------------------------------------
# Key changes vs. original:
#   * notch_freqs removed entirely  – already filtered by the acquisition
#     system (per dataset documentation).
#   * Preprocessing is now applied per-trial, not per-recording, to avoid
#     cross-trial edge artifacts at segment boundaries.
# ---------------------------------------------------------------------------
EEG_MINPREP_DEFAULTS = {
    "enabled": True,
    # notch_freqs intentionally omitted:
    #   documentation confirms 50 Hz mains noise was removed at acquisition.
    "bp_l": None,
    "bp_h": 47.0,
    "filter_method": "fir",   # zero-phase FIR (MNE default when phase="zero")
    "filter_phase": "zero",
    "mad_k": 3.0,
    "bad_ratio": 0.30,        # >30 % outliers → bad channel
    "eps": 1e-12,
    "ica_n_components": 0.99, # keep 99 % variance
    "ica_method": "fastica",
    "ica_random_state": 97,
    "ica_fit_l": 1.0,         # higher HP for ICA stability
    "ica_fit_h": 47.0,
    "ica_eog_threshold": 3.0,
}

# ---------------------------------------------------------------------------
# Channel metadata
# ---------------------------------------------------------------------------
CHS_ORIG = [
    "FP1", "FP2", "F7",  "F3",  "FZ",  "F4",  "F8",
    "FT7", "FC3", "FCZ", "FC4", "FT8",
    "T3",  "C3",  "CZ",  "C4",  "T4",
    "TP7", "CP3", "CPZ", "CP4", "TP8",
    "T5",  "P3",  "PZ",  "P4",  "T6",
    "O1",  "OZ",  "O2",
]

RENAME_FOR_MONTAGE = {
    "FP1": "Fp1", "FP2": "Fp2",
    "FZ":  "Fz",  "FCZ": "FCz",
    "CZ":  "Cz",  "CPZ": "CPz",
    "PZ":  "Pz",  "OZ":  "Oz",
    "T3":  "T7",  "T4":  "T8",
    "T5":  "P7",  "T6":  "P8",
}


def _to_mne_name(ch: str) -> str:
    return RENAME_FOR_MONTAGE.get(ch, ch)


CHS_MNE = [_to_mne_name(c) for c in CHS_ORIG]


# ---------------------------------------------------------------------------
# MNE helpers
# ---------------------------------------------------------------------------

def _make_raw_eeg(data_samp_ch: np.ndarray, sfreq: float) -> "mne.io.Raw":
    """
    Build an mne.io.RawArray from a (n_samples, n_channels) array.
    """
    if not _HAS_MNE:
        raise RuntimeError("MNE is not available.")
    if data_samp_ch.ndim != 2:
        raise ValueError(f"Expected 2-D array, got shape {data_samp_ch.shape}")
    if data_samp_ch.shape[1] != len(CHS_MNE):
        raise ValueError(
            f"Expected {len(CHS_MNE)} channels, got {data_samp_ch.shape[1]}"
        )

    info = mne.create_info(
        ch_names=CHS_MNE,
        sfreq=float(sfreq),
        ch_types=["eeg"] * len(CHS_MNE),
    )
    # MNE stores data as (n_channels, n_times)
    raw = mne.io.RawArray(data_samp_ch.T, info, verbose=False)
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, on_missing="warn", verbose=False)
    return raw


def _mad_outlier_ratio(x: np.ndarray, k: float, eps: float) -> float:
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + eps
    return float((np.abs(x - med) > k * mad).mean())


def detect_bads_mad(raw: "mne.io.Raw", cfg: dict) -> list[str]:
    """
    Mark a channel as bad if its MAD-outlier ratio exceeds the threshold
    on *any* 1-second epoch of the trial.

    For short trials (< 2 s) we fall back to evaluating the whole trial
    as a single segment so that the function is always well-defined.
    """
    data = raw.get_data(picks="eeg")          # (n_ch, n_times)
    n_times  = data.shape[1]
    seg_pts  = int(raw.info["sfreq"])         # 1-second epochs
    n_seg    = max(1, n_times // seg_pts)

    bads: set[str] = set()
    for si in range(n_seg):
        s   = si * seg_pts
        e   = min(s + seg_pts, n_times)
        seg = data[:, s:e]
        for ci, ch in enumerate(raw.ch_names):
            if _mad_outlier_ratio(seg[ci], cfg["mad_k"], cfg["eps"]) > cfg["bad_ratio"]:
                bads.add(ch)
    return sorted(bads)


def add_eog_proxy_channel(raw: "mne.io.Raw") -> "mne.io.Raw":
    """
    Append a synthetic EOG channel = mean(Fp1, Fp2) for ICA artefact detection.
    """
    picks = [ch for ch in ("Fp1", "Fp2") if ch in raw.ch_names]
    if not picks:
        raise RuntimeError("Cannot build EOG proxy: Fp1 and Fp2 are both absent.")

    eog_data = raw.get_data(picks=picks).mean(axis=0, keepdims=True)
    eog_info = mne.create_info(["EOG"], sfreq=float(raw.info["sfreq"]), ch_types=["eog"])
    raw_eog  = mne.io.RawArray(eog_data, eog_info, verbose=False)

    raw2 = raw.copy()
    raw2.add_channels([raw_eog], force_update_info=True)
    return raw2


def run_ica_eog(raw: "mne.io.Raw", cfg: dict) -> "mne.io.Raw":
    """
    Fit ICA on a 1–47 Hz copy; apply to `raw`; return cleaned copy without EOG.
    """
    raw_with_eog = add_eog_proxy_channel(raw)

    raw_fit = raw_with_eog.copy().filter(
        l_freq=float(cfg["ica_fit_l"]),
        h_freq=float(cfg["ica_fit_h"]),
        method=cfg["filter_method"],
        phase=cfg["filter_phase"],
        picks="eeg",
        verbose=False,
    )

    ica = ICA(
        n_components=cfg["ica_n_components"],
        method=cfg["ica_method"],
        random_state=int(cfg["ica_random_state"]),
        max_iter="auto",
        verbose=False,
    )
    ica.fit(raw_fit, picks="eeg")

    eog_inds, _ = ica.find_bads_eog(
        raw_fit,
        ch_name="EOG",
        threshold=float(cfg["ica_eog_threshold"]),
        verbose=False,
    )
    ica.exclude = eog_inds

    raw_clean = raw_with_eog.copy()
    ica.apply(raw_clean, verbose=False)
    raw_clean.drop_channels(["EOG"])
    return raw_clean


def preprocess_trial(trial_samp_ch: np.ndarray, fs: int, cfg: dict | None = None) -> np.ndarray:
    """
    Run the full preprocessing pipeline on a **single, continuous trial**.

    Parameters
    ----------
    trial_samp_ch : ndarray, shape (n_samples, n_channels)
        Raw EEG for one trial.
    fs : int
        Sampling frequency of the input data.
    cfg : dict or None
        Overrides for EEG_MINPREP_DEFAULTS.  Pass ``{"enabled": False}`` to
        return the raw data unchanged.

    Returns
    -------
    ndarray, shape (n_samples, n_channels), dtype float32
        Cleaned EEG.

    Notes
    -----
    This function intentionally processes one trial at a time so that the
    FIR filter transients (edge artifacts) cannot bleed across the
    boundaries between discontinuous video segments.

    No 50 Hz notch filter is applied because the dataset documentation
    states that mains interference was already removed during acquisition.
    """
    cfg = dict(EEG_MINPREP_DEFAULTS if cfg is None else cfg)

    if not cfg.get("enabled", True):
        return trial_samp_ch.astype(np.float32, copy=False)

    if not _HAS_MNE:
        log.warning(
            "MNE not installed – skipping bandpass / bad-channel / ICA preprocessing."
        )
        return trial_samp_ch.astype(np.float32, copy=False)

    # --- build Raw object for this single trial ---
    raw = _make_raw_eeg(trial_samp_ch.astype(np.float64, copy=False), sfreq=float(fs))

    # --- bandpass only (NO notch – already done at acquisition) ---
    raw.filter(
        l_freq=float(cfg["bp_l"]),
        h_freq=float(cfg["bp_h"]),
        method=cfg["filter_method"],
        phase=cfg["filter_phase"],
        picks="eeg",
        verbose=False,
    )

    # --- bad-channel detection & interpolation ---
    bads = detect_bads_mad(raw, cfg)
    if bads:
        log.info("   bad channels (trial-level MAD): %s", bads)
        raw.info["bads"] = bads
        try:
            raw.interpolate_bads(reset_bads=True, verbose=False)
        except Exception as exc:
            log.warning("   interpolate_bads failed: %r – continuing without.", exc)
            raw.info["bads"] = []

    # --- ICA (EOG artefact removal) ---
    try:
        raw = run_ica_eog(raw, cfg)
        log.info("   ICA(EOG) applied.")
    except Exception as exc:
        log.warning("   ICA failed: %r – continuing without.", exc)

    # (n_channels, n_times).T → (n_times, n_channels)
    return raw.get_data(picks="eeg").T.astype(np.float32)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Preprocess emotion EEG .mat files into a CBraMod LMDB."
    )
    p.add_argument("--project_root",      required=True,
                   help="Dataset root (contains train/HC, train/DEP, optionally test/)")
    p.add_argument("--output_dir",        required=True,
                   help="Directory for the output LMDB")
    p.add_argument("--fs",                type=int, default=250,
                   help="Original sampling rate (Hz)")
    p.add_argument("--target_fs",         type=int, default=200,
                   help="Target sampling rate to match CBraMod patch size")
    p.add_argument("--train_seg_pts",     type=int, default=12500,
                   help="Samples per training trial (one video segment)")
    p.add_argument("--train_n_seg",       type=int, default=4,
                   help="Number of trials per emotion in each training .mat")
    p.add_argument("--test_seg_pts",      type=int, default=2500,
                   help="Samples per public-test trial")
    p.add_argument("--test_n_seg",        type=int, default=8,
                   help="Number of trials per public-test .mat")
    p.add_argument("--sample_sec",        type=int, default=4,
                   help="Seconds per CBraMod sample window")
    p.add_argument("--sample_stride_sec", type=int, default=2,
                   help="Sliding-window stride for training samples (seconds)")
    p.add_argument("--val_ratio",         type=float, default=0.1,
                   help="Subject-level validation fraction")
    p.add_argument("--test_ratio",        type=float, default=0.05,
                   help="Subject-level held-out test fraction")
    p.add_argument("--seed",              type=int, default=3407,
                   help="Random seed for reproducible split")
    p.add_argument("--map_size_gb",       type=int, default=16,
                   help="LMDB map size in GB")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _ensure_samples_first(arr: np.ndarray) -> np.ndarray:
    """Guarantee shape is (n_samples, n_channels) where n_samples > n_channels."""
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D array, got shape {arr.shape}")
    if arr.shape[0] <= arr.shape[1]:
        arr = arr.T
    return arr


def load_mat_variable(mat_path: Path, var_name: str) -> np.ndarray:
    try:
        with h5py.File(str(mat_path), "r") as f:
            data = np.asarray(f[var_name][()])
            return _ensure_samples_first(data).astype(np.float32)
    except Exception:
        mat = sio.loadmat(str(mat_path))
        if var_name not in mat:
            raise KeyError(f"Variable '{var_name}' not found in {mat_path}")
        return _ensure_samples_first(np.asarray(mat[var_name])).astype(np.float32)


# ---------------------------------------------------------------------------
# Signal processing helpers
# ---------------------------------------------------------------------------

def resample_segment(x: np.ndarray, src_fs: int, dst_fs: int) -> np.ndarray:
    if src_fs == dst_fs:
        return x.astype(np.float32)
    dst_pts = int(round(x.shape[0] * dst_fs / src_fs))
    return signal.resample(x, dst_pts, axis=0).astype(np.float32)


def iter_sliding_windows(trial: np.ndarray, fs: int, sample_sec: int, stride_sec: int):
    """Yield (start_sample, window_array) pairs for one trial."""
    win_pts    = fs * sample_sec
    stride_pts = fs * stride_sec
    if stride_pts <= 0:
        raise ValueError(f"sample_stride_sec must be > 0, got {stride_sec}")
    if trial.shape[0] < win_pts:
        return
    for start in range(0, trial.shape[0] - win_pts + 1, stride_pts):
        yield start, trial[start:start + win_pts, :]


def to_cbramod_sample(segment: np.ndarray, target_fs: int, sample_sec: int) -> np.ndarray:
    """Reshape (target_fs*sample_sec, 30) → (30, sample_sec, target_fs)."""
    expected = target_fs * sample_sec
    if segment.shape[0] != expected:
        raise ValueError(f"Expected {expected} points, got {segment.shape[0]}")
    # (time, ch) → (ch, sec, fs_per_sec)
    return segment.reshape(sample_sec, target_fs, segment.shape[1]).transpose(2, 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# Subject-level train/val/test split (deterministic)
# ---------------------------------------------------------------------------

def stable_subject_split(subject_ids, val_ratio, test_ratio, seed):
    ordered = sorted(
        subject_ids,
        key=lambda sid: hashlib.md5(f"{seed}:{sid}".encode()).hexdigest(),
    )
    n = len(ordered)
    n_test = max(1, round(n * test_ratio)) if n >= 3 and test_ratio > 0 else 0
    n_val  = max(1, round(n * val_ratio))  if n >= 3 and val_ratio  > 0 else 0
    if n_test + n_val >= n:
        n_test, n_val = (1, 1) if n >= 3 else (0, 0)

    test_subjects  = set(ordered[:n_test])
    val_subjects   = set(ordered[n_test:n_test + n_val])
    train_subjects = set(ordered[n_test + n_val:])
    return train_subjects, val_subjects, test_subjects


# ---------------------------------------------------------------------------
# Sample iterators  (FIXED: preprocess per trial, not per recording)
# ---------------------------------------------------------------------------

def iter_train_samples(mat_path: Path, group: int, args):
    """
    Yield (key, record) pairs for every sliding window in every trial.

    Critically, preprocessing is applied **per trial** (after slicing the
    raw concatenated array) so that FIR edge artifacts cannot cross segment
    boundaries.
    """
    user_id = mat_path.stem
    emotion_map = {"EEG_data_neu": 0, "EEG_data_pos": 1}

    for var_name, label in emotion_map.items():
        # Load the raw concatenated array (train_n_seg * train_seg_pts, 30).
        raw_data = load_mat_variable(mat_path, var_name)

        for seg_idx in range(args.train_n_seg):
            seg_start = seg_idx * args.train_seg_pts
            seg_end   = seg_start + args.train_seg_pts

            # --- slice FIRST, preprocess AFTER ---
            trial_raw = raw_data[seg_start:seg_end, :]
            if trial_raw.shape[0] != args.train_seg_pts:
                log.warning(
                    "  %s / %s / trial %d: expected %d pts, got %d – skipping.",
                    mat_path.name, var_name, seg_idx,
                    args.train_seg_pts, trial_raw.shape[0],
                )
                continue

            log.info(
                "  preprocessing %s | %s | trial %d/%d",
                user_id, var_name, seg_idx + 1, args.train_n_seg,
            )
            trial = preprocess_trial(trial_raw, fs=args.fs)

            window_id = 0
            for _, chunk in iter_sliding_windows(
                trial, args.fs, args.sample_sec, args.sample_stride_sec
            ):
                chunk  = resample_segment(chunk, args.fs, args.target_fs)
                sample = to_cbramod_sample(chunk, args.target_fs, args.sample_sec)
                key    = (
                    f"{user_id}_{'dep' if group == 1 else 'hc'}"
                    f"_{var_name}_{seg_idx}_{window_id}"
                )
                yield key, {
                    "sample":     sample,
                    "label":      float(label),
                    "subject_id": user_id,
                    "group":      int(group),
                    "trial_id":   int(seg_idx),
                    "window_id":  int(window_id),
                }
                window_id += 1


def iter_public_test_samples(mat_path: Path, args):
    """
    Yield (sample, meta) pairs for the public test set.

    Same fix as iter_train_samples: slice first, preprocess per trial.
    """
    user_id = mat_path.stem

    try:
        raw_data = load_mat_variable(mat_path, "test_eeg_c")
    except KeyError:
        log.warning("Skipping %s – variable 'test_eeg_c' not found.", mat_path)
        return

    for seg_idx in range(args.test_n_seg):
        seg_start = seg_idx * args.test_seg_pts
        seg_end   = seg_start + args.test_seg_pts

        # --- slice FIRST, preprocess AFTER ---
        trial_raw = raw_data[seg_start:seg_end, :]
        if trial_raw.shape[0] != args.test_seg_pts:
            log.warning(
                "  %s / trial %d: expected %d pts, got %d – skipping.",
                mat_path.name, seg_idx, args.test_seg_pts, trial_raw.shape[0],
            )
            continue

        log.info(
            "  preprocessing %s | trial %d/%d",
            user_id, seg_idx + 1, args.test_n_seg,
        )
        trial = preprocess_trial(trial_raw, fs=args.fs)

        window_id = 0
        for _, chunk in iter_sliding_windows(
            trial, args.fs, args.sample_sec, args.sample_stride_sec
        ):
            chunk  = resample_segment(chunk, args.fs, args.target_fs)
            sample = to_cbramod_sample(chunk, args.target_fs, args.sample_sec)
            yield sample, {
                "subject_id": user_id,
                "trial_id":   seg_idx,
                "window_id":  window_id,
                "sample_key": f"{user_id}_trial_{seg_idx}_{window_id}",
            }
            window_id += 1


# ---------------------------------------------------------------------------
# LMDB I/O
# ---------------------------------------------------------------------------

def write_lmdb(db: lmdb.Environment, key: str, value) -> None:
    with db.begin(write=True) as txn:
        txn.put(key.encode(), pickle.dumps(value))


def build_train_lmdb(args) -> None:
    root       = Path(args.project_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    db = lmdb.open(str(output_dir), map_size=args.map_size_gb * (1024 ** 3))

    subject_entries: dict[str, list[str]] = {}
    for group_dir, group_label in [("train/HC", 0), ("train/DEP", 1)]:
        for mat_path in sorted((root / group_dir).glob("*.mat")):
            log.info("Processing %s", mat_path)
            for key, record in iter_train_samples(mat_path, group_label, args):
                write_lmdb(db, key, record)
                subject_entries.setdefault(record["subject_id"], []).append(key)

    train_subj, val_subj, test_subj = stable_subject_split(
        subject_entries.keys(), args.val_ratio, args.test_ratio, args.seed
    )

    split_keys: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for sid, keys in subject_entries.items():
        if sid in train_subj:
            split_keys["train"].extend(keys)
        elif sid in val_subj:
            split_keys["val"].extend(keys)
        else:
            split_keys["test"].extend(keys)

    write_lmdb(db, "__keys__", split_keys)
    db.close()

    log.info(
        "LMDB built – train: %d  val: %d  test: %d",
        len(split_keys["train"]),
        len(split_keys["val"]),
        len(split_keys["test"]),
    )


def export_public_test_preview(args) -> None:
    test_dir = Path(args.project_root) / "test"
    if not test_dir.exists():
        return

    out_root      = Path(args.output_dir).parent
    samples_path  = out_root / "public_test_samples.npy"
    manifest_path = out_root / "public_test_manifest.pkl"

    sample_list, meta_list = [], []
    for mat_path in sorted(test_dir.glob("*.mat")):
        for sample, meta in iter_public_test_samples(mat_path, args):
            sample_list.append(sample)
            meta_list.append(meta)

    if sample_list:
        np.save(samples_path, np.stack(sample_list).astype(np.float32))
        with open(manifest_path, "wb") as f:
            pickle.dump(meta_list, f)
        log.info("Public test preview → %s, %s", samples_path, manifest_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    log.info("Building CBraMod emotion LMDB from %s", args.project_root)
    build_train_lmdb(args)
    export_public_test_preview(args)
    log.info("Done.")


if __name__ == "__main__":
    main()
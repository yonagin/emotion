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


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


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
    parser.add_argument("--sample_sec", type=int, default=10, help="Seconds per CBraMod sample")
    parser.add_argument("--sample_stride_sec", type=int, default=5, help="Stride for train sliding windows")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Subject-level validation ratio")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Subject-level held-out test ratio")
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
    sample_points = args.fs * args.sample_sec
    stride_points = args.fs * args.sample_stride_sec

    for var_name, label in emotion_map.items():
        data = load_mat_variable(mat_path, var_name)
        for seg_idx in range(args.train_n_seg):
            seg_start = seg_idx * args.train_seg_pts
            seg_end = seg_start + args.train_seg_pts
            trial = data[seg_start:seg_end, :]
            if trial.shape[0] != args.train_seg_pts:
                continue
            window_id = 0
            for win_start in range(0, args.train_seg_pts - sample_points + 1, stride_points):
                chunk = trial[win_start:win_start + sample_points, :]
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
        user_id = mat_path.stem
        try:
            data = load_mat_variable(mat_path, "test_eeg_c")
        except KeyError:
            log.warning("skip %s because test_eeg_c is missing", mat_path)
            continue
        for seg_idx in range(args.test_n_seg):
            seg_start = seg_idx * args.test_seg_pts
            seg_end = seg_start + args.test_seg_pts
            segment = data[seg_start:seg_end, :]
            if segment.shape[0] != args.test_seg_pts:
                continue
            segment = resample_segment(segment, args.fs, args.target_fs)
            sample = to_cbramod_sample(segment, args.target_fs, args.sample_sec)
            sample_list.append(sample)
            meta_list.append(
                {
                    "subject_id": user_id,
                    "trial_id": seg_idx,
                    "sample_key": f"{user_id}_trial_{seg_idx}",
                }
            )

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

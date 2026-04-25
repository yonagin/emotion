import argparse
from collections import defaultdict
from pathlib import Path
import pickle
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from train_dsainet_faced import DSAINetAdapter, setup_seed, str2bool

CHANNEL_MAPPING = [0, 1, 2, 3, 4, 5, 6, 12, 13, 14, 15, 16, 22, 23, 24, 25, 26]


class NpyDataset(Dataset):
    def __init__(self, npy_path: str):
        self.samples = np.load(npy_path)
        if self.samples.ndim != 4:
            raise ValueError(
                f"Expected npy shape (N, C, S, P), but got {self.samples.shape}"
            )
        max_channel = max(CHANNEL_MAPPING)
        if self.samples.shape[1] <= max_channel:
            raise ValueError(
                f"Input npy only has {self.samples.shape[1]} channels, "
                f"but channel mapping requires index {max_channel}"
            )
        self.samples = self.samples[:, CHANNEL_MAPPING, :, :]

    def __len__(self):
        return int(self.samples.shape[0])

    def __getitem__(self, idx):
        sample = self.samples[idx].astype(np.float32)
        return torch.from_numpy(sample)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run DSAINet inference on public test npy and export Excel."
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Folder containing public_test_samples.npy and public_test_manifest.pkl",
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained .pth model")
    parser.add_argument("--output", type=str, required=True, help="Path to output .xlsx file")

    parser.add_argument("--seed", type=int, default=3407, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=16, help="Inference batch size")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device, e.g. cuda:0 or cpu")
    parser.add_argument(
        "--trial_id_offset",
        type=int,
        default=1,
        help="Added to manifest trial_id before export; default 1 matches the required 1-based format",
    )

    parser.add_argument("--num_of_classes", type=int, default=2)
    parser.add_argument("--chans", type=int, default=len(CHANNEL_MAPPING), help="number of channels after mapping")
    parser.add_argument("--samples", type=int, default=2000, help="time length after reshape (10*200=2000)")

    # DSAINet hparams (defaults match models/DSAINet.py)
    parser.add_argument("--emb_size", type=int, default=40)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--attn_depth", type=int, default=1)
    parser.add_argument("--attn_dropout", type=float, default=0.25)

    parser.add_argument("--eeg1_f1", type=int, default=16)
    parser.add_argument("--eeg1_kernel_size", type=int, default=64)
    parser.add_argument("--eeg1_D", type=int, default=2)
    parser.add_argument("--eeg1_pooling_size1", type=int, default=4)
    parser.add_argument("--eeg1_pooling_size2", type=int, default=8)
    parser.add_argument("--eeg1_dropout_rate", type=float, default=0.25)

    parser.add_argument("--branch_1_kernels", type=int, nargs="*", default=[11, 15], help="ConvTime kernels for branch 1")
    parser.add_argument("--branch_2_kernels", type=int, nargs="*", default=[3, 7], help="ConvTime kernels for branch 2")
    parser.add_argument("--conv_expansion", type=int, default=4)
    parser.add_argument("--conv_dropout", type=float, default=0.25)

    parser.add_argument("--intra_ffn_expansion", type=int, default=2)
    parser.add_argument("--inter_ffn_expansion", type=int, default=2)

    parser.add_argument("--big_residual", type=str2bool, default=True)
    parser.add_argument("--big_residual_learnable", type=str2bool, default=True)
    parser.add_argument("--dropout", type=float, default=0.25)

    return parser.parse_args()


def load_manifest(manifest_pkl: str):
    with open(manifest_pkl, "rb") as file:
        manifest = pickle.load(file)
    if not isinstance(manifest, list):
        raise ValueError("manifest_pkl must contain a list of metadata dicts")
    return manifest


def resolve_input_paths(input_dir: str):
    input_root = Path(input_dir)
    if not input_root.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_root}")
    if not input_root.is_dir():
        raise NotADirectoryError(f"Input path is not a folder: {input_root}")

    npy_path = input_root / "public_test_samples.npy"
    manifest_path = input_root / "public_test_manifest.pkl"

    if not npy_path.exists():
        raise FileNotFoundError(f"Missing file: {npy_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing file: {manifest_path}")

    return npy_path, manifest_path


def run_inference(model, data_loader, device):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
    if not all_probs:
        return np.empty((0, 2), dtype=np.float32)
    return np.concatenate(all_probs, axis=0)


def aggregate_trial_predictions(probabilities, manifest, trial_id_offset):
    if len(probabilities) != len(manifest):
        raise ValueError(
            f"Number of predictions ({len(probabilities)}) does not match "
            f"manifest entries ({len(manifest)})"
        )

    grouped = defaultdict(list)
    for prob, meta in zip(probabilities, manifest):
        user_id = meta["user_id"]
        trial_id = int(meta["trial_id"]) + trial_id_offset
        grouped[(user_id, trial_id)].append(prob)

    rows = []
    for (user_id, trial_id) in sorted(grouped.keys(), key=lambda item: (natural_user_key(item[0]), item[1])):
        mean_prob = np.mean(np.stack(grouped[(user_id, trial_id)], axis=0), axis=0)
        pred_label = int(np.argmax(mean_prob))
        rows.append(
            {
                "user_id": user_id,
                "trial_id": trial_id,
                "Emotion_label": pred_label,
            }
        )
    return rows


def natural_user_key(user_id: str):
    match = re.match(r"^(.*?)(\d+)$", user_id)
    if match:
        prefix, number = match.groups()
        return prefix, int(number)
    return user_id, -1


def summarize_manifest(manifest):
    trial_counts = defaultdict(set)
    window_counts = defaultdict(int)

    for meta in manifest:
        user_id = meta["user_id"]
        trial_id = int(meta["trial_id"])
        trial_counts[user_id].add(trial_id)
        window_counts[(user_id, trial_id)] += 1

    unique_users = sorted(trial_counts.keys(), key=natural_user_key)
    unique_trials = sorted({trial_id for trials in trial_counts.values() for trial_id in trials})
    windows_per_trial = sorted(set(window_counts.values()))

    print(f"[INFO] manifest samples: {len(manifest)}")
    print(f"[INFO] unique users: {len(unique_users)} -> {unique_users}")
    print(f"[INFO] unique trial ids: {len(unique_trials)} -> {[trial_id + 1 for trial_id in unique_trials]}")
    print(
        f"[INFO] trials per user: "
        f"{sorted(((user_id, len(trials)) for user_id, trials in trial_counts.items()), key=lambda item: natural_user_key(item[0]))}"
    )
    print(f"[INFO] windows per trial: {windows_per_trial}")

    return {
        "n_users": len(unique_users),
        "n_trials": len(unique_trials),
        "windows_per_trial": windows_per_trial,
    }


def main():
    args = parse_args()
    setup_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or str(args.device) == "cpu" else "cpu")
    npy_path, manifest_path = resolve_input_paths(args.input_dir)

    dataset = NpyDataset(str(npy_path))
    manifest = load_manifest(str(manifest_path))
    if len(dataset) != len(manifest):
        raise ValueError(
            f"Sample count in npy ({len(dataset)}) does not match manifest count ({len(manifest)})"
        )

    args.chans = int(dataset.samples.shape[1])
    args.samples = int(dataset.samples.shape[2] * dataset.samples.shape[3])
    manifest_summary = summarize_manifest(manifest)

    print(f"[INFO] raw npy shape: {np.load(str(npy_path), mmap_mode='r').shape}")
    print(f"[INFO] mapped npy shape: {dataset.samples.shape}")
    print(f"[INFO] model input chans: {args.chans}")
    print(f"[INFO] model input samples: {args.samples}")

    if manifest_summary["n_users"] != 10:
        print(f"[WARN] expected 10 users, got {manifest_summary['n_users']}")
    if manifest_summary["n_trials"] != 8:
        print(f"[WARN] expected 8 trial ids, got {manifest_summary['n_trials']}")

    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = DSAINetAdapter(args).to(device)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)

    probabilities = run_inference(model, data_loader, device)
    print(f"[INFO] probability array shape: {probabilities.shape}")
    rows = aggregate_trial_predictions(probabilities, manifest, args.trial_id_offset)

    df = pd.DataFrame(rows, columns=["user_id", "trial_id", "Emotion_label"])
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, index=False)

    print(f"Saved predictions to: {output_path}")


if __name__ == "__main__":
    main()

"""
Emotion EEG: .mat → LMDB (CBraMod format)

Input layout:
    train/HC/*.mat   (variables: EEG_data_neu, EEG_data_pos)
    train/DEP/*.mat
    test/*.mat       (variable: test_eeg_c, optional)

Output:
    LMDB with keys → {sample, label, user_id, group, trial_id, window_id}
    __keys__        → {"train": [...], "val": [...], "test": [...]}
    public_test_samples.npy  (optional)
    public_test_manifest.pkl (optional)

Each sample shape: (30, 4, 200)  → (n_channels, n_seconds, samples_per_sec)

Z-score normalization:
    - 两阶段：先收集全部数据（训练集+公开测试集）的所有受试者原始数据，
      在整个数据集范围内，对每个受试者按通道计算 μ/σ
    - 再用该统计量对所有样本做标准化
    - 消除受试者个体差异、分段差异、时间基线漂移
"""

import hashlib
import logging
import pickle
from pathlib import Path
from collections import defaultdict

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

# ---------------------------------------------------------------------------
# 硬编码配置
# ---------------------------------------------------------------------------
CFG = {
    # 路径
    "project_root": "/vq/emotion/raw_data",
    "output_dir":   "/vq/emotion/data",

    # 采样率
    "fs":        250,
    "target_fs": 200,

    # 训练集分段参数
    "train_seg_pts": 12500,
    "train_n_seg":   4,

    # 公开测试集分段参数
    "test_seg_pts": 2500,
    "test_n_seg":   8,

    # 滑窗参数
    "sample_sec":        10,
    "sample_stride_sec": 5,

    # 受试者级别划分比例
    "val_ratio":  0.15,
    "test_ratio": 0.05,
    "seed":       3407,

    # LMDB
    "map_size_gb": 16,

    # Z-score 标准化
    "znorm_eps": 1e-8,
}

# 情绪变量名 → 标签
EMOTION_MAP = {"EEG_data_neu": 0, "EEG_data_pos": 1}


# ---------------------------------------------------------------------------
# MAT 文件加载
# ---------------------------------------------------------------------------

def _ensure_samples_first(arr: np.ndarray) -> np.ndarray:
    """保证形状为 (n_samples, n_channels)，n_samples > n_channels。"""
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D array, got shape {arr.shape}")
    if arr.shape[0] <= arr.shape[1]:
        arr = arr.T
    return arr


def load_mat_variable(mat_path: Path, var_name: str) -> np.ndarray:
    """从 .mat 文件中加载指定变量，返回 (n_samples, n_channels) float32。"""
    try:
        with h5py.File(str(mat_path), "r") as f:
            data = np.asarray(f[var_name][()])
        return _ensure_samples_first(data).astype(np.float32)
    except Exception:
        pass

    mat = sio.loadmat(str(mat_path))
    if var_name not in mat:
        raise KeyError(f"Variable '{var_name}' not found in {mat_path}")
    return _ensure_samples_first(np.asarray(mat[var_name])).astype(np.float32)


# ---------------------------------------------------------------------------
# 信号处理工具
# ---------------------------------------------------------------------------

def resample_segment(x: np.ndarray, src_fs: int, dst_fs: int) -> np.ndarray:
    """对 (n_samples, n_channels) 数组做重采样。"""
    if src_fs == dst_fs:
        return x.astype(np.float32)
    dst_pts = int(round(x.shape[0] * dst_fs / src_fs))
    return signal.resample(x, dst_pts, axis=0).astype(np.float32)


def iter_sliding_windows(trial: np.ndarray, fs: int, sample_sec: int, stride_sec: int):
    """在单个 trial 上做滑窗，yield (start, window) 对。"""
    win_pts    = fs * sample_sec
    stride_pts = fs * stride_sec
    if trial.shape[0] < win_pts:
        return
    for start in range(0, trial.shape[0] - win_pts + 1, stride_pts):
        yield start, trial[start : start + win_pts, :]


def to_cbramod_sample(segment: np.ndarray, target_fs: int, sample_sec: int) -> np.ndarray:
    """
    将 (target_fs * sample_sec, n_ch) reshape 为 CBraMod 格式 (n_ch, sample_sec, target_fs)。
    例如: (2000, 30) → (30, 10, 200)
    """
    expected = target_fs * sample_sec
    if segment.shape[0] != expected:
        raise ValueError(f"Expected {expected} pts, got {segment.shape[0]}")
    n_ch = segment.shape[1]
    return segment.reshape(sample_sec, target_fs, n_ch).transpose(2, 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# Z-score 标准化
# ---------------------------------------------------------------------------

def compute_subject_stats(
    all_trials: list,
    eps: float = 1e-8,
) -> tuple:
    """
    计算受试者级别的逐通道 Z-score 统计量。

    Parameters
    ----------
    all_trials : list of ndarray，每个元素形状 (n_pts, n_channels)
                 包含该受试者【所有来源】的原始数据（训练+测试均纳入）
    eps        : 防止除零的最小 sigma 值

    Returns
    -------
    mu    : (1, n_channels) float64
    sigma : (1, n_channels) float64

    关键修正
    --------
    统计量必须在「整个数据集」（训练集+公开测试集）的所有 trial 上联合计算，
    而非仅在当前子集内计算，否则训练集与测试集的归一化基准不一致。
    本函数被调用时，调用方需确保 all_trials 已包含该受试者在两个子集中的
    全部原始 trial 数据。
    """
    concat = np.concatenate(all_trials, axis=0)          # (total_pts, n_ch)
    mu     = concat.mean(axis=0, keepdims=True)           # (1, n_ch)
    sigma  = concat.std( axis=0, keepdims=True)           # (1, n_ch)
    sigma  = np.where(sigma < eps, eps, sigma)
    return mu.astype(np.float64), sigma.astype(np.float64)


def znorm(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    对单段数据做 Z-score 标准化。

    Parameters
    ----------
    x     : (n_pts, n_ch)
    mu    : (1,    n_ch)
    sigma : (1,    n_ch)
    """
    return np.clip(((x - mu) / sigma).astype(np.float32), -3.0, 3.0)


# ---------------------------------------------------------------------------
# 阶段一：收集所有受试者的原始 trial 数据
# ---------------------------------------------------------------------------

def collect_all_raw_trials(cfg: dict) -> dict:
    """
    遍历训练集（HC + DEP）和公开测试集，按受试者 ID 汇总所有原始 trial。

    Returns
    -------
    subject_raw : dict[user_id -> list[np.ndarray]]
        每个 user_id 对应该受试者在全部数据源中的所有原始 trial（未归一化）。
        trial 形状：(n_pts, n_channels)，float32。

    说明
    ----
    这是「整个数据集计算统计量」的关键步骤。
    只有把训练集 trial 和测试集 trial 放在一起，才能保证：
        compute_subject_stats(subject_raw[uid]) 
    得到的 mu/sigma 对该受试者在两个子集中均适用，归一化基准完全一致。
    """
    root = Path(cfg["project_root"])
    # user_id → list of (n_pts, n_ch) arrays
    subject_raw: dict[str, list[np.ndarray]] = defaultdict(list)

    # ── 训练集：HC + DEP ──────────────────────────────────────────────────
    for group_dir in ("train/HC", "train/DEP"):
        group_path = root / group_dir
        if not group_path.exists():
            log.warning("目录不存在，跳过: %s", group_path)
            continue

        for mat_path in sorted(group_path.glob("*.mat")):
            user_id  = mat_path.stem
            seg_pts  = cfg["train_seg_pts"]
            n_seg    = cfg["train_n_seg"]

            for var_name in EMOTION_MAP:
                try:
                    raw_data = load_mat_variable(mat_path, var_name)
                except Exception as e:
                    log.warning("  [collect] 跳过 %s/%s: %s", mat_path.name, var_name, e)
                    continue

                for seg_idx in range(n_seg):
                    trial = raw_data[seg_idx * seg_pts : (seg_idx + 1) * seg_pts, :]
                    if trial.shape[0] != seg_pts:
                        continue
                    subject_raw[user_id].append(trial)

            log.info("[collect] 训练集 %s — 已累积 %d 个 trial",
                     user_id, len(subject_raw[user_id]))

    # ── 公开测试集 ────────────────────────────────────────────────────────
    test_dir = root / "test"
    if test_dir.exists():
        for mat_path in sorted(test_dir.glob("*.mat")):
            user_id = mat_path.stem
            seg_pts = cfg["test_seg_pts"]
            n_seg   = cfg["test_n_seg"]

            try:
                raw_data = load_mat_variable(mat_path, "test_eeg_c")
            except KeyError:
                log.warning("[collect] 跳过 %s：未找到变量 'test_eeg_c'", mat_path)
                continue

            for seg_idx in range(n_seg):
                trial = raw_data[seg_idx * seg_pts : (seg_idx + 1) * seg_pts, :]
                if trial.shape[0] != seg_pts:
                    continue
                subject_raw[user_id].append(trial)

            log.info("[collect] 测试集  %s — 已累积 %d 个 trial",
                     user_id, len(subject_raw[user_id]))
    else:
        log.info("未找到 test/ 目录，仅使用训练集计算统计量。")

    return dict(subject_raw)


# ---------------------------------------------------------------------------
# 阶段二：计算全局统计量（每个受试者）
# ---------------------------------------------------------------------------

def build_subject_stats(subject_raw: dict, eps: float) -> dict:
    """
    对 subject_raw 中每个受试者，在其所有 trial（来自训练集+测试集）上
    计算逐通道 mu/sigma。

    Parameters
    ----------
    subject_raw : collect_all_raw_trials() 的返回值
    eps         : 防止除零

    Returns
    -------
    subject_stats : dict[user_id -> {"mu": ndarray(1,C), "sigma": ndarray(1,C)}]

    这样得到的统计量对该受试者在训练集和测试集中的所有样本均适用，
    确保 Z-score 归一化基准完全一致。
    """
    subject_stats: dict[str, dict] = {}
    for user_id, trials in subject_raw.items():
        mu, sigma = compute_subject_stats(trials, eps=eps)
        subject_stats[user_id] = {"mu": mu, "sigma": sigma}
        log.info(
            "  [stats] %s — mu∈[%.4f, %.4f]  sigma∈[%.4f, %.4f]",
            user_id,
            float(mu.min()),    float(mu.max()),
            float(sigma.min()), float(sigma.max()),
        )
    return subject_stats


# ---------------------------------------------------------------------------
# 阶段三：样本生成（使用预计算的全局统计量）
# ---------------------------------------------------------------------------

def iter_train_samples(
    mat_path: Path,
    group: int,
    cfg: dict,
    subject_stats: dict,          # ← 新增：全局统计量
):
    """
    对单个受试者 .mat 文件：
      1. 加载所有情绪的全部 trial 原始数据
      2. 用「全局统计量」（包含测试集 trial）做 Z-score 标准化
      3. 滑窗切片 → 重采样 → reshape → yield

    Parameters
    ----------
    subject_stats : build_subject_stats() 的返回值
        必须在调用本函数之前，已纳入测试集数据计算完毕。

    Yields
    ------
    (key: str, record: dict)
    """
    user_id    = mat_path.stem
    fs         = cfg["fs"]
    target_fs  = cfg["target_fs"]
    seg_pts    = cfg["train_seg_pts"]
    n_seg      = cfg["train_n_seg"]
    sample_sec = cfg["sample_sec"]
    stride_sec = cfg["sample_stride_sec"]
    group_tag  = "dep" if group == 1 else "hc"

    if user_id not in subject_stats:
        log.warning("  %s 无统计量，跳过", user_id)
        return

    mu    = subject_stats[user_id]["mu"]    # (1, 30)
    sigma = subject_stats[user_id]["sigma"] # (1, 30)

    for var_name, label in EMOTION_MAP.items():
        log.info("  Processing %s / %s", mat_path.name, var_name)
        try:
            raw_data = load_mat_variable(mat_path, var_name)
        except Exception as e:
            log.warning("  跳过 %s/%s: %s", mat_path.name, var_name, e)
            continue

        for seg_idx in range(n_seg):
            trial = raw_data[seg_idx * seg_pts : (seg_idx + 1) * seg_pts, :]
            if trial.shape[0] != seg_pts:
                log.warning(
                    "  %s/%s trial %d: 期望 %d pts，实际 %d pts，跳过",
                    mat_path.name, var_name, seg_idx, seg_pts, trial.shape[0],
                )
                continue

            # 使用全局统计量标准化（基准与测试集一致）
            trial_normed = znorm(trial, mu, sigma)

            window_id = 0
            for _, chunk in iter_sliding_windows(trial_normed, fs, sample_sec, stride_sec):
                chunk  = resample_segment(chunk, fs, target_fs)
                sample = to_cbramod_sample(chunk, target_fs, sample_sec)

                key = f"{user_id}_{group_tag}_{var_name}_{seg_idx}_{window_id}"
                yield key, {
                    "sample":    sample,
                    "label":     float(label),
                    "user_id":   user_id,
                    "group":     int(group),
                    "trial_id":  int(seg_idx),
                    "window_id": int(window_id),
                }
                window_id += 1


def iter_public_test_samples(
    mat_path: Path,
    cfg: dict,
    subject_stats: dict,          # ← 新增：全局统计量
):
    """
    对公开测试集 .mat 文件：
      1. 加载所有 trial 原始数据
      2. 用「全局统计量」（包含训练集 trial）做 Z-score 标准化
      3. 滑窗切片 → 重采样 → reshape → yield

    Parameters
    ----------
    subject_stats : build_subject_stats() 的返回值
        与训练集使用完全相同的统计量，保证归一化基准一致。

    Yields
    ------
    (sample: ndarray, meta: dict)
    """
    user_id    = mat_path.stem
    fs         = cfg["fs"]
    target_fs  = cfg["target_fs"]
    seg_pts    = cfg["test_seg_pts"]
    n_seg      = cfg["test_n_seg"]
    sample_sec = cfg["sample_sec"]
    stride_sec = cfg["sample_stride_sec"]

    if user_id not in subject_stats:
        log.warning("  %s 无统计量，跳过", user_id)
        return

    mu    = subject_stats[user_id]["mu"]
    sigma = subject_stats[user_id]["sigma"]

    try:
        raw_data = load_mat_variable(mat_path, "test_eeg_c")
    except KeyError:
        log.warning("跳过 %s：未找到变量 'test_eeg_c'", mat_path)
        return

    for seg_idx in range(n_seg):
        trial = raw_data[seg_idx * seg_pts : (seg_idx + 1) * seg_pts, :]
        if trial.shape[0] != seg_pts:
            log.warning(
                "  %s trial %d: 期望 %d pts，实际 %d pts，跳过",
                mat_path.name, seg_idx, seg_pts, trial.shape[0],
            )
            continue

        # 使用全局统计量标准化（基准与训练集一致）
        trial_normed = znorm(trial, mu, sigma)

        window_id = 0
        for _, chunk in iter_sliding_windows(trial_normed, fs, sample_sec, stride_sec):
            chunk  = resample_segment(chunk, fs, target_fs)
            sample = to_cbramod_sample(chunk, target_fs, sample_sec)
            yield sample, {
                "user_id":    user_id,
                "trial_id":   seg_idx,
                "window_id":  window_id,
                "sample_key": f"{user_id}_trial_{seg_idx}_{window_id}",
            }
            window_id += 1


# ---------------------------------------------------------------------------
# 受试者级别 train/val/test 划分
# ---------------------------------------------------------------------------

def stable_subject_split(user_ids, val_ratio: float, test_ratio: float, seed: int):
    """
    按哈希排序后切分，保证相同 seed 下结果一致。
    返回三个集合: (train_subjects, val_subjects, test_subjects)
    """
    ordered = sorted(
        user_ids,
        key=lambda sid: hashlib.md5(f"{seed}:{sid}".encode()).hexdigest(),
    )
    n      = len(ordered)
    n_test = max(0, round(n * test_ratio))
    n_val  = max(0, round(n * val_ratio))

    test_subj  = set(ordered[:n_test])
    val_subj   = set(ordered[n_test : n_test + n_val])
    train_subj = set(ordered[n_test + n_val :])
    return train_subj, val_subj, test_subj


# ---------------------------------------------------------------------------
# LMDB 写入
# ---------------------------------------------------------------------------

def write_lmdb(db: lmdb.Environment, key: str, value) -> None:
    with db.begin(write=True) as txn:
        txn.put(key.encode(), pickle.dumps(value))


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def build_train_lmdb(cfg: dict, subject_stats: dict) -> None:
    """
    遍历 train/HC 和 train/DEP，写入 LMDB，并附加 __keys__ 索引。

    Parameters
    ----------
    subject_stats : build_subject_stats() 的返回值（已纳入测试集数据）
    """
    root       = Path(cfg["project_root"])
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    db = lmdb.open(str(output_dir), map_size=cfg["map_size_gb"] * (1024 ** 3))

    subject_entries: dict[str, list[str]] = {}

    for group_dir, group_label in [("train/HC", 0), ("train/DEP", 1)]:
        group_path = root / group_dir
        if not group_path.exists():
            log.warning("目录不存在，跳过: %s", group_path)
            continue

        for mat_path in sorted(group_path.glob("*.mat")):
            log.info("Processing %s", mat_path)
            for key, record in iter_train_samples(
                mat_path, group_label, cfg, subject_stats   # 传入全局统计量
            ):
                write_lmdb(db, key, record)
                subject_entries.setdefault(record["user_id"], []).append(key)

    # 受试者级别划分
    train_subj, val_subj, test_subj = stable_subject_split(
        subject_entries.keys(), cfg["val_ratio"], cfg["test_ratio"], cfg["seed"]
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
        "LMDB 构建完成 — train: %d  val: %d  test: %d",
        len(split_keys["train"]),
        len(split_keys["val"]),
        len(split_keys["test"]),
    )


def export_public_test(cfg: dict, subject_stats: dict) -> None:
    """
    若存在 test/ 目录，将公开测试集导出为 .npy + .pkl。

    Parameters
    ----------
    subject_stats : build_subject_stats() 的返回值（已纳入训练集数据）
    """
    test_dir = Path(cfg["project_root"]) / "test"
    if not test_dir.exists():
        return

    out_root      = Path(cfg["output_dir"]).parent
    samples_path  = out_root / "public_test_samples.npy"
    manifest_path = out_root / "public_test_manifest.pkl"

    sample_list, meta_list = [], []
    for mat_path in sorted(test_dir.glob("*.mat")):
        log.info("Public test: %s", mat_path)
        for sample, meta in iter_public_test_samples(
            mat_path, cfg, subject_stats            # 传入全局统计量
        ):
            sample_list.append(sample)
            meta_list.append(meta)

    if sample_list:
        np.save(samples_path, np.stack(sample_list).astype(np.float32))
        with open(manifest_path, "wb") as f:
            pickle.dump(meta_list, f)
        log.info(
            "公开测试集已导出 → %s (%d 样本), %s",
            samples_path, len(sample_list), manifest_path,
        )
    else:
        log.warning("test/ 目录下未找到有效样本。")


def main():
    log.info("开始构建 CBraMod Emotion LMDB，数据根目录: %s", CFG["project_root"])

    # ── 阶段一：收集全部原始 trial（训练集 + 测试集）─────────────────────
    log.info("=== 阶段一：收集全部原始 trial ===")
    subject_raw = collect_all_raw_trials(CFG)

    # ── 阶段二：在整个数据集上计算每个受试者的统计量 ─────────────────────
    log.info("=== 阶段二：计算全局 Z-score 统计量 ===")
    subject_stats = build_subject_stats(subject_raw, eps=CFG["znorm_eps"])

    # 释放原始数据，节省内存
    del subject_raw

    # ── 阶段三：用全局统计量标准化并写入 ─────────────────────────────────
    log.info("=== 阶段三：标准化并写入 LMDB / npy ===")
    build_train_lmdb(CFG, subject_stats)
    export_public_test(CFG, subject_stats)

    log.info("全部完成。")


if __name__ == "__main__":
    main()
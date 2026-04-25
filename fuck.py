%cd /vq/emotion
!git pull
%uv pip install mne pyarrow einops
import os
import gc
import ast
import random
import warnings
import sympy
import sympy.printing
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = torch.cuda.is_available()
print(f"device = {device}")

ELECTRODE_NAMES = [
    "FP1", "FP2", "F7",  "F3",  "FZ",  "F4",  "F8",
    "FT7", "FC3", "FCZ", "FC4", "FT8",
    "T3",  "C3",  "CZ",  "C4",  "T4",
    "TP7", "CP3", "CPZ", "CP4", "TP8",
    "T5",  "P3",  "PZ",  "P4",  "T6",
    "O1",  "OZ",  "O2",
]

ELECTRODE_POS_3D = {
    "FP1": (-0.309, 0.951, -0.035), "FP2": ( 0.309, 0.951, -0.035),
    "F7":  (-0.809, 0.587, -0.035), "F3":  (-0.545, 0.673,  0.500),
    "FZ":  ( 0.000, 0.719,  0.695), "F4":  ( 0.545, 0.673,  0.500),
    "F8":  ( 0.809, 0.587, -0.035), "FT7": (-0.951, 0.309, -0.035),
    "FC3": (-0.650, 0.350,  0.680), "FCZ": ( 0.000, 0.391,  0.920),
    "FC4": ( 0.650, 0.350,  0.680), "FT8": ( 0.951, 0.309, -0.035),
    "T3":  (-1.000, 0.000, -0.035), "C3":  (-0.719, 0.000,  0.695),
    "CZ":  ( 0.000, 0.000,  1.000), "C4":  ( 0.719, 0.000,  0.695),
    "T4":  ( 1.000, 0.000, -0.035), "TP7": (-0.951,-0.309, -0.035),
    "CP3": (-0.650,-0.350,  0.680), "CPZ": ( 0.000,-0.391,  0.920),
    "CP4": ( 0.650,-0.350,  0.680), "TP8": ( 0.951,-0.309, -0.035),
    "T5":  (-0.809,-0.587, -0.035), "P3":  (-0.545,-0.673,  0.500),
    "PZ":  ( 0.000,-0.719,  0.695), "P4":  ( 0.545,-0.673,  0.500),
    "T6":  ( 0.809,-0.587, -0.035), "O1":  (-0.309,-0.951, -0.035),
    "OZ":  ( 0.000,-1.000, -0.035), "O2":  ( 0.309,-0.951, -0.035),
}

# =========================
# 配置
# =========================
train_parquet = "./output/train.parquet"
test_parquet  = "./output/test.parquet"
save_dir = "./output"
os.makedirs(save_dir, exist_ok=True)

SEED         = 42
EPOCHS       = 25
BATCH_SIZE   = 128 if torch.cuda.is_available() else 32
LR           = 1e-3
PATIENCE     = 15

# <<<< 修改：新增平滑标签与投票集成参数
LABEL_SMOOTH        = 0.1   # BCE 标签平滑，0.0 表示不平滑，建议尝试 0.05~0.1
USE_VOTING_ENSEMBLE = False # False=概率平均+全局阈值；True=各折硬标签投票
# >>>>

# -----------------------------------------------
# GRL 超参数
# ----------------------------------------------
# 第一重：Train vs Test 域对齐（个体无关）
GRL_DOMAIN_LAMBDA  = 0.1   # 域分类器梯度反转权重

# 第二重：HC vs DEP 组别对齐（病理无关）
GRL_GROUP_LAMBDA   = 0.1   # 组别分类器梯度反转权重

# GRL λ 预热调度：从 0 线性增长到目标值，避免训练初期反转梯度破坏特征提取
GRL_WARMUP_EPOCHS  = 5

# -----------------------------------------------
ADJ_TYPE       = "distance"
DIST_SIGMA     = None
DIST_THRESH    = 0.1
PEARSON_THRESH = 0.3
PEARSON_N_SAMPLES = 5000

FAST_DEBUG         = False
FAST_DEBUG_N_USERS = 10

if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

# =========================
# 1. 随机种子与3D坐标
# =========================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(SEED)

# =========================
# 2. 邻接矩阵
# =========================
def build_distance_adjacency(names, sigma=None, threshold=0.0):
    coords = np.array([ELECTRODE_POS_3D[n.upper()] for n in names], dtype=np.float32)
    diff   = coords[:, None, :] - coords[None, :, :]
    dist   = np.sqrt((diff ** 2).sum(-1))
    if sigma is None:
        sigma = dist[dist > 0].mean()
    A = np.exp(-(dist ** 2) / (2 * sigma ** 2))
    np.fill_diagonal(A, 0.0)
    if threshold > 0:
        A[A < threshold] = 0.0
    return A

def build_pearson_adjacency(X, abs_value=True, threshold=0.0, n_samples=5000):
    N, C, T = X.shape
    if N > n_samples:
        idx = np.random.RandomState(0).choice(N, n_samples, replace=False)
        Xs  = X[idx].astype(np.float32)
    else:
        Xs  = X.astype(np.float32)
    Xs  = Xs - Xs.mean(axis=2, keepdims=True)
    Xs  = Xs / (Xs.std(axis=2, keepdims=True) + 1e-8)
    corr = np.einsum("nct,ndt->ncd", Xs, Xs) / T
    A    = corr.mean(axis=0)
    if abs_value:
        A = np.abs(A)
    np.fill_diagonal(A, 0.0)
    if threshold > 0:
        A[A < threshold] = 0.0
    return A

# =========================
# 3. 数据处理
# =========================
def parse_signal(x):
    if isinstance(x, np.ndarray): return x.astype(np.float32)
    if isinstance(x, list):       return np.asarray(x, dtype=np.float32)
    if isinstance(x, str):        return np.asarray(ast.literal_eval(x), dtype=np.float32)
    return np.asarray(x, dtype=np.float32)

def build_array_from_df(df, ch_cols, out_dtype=np.float16, desc="build array"):
    n       = len(df)
    seq_len = len(parse_signal(df.iloc[0][ch_cols[0]]))
    n_ch    = len(ch_cols)
    print(f"{desc}: N={n}, C={n_ch}, T={seq_len}")
    X = np.empty((n, n_ch, seq_len), dtype=out_dtype)
    for ci, col in enumerate(tqdm(ch_cols, desc=desc)):
        arr = np.stack([parse_signal(v) for v in df[col].tolist()]).astype(out_dtype)
        X[:, ci, :] = arr
        del arr; gc.collect()
    return X



def get_hard_labels_by_p(df, p, user_col="user_id", prob_col="prob"):
    """
    按用户分组，将每个用户试次内预测概率最高的前 p% 设为 1，其余设为 0。
    p 的取值范围是 [0, 1]
    """
    res = df.copy()
    res["hard"] = 0
    
    for uid, group in res.groupby(user_col):
        n = len(group)
        n_pos = int(np.round(n * p))
        # 排序获取 top n_pos 的 index
        if n_pos > 0:
            top_indices = group.nlargest(n_pos, prob_col).index
            res.loc[top_indices, "hard"] = 1
            
    return res["hard"].values

def find_best_p(df, user_col="user_id", prob_col="prob", label_col="label"):
    """
    在验证集/OOF中搜索最佳的比例 p
    """
    best_p, best_acc = 0.5, -1
    # 搜索范围例如 10% 到 90% (步长可以调整)
    for p in np.linspace(0.1, 0.9, 81):
        preds = get_hard_labels_by_p(df, p, user_col, prob_col)
        acc = accuracy_score(df[label_col], preds)
        if acc > best_acc:
            best_acc, best_p = acc, p
    return best_p, best_acc


# =========================
# 预测工具
# =========================
def predict_probs(model, loader, device):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            xb = batch[0]
            xb = xb.to(device, non_blocking=True).float()
            emotion_logit, _, _, _ = model(xb)
            all_probs.append(torch.sigmoid(emotion_logit).cpu().numpy())
    return np.concatenate(all_probs)


# =========================
# 单折训练（双重 GRL）
# =========================
def train_one_fold(model, train_loader, test_loader_for_adv,
                   optimizer, scheduler, scaler, criterion,
                   epochs, patience, device, use_amp,
                   val_loader, val_idx, meta_train, 
                   save_path, fold_id, label_smooth=0.0):

    test_iter = iter(test_loader_for_adv)
    adv_criterion = nn.BCEWithLogitsLoss()

    # <<<< 修改：由于现在是计算 val_loss 并以此早停，所以 best_score 初始应无穷大，且寻找更小的值
    best_score, best_epoch, no_improve = float('inf'), 0, 0
    # >>>>

    for epoch in range(1, epochs + 1):
        cur_domain_lam = get_grl_lambda(epoch, epochs, GRL_DOMAIN_LAMBDA, GRL_WARMUP_EPOCHS)
        cur_group_lam = get_grl_lambda(epoch, epochs, GRL_GROUP_LAMBDA, GRL_WARMUP_EPOCHS)
        model.set_grl_lambdas(cur_domain_lam, cur_group_lam)
        
        model.train()
        running = {"emotion": 0.0, "domain": 0.0, "group": 0.0, "total": 0.0}
        
        for batch in train_loader:
            xb_train, yb_emotion, yb_group = batch
            xb_train = xb_train.to(device, non_blocking=True).float()
            yb_emotion = yb_emotion.to(device, non_blocking=True).float()
            yb_group = yb_group.to(device, non_blocking=True).long()

            try:
                test_batch = next(test_iter)
            except StopIteration:
                test_iter = iter(test_loader_for_adv)
                test_batch = next(test_iter)
            xb_test = test_batch[0].to(device, non_blocking=True).float()
            
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                emo_logit, dom_logit_tr, grp_logit_tr, _ = model(xb_train)
                _, dom_logit_te, _, _ = model(xb_test)

                if label_smooth > 0:
                    yb_emotion = yb_emotion * (1.0 - label_smooth) + 0.5 * label_smooth
                loss_emotion = criterion(emo_logit, yb_emotion)

                bs_tr = xb_train.size(0)
                bs_te = xb_test.size(0)
                dom_logits_all = torch.cat([dom_logit_tr, dom_logit_te], dim=0)
                dom_labels_all = torch.cat([
                    torch.zeros(bs_tr, dtype=torch.float32, device=device),  
                    torch.ones(bs_te, dtype=torch.float32, device=device),   
                ], dim=0)
                loss_domain = adv_criterion(dom_logits_all, dom_labels_all)

                valid_mask = (yb_group >= 0)
                if valid_mask.sum() > 1:
                    loss_group = adv_criterion(
                        grp_logit_tr[valid_mask],
                        yb_group[valid_mask].float()
                    )
                else:
                    loss_group = torch.tensor(0.0, device=device)

                loss = loss_emotion + loss_domain + loss_group

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            n = xb_train.size(0)
            running["emotion"] += loss_emotion.item() * n
            running["domain"] += loss_domain.item() * n
            running["group"] += loss_group.item() * n
            running["total"] += loss.item() * n
            
        scheduler.step()
        n_train = len(train_loader.dataset)
        print(f" Fold {fold_id} | Epoch {epoch:02d} | "
              f"emo={running['emotion']/n_train:.4f} | "
              f"dom={running['domain']/n_train:.4f} | "
              f"grp={running['group']/n_train:.4f} | "
              f"total={running['total']/n_train:.4f}")

        # ==================================================
        # <<<< 修改：验证集只算 val_loss (主任务)，跳过 AUC、ACC 结算
        # ==================================================
        model.eval()
        val_loss_sum = 0.0
        n_val = 0
        with torch.no_grad():
            for val_batch in val_loader:
                vx, vy_emo, _ = val_batch
                vx = vx.to(device, non_blocking=True).float()
                vy_emo = vy_emo.to(device, non_blocking=True).float()
                
                with torch.cuda.amp.autocast(enabled=use_amp):
                    v_emo_logit, _, _, _ = model(vx)
                    v_loss = criterion(v_emo_logit, vy_emo)
                    
                n_b = vx.size(0)
                val_loss_sum += v_loss.item() * n_b
                n_val += n_b
                
        val_loss = val_loss_sum / max(n_val, 1)
        print(f" -> val_loss={val_loss:.4f}")

        # 使用 val_loss 作为早停和保存最佳模型的标准 (越小越好)
        if val_loss < best_score:
            best_score, best_epoch, no_improve = val_loss, epoch, 0
            torch.save(model.state_dict(), save_path)
        else:
            no_improve += 1
            
        if no_improve >= patience:
            print(f" Early stopping at epoch {epoch}")
            break
        # >>>> 

    print(f" >> Fold {fold_id} best epoch={best_epoch}, best val_loss={best_score:.6f}")
    model.load_state_dict(torch.load(save_path, map_location=device))
    return model


# =========================
# 梯度反转层（GRL）
# =========================
class GradientReversalFunction(torch.autograd.Function):
    """
    前向：恒等变换
    反向：梯度乘以 -λ（反转方向）

    λ 采用标准 DANN 调度或外部传入固定值。
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(torch.tensor(lambda_))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        (lambda_,) = ctx.saved_tensors
        return -lambda_.item() * grad_output, None


class GradientReversal(nn.Module):
    """包装 GRL，λ 可在训练中动态调整"""
    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def set_lambda(self, lam: float):
        self.lambda_ = lam

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_)


def get_grl_lambda(epoch: int, total_epochs: int,
                   target_lambda: float,
                   warmup_epochs: int = 5) -> float:
    """
    GRL λ 调度策略：
    - warmup 阶段线性从 0 → target_lambda（避免训练初期破坏特征提取）
    - warmup 结束后保持 target_lambda 不变
    
    也可改用 DANN 原文的 sigmoid 调度：
        p = epoch / total_epochs
        return target_lambda * (2/(1+exp(-10*p)) - 1)
    """
    if epoch <= warmup_epochs:
        return target_lambda * (epoch / warmup_epochs)
    return target_lambda

# =========================
#  GNN层（参考GATv2实现）
# =========================
class GATv2Layer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int,
                 heads: int = 4, dropout: float = 0.1,
                 share_weights: bool = False):
        super().__init__()
        assert out_dim % heads == 0
        self.heads    = heads
        self.head_dim = out_dim // heads
        self.dropout  = dropout

        self.W_src = nn.Linear(in_dim, out_dim, bias=False)
        self.W_dst = self.W_src if share_weights else \
                     nn.Linear(in_dim, out_dim, bias=False)

        self.att_vec = nn.Parameter(torch.empty(1, heads, self.head_dim))
        self.bias    = nn.Parameter(torch.zeros(out_dim))

        nn.init.xavier_uniform_(self.W_src.weight)
        if not share_weights:
            nn.init.xavier_uniform_(self.W_dst.weight)
        nn.init.xavier_uniform_(self.att_vec.view(1, -1).unsqueeze(0))

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        H, D    = self.heads, self.head_dim

        src = self.W_src(x).view(B, N, H, D)
        dst = self.W_dst(x).view(B, N, H, D)

        e = F.leaky_relu(
            src.unsqueeze(2) + dst.unsqueeze(1),
            negative_slope=0.2
        )
        score = (e * self.att_vec.view(1, 1, 1, H, D)).sum(dim=-1)

        mask  = (A == 0).unsqueeze(0).unsqueeze(-1)
        score = score.masked_fill(mask, float('-inf'))

        alpha = F.softmax(score, dim=2)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = torch.einsum('bijh,bjhd->bihd', alpha, dst.view(B, N, H, D))
        out = out.reshape(B, N, H * D) + self.bias
        return out


def normalize_adj(A: np.ndarray) -> torch.Tensor:
    A        = A + np.eye(A.shape[0], dtype=np.float32)
    D        = A.sum(axis=1)
    D_inv_sq = 1.0 / np.sqrt(D + 1e-8)
    return torch.from_numpy(
        (A * D_inv_sq[:, None] * D_inv_sq[None, :]).astype(np.float32)
    )


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.score(h), dim=1)
        return (w * h).sum(dim=1)


# =========================
# 完整 GNN 模型
# =========================
class EEGFreqGNN(nn.Module):
    """
    特征提取器(GATv2)
        │
        ├─► emotion_head          主任务（正向梯度）
        │
        ├─► GRL(λ₁) → domain_head   第一重：Train/Test 域对齐
        │
        └─► GRL(λ₂) → group_head    第二重：HC/DEP 组别对齐
    """
    def __init__(self, adj,
                 node_hidden:    int   = 64,
                 gcn_hidden:     int   = 64,
                 heads:          int   = 4,
                 drop_rate:      float = 0.1,
                 drop_edge:      float = 0.1,
                 domain_lambda:  float = 0.1,   # 第一重 GRL 初始λ
                 group_lambda:   float = 0.1,   # 第二重 GRL 初始λ
                 ):
        super().__init__()

        # ---------- 邻接矩阵 ----------
        self.register_buffer("A", normalize_adj(adj))
        self.drop_edge = drop_edge
        feat_dim = gcn_hidden

        # ---------- 节点特征编码器 ----------
        self.node_encoder = NodeEncoder()
        # ---------- GATv2 ----------
        self.gat  = GATv2Layer(
            in_dim=gcn_hidden, out_dim=gcn_hidden,
            heads=heads, dropout=drop_rate, share_weights=True,
        )
        self.ln   = nn.LayerNorm(gcn_hidden)
        self.drop = nn.Dropout(drop_rate)
        self.skip = nn.Linear(node_hidden, gcn_hidden, bias=False) \
                    if node_hidden != gcn_hidden else nn.Identity()

        # ---------- 注意力池化 ----------
        self.pool = AttentionPooling(gcn_hidden)

        # ---------- 情绪分类头（主任务）----------
        self.emotion_head = nn.Linear(feat_dim, 1)

        # ---------- 第一重 GRL：Train vs Test 域对齐 ----------
        self.grl_domain   = GradientReversal(lambda_=domain_lambda)
        self.domain_head  = nn.Linear(feat_dim, 1)

        self.grl_group    = GradientReversal(lambda_=group_lambda)
        self.group_head   = nn.Linear(feat_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in [self.emotion_head]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def set_grl_lambdas(self, domain_lambda: float, group_lambda: float):
        """训练循环中动态调整 GRL 强度"""
        self.grl_domain.set_lambda(domain_lambda)
        self.grl_group.set_lambda(group_lambda)

    def _drop_edge(self, A: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_edge == 0.0:
            return A
        mask = torch.bernoulli(torch.full_like(A, 1.0 - self.drop_edge))
        mask = mask * mask.T
        eye  = torch.eye(A.size(0), device=A.device)
        return A * mask + eye * (1 - mask) * A

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        A  = self._drop_edge(self.A)
        h  = self.node_encoder(x)
        h2 = self.gat(h, A)
        h2 = self.drop(F.gelu(self.ln(h2)))
        h2 = h2 + self.skip(h)
        return self.pool(h2)                  # (B, gcn_hidden)

    def forward(self, x: torch.Tensor):
        """
        返回:
            emotion_logit : (B,)       主任务 logit
            domain_logit  : (B,)       域分类 logit（经 GRL, 1维）
            group_logit   : (B,)       组别分类 logit（经 GRL, 1维）
            feat          : (B, D)     原始特征
        """
        feat          = self.extract_features(x)
        emotion_logit = self.emotion_head(feat).squeeze(1)
        domain_logit  = self.domain_head(self.grl_domain(feat)).squeeze(1)
        group_logit   = self.group_head(self.grl_group(feat)).squeeze(1)
        return emotion_logit, domain_logit, group_logit, feat

def build_freq_array(X, desc="build freq"):
    N, C, T = X.shape
    F       = T // 2 + 1
    print(f"{desc}: N={N}, C={C}, T={T} -> F={F}")
    
    X_f  = X.astype(np.float32)
    
    # 加汉宁窗 (Hanning Window) 以减少截断造成的频谱泄露
    window = np.hanning(T).astype(np.float32)
    X_f = X_f * window 
    
    # 执行快速傅里叶变换，并计算对数幅度谱
    X_fq = np.log1p(np.abs(np.fft.rfft(X_f, axis=-1)))
    
    del X_f; gc.collect()
    return X_fq.astype(np.float32)

class EEGFreqDataset(Dataset):
    """
    增加了 mean 和 std 参数，用于在获取数据时动态进行 Z-score 归一化
    """
    def __init__(self, X_freq, y=None, group=None, indices=None, mean=None, std=None):
        self.X       = X_freq
        self.y       = y
        self.group   = group                        
        self.indices = np.arange(len(X_freq)) if indices is None else np.asarray(indices)
        
        # 保存传入的统计量 (shape: [1, C, F])
        self.mean    = mean
        self.std     = std

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        x_np = self.X[idx]  # [C, F]
        
        # ---- 实时应用跨 Trial 归一化 ----
        if self.mean is not None and self.std is not None:
            # self.mean[0] 的 shape 是 [C, F]
            x_np = (x_np - self.mean[0]) / self.std[0]
            
        x = torch.from_numpy(x_np)
        
        if self.y is None:
            return (x,)                             
            
        g = self.group[idx] if self.group is not None else -1
        return (
            x,
            torch.tensor(self.y[idx],    dtype=torch.float32),  
            torch.tensor(g,              dtype=torch.long),      
        )


# =========================
# 11. 主流程
# =========================
if __name__ == "__main__":

    # -------- 读取训练集 --------
    print("\n========== 读取训练集 ==========")
    train_df = pd.read_parquet(train_parquet, engine="pyarrow")
    train_df["user_id"]   = train_df["user_id"].astype(str)
    train_df["trial_key"] = (train_df["user_id"] + "__" +
                             train_df["emotion"]  + "__" +
                             train_df["trial_id"].astype(str))

    ch_cols = sorted([c for c in train_df.columns if c.startswith("ch")])
    assert len(ch_cols) == len(ELECTRODE_NAMES)

    if FAST_DEBUG:
        keep     = sorted(train_df["user_id"].unique())[:FAST_DEBUG_N_USERS]
        train_df = train_df[train_df["user_id"].isin(keep)].reset_index(drop=True)

    # ---- 提取 group 标签（0=HC, 1=DEP）----
    # 假设列名为 "group"；若不存在则填 -1（后续训练会跳过该约束）
    if "group" in train_df.columns:
        group_all = train_df["group"].values.astype(np.int64)
        print(f"group 标签分布: HC={( group_all==0).sum()}, DEP={(group_all==1).sum()}")
    else:
        print("警告: 训练集中未找到 'group' 列，第二重约束将被跳过")
        group_all = np.full(len(train_df), -1, dtype=np.int64)

    meta_train = train_df[["user_id","trial_id","emotion","label","trial_key"]].copy()
    X_all      = build_array_from_df(train_df, ch_cols, np.float16, "build train array")
    y_all      = train_df["label"].values.astype(np.float32)
    del train_df; gc.collect()

    # -------- 邻接矩阵 --------
    print(f"\n========== 构建邻接矩阵: {ADJ_TYPE} ==========")
    if ADJ_TYPE == "distance":
        adj = build_distance_adjacency(ELECTRODE_NAMES,
                                       sigma=DIST_SIGMA,
                                       threshold=DIST_THRESH)
    else:
        rng = np.random.RandomState(SEED)
        si  = rng.choice(len(X_all), min(PEARSON_N_SAMPLES, len(X_all)), replace=False)
        adj = build_pearson_adjacency(
            X_all[si].astype(np.float32),
            abs_value=True, threshold=PEARSON_THRESH, n_samples=len(si)
        )

    # -------- 频域转换 --------
    print("\n========== 时域→频域 ==========")
    X_freq_all = build_freq_array(X_all, "train freq")
    freq_bins  = X_freq_all.shape[2]
    del X_all; gc.collect()

    # -------- 读取测试集 --------
    print("\n========== 读取测试集 ==========")
    test_df   = pd.read_parquet(test_parquet, engine="pyarrow")
    meta_test = test_df[["user_id","trial_id"]].copy().reset_index(drop=True)
    X_test    = build_array_from_df(test_df, ch_cols, np.float16, "build test array")
    del test_df; gc.collect()

    X_freq_test = build_freq_array(X_test, "test freq")
    del X_test; gc.collect()

    pin_memory = torch.cuda.is_available()

    # -------- LOSO 交叉验证 --------
    print("\n========== LOSO 交叉验证 ==========")
    all_users = meta_train["user_id"].unique()
    n_users = len(all_users)
    print(f"总被试数: {n_users}")
    
    oof_records = []
    fold_p_values = [] # 用于保存每一折搜索到的最佳 p

    for fold_i, leave_user in enumerate(all_users):
        print(f"\n{'='*60}")
        print(f"Fold {fold_i+1}/{n_users}  | 留出被试: {leave_user}")
        print(f"{'='*60}")

        val_mask   = meta_train["user_id"] == leave_user
        train_mask = ~val_mask
        train_idx  = np.where(train_mask)[0]
        val_idx    = np.where(val_mask)[0]

        # ---- 构建 Dataset（含 group 标签）----
        X_train_fold = X_freq_all[train_idx]
        
        # 沿 Trial 维度 (axis=0) 计算均值和标准差
        # 保持维度为 [1, C, F]，方便利用广播机制运算
        fold_mean = np.mean(X_train_fold, axis=0, keepdims=True)
        fold_std  = np.std(X_train_fold,  axis=0, keepdims=True) + 1e-8
        
        del X_train_fold # 释放内存

        # 测试集 DataLoader（无标签，用于域对齐）
        test_ds_adv  = EEGFreqDataset(X_freq_test, mean=fold_mean, std=fold_std)  
        test_loader_adv = DataLoader(test_ds_adv,
                                     batch_size=BATCH_SIZE,
                                     shuffle=True,
                                     pin_memory=pin_memory,
                                     drop_last=True)


        # ---- 构建 Dataset（传入 mean 和 std）----
        train_ds = EEGFreqDataset(X_freq_all, y_all, group_all, train_idx, mean=fold_mean, std=fold_std)
        val_ds   = EEGFreqDataset(X_freq_all, y_all, group_all, val_idx,   mean=fold_mean, std=fold_std)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                                  shuffle=True,  pin_memory=pin_memory,
                                  drop_last=True)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                                  shuffle=False, pin_memory=pin_memory)

        # ---- 模型初始化 ----
        seed_everything(SEED + fold_i)
        model = EEGFreqGNN(
            adj=adj,
            domain_lambda=GRL_DOMAIN_LAMBDA,
            group_lambda=GRL_GROUP_LAMBDA,
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Trainable params: {n_params/1e6:.3f} M")

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        scaler    = torch.cuda.amp.GradScaler(enabled=use_amp)
        criterion = nn.BCEWithLogitsLoss()

        save_path = os.path.join(save_dir, f"best_fold{fold_i+1}.pt")

        # <<<< 修改：传入 label_smooth 参数
        model = train_one_fold(
            model               = model,
            train_loader        = train_loader,
            test_loader_for_adv = test_loader_adv,
            optimizer           = optimizer,
            scheduler           = scheduler,
            scaler              = scaler,
            criterion           = criterion,
            epochs              = EPOCHS,
            patience            = PATIENCE,
            device              = device,
            use_amp             = use_amp,
            val_loader          = val_loader,
            val_idx             = val_idx,
            meta_train          = meta_train,
            save_path           = save_path,
            fold_id             = fold_i + 1,
            label_smooth        = LABEL_SMOOTH,
        )
        # >>>>

        # ---- OOF 预测 ----
        val_probs = predict_probs(model, val_loader, device)
        val_meta  = meta_train.iloc[val_idx].copy().reset_index(drop=True)
        val_meta["prob"] = val_probs
        val_trial = val_meta.groupby("trial_key", as_index=False).agg(
            user_id =("user_id",  "first"),
            trial_id=("trial_id", "first"),
            emotion =("emotion",  "first"),
            label   =("label",    "first"),
            prob    =("prob",     "mean"),
        )

        # <<<< 修改：使用被试内组排序取前 p% 的方式搜出最优 p
        best_p, fold_acc = find_best_p(val_trial, user_col="user_id", prob_col="prob", label_col="label")
        fold_p_values.append(best_p)
        
        print(f" Fold {fold_i+1} OOF | acc={fold_acc:.4f} best_p={best_p:.3f}")
        val_trial["fold"] = fold_i + 1
        oof_records.append(val_trial)

        del model, train_ds, val_ds, train_loader, val_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -------- 全局 OOF 汇总 --------
    print("\n========== 全局 OOF 指标 ==========")
    oof_df  = pd.concat(oof_records, ignore_index=True)
    oof_df.to_csv(os.path.join(save_dir, "oof_predictions.csv"), index=False)

    global_best_p, oof_acc = find_best_p(oof_df, user_col="user_id", prob_col="prob", label_col="label")
    oof_df["hard"] = get_hard_labels_by_p(oof_df, global_best_p, user_col="user_id", prob_col="prob")
    oof_f1 = f1_score(oof_df["label"], oof_df["hard"])

    print(f"OOF ACC = {oof_acc:.4f}  (thr={best_p:.3f})")
    print(f"OOF F1  = {oof_f1:.4f}")

    # -------- 测试集推断 --------
    print("\n========== 测试集推断（LOSO Ensemble） ==========")
    
    global_mean = np.mean(X_freq_all, axis=0, keepdims=True)
    global_std  = np.std(X_freq_all,  axis=0, keepdims=True) + 1e-8
    
    # 传入 Dataset
    test_ds     = EEGFreqDataset(X_freq_test, mean=global_mean, std=global_std)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                             shuffle=False, pin_memory=pin_memory)
    
    submission_template = (
        meta_test.groupby(["user_id","trial_id"], as_index=False)
                 .size()
                 .drop(columns=["size"])
                 .sort_values(["user_id","trial_id"])
                 .reset_index(drop=True)
    )

    vote_acc = None
    n_valid_folds = 0
    for fold_i, leave_user in enumerate(all_users):
        save_path = os.path.join(save_dir, f"best_fold{fold_i+1}.pt")
        if not os.path.exists(save_path):
            print(f"  Fold {fold_i+1} 权重文件不存在，跳过")
            continue
        m = EEGFreqGNN(
            adj=adj, freq_bins=freq_bins, n_nodes=len(ch_cols),
            domain_lambda=GRL_DOMAIN_LAMBDA,
            group_lambda=GRL_GROUP_LAMBDA,
        ).to(device)
        m.load_state_dict(torch.load(save_path, map_location=device))
        probs = predict_probs(m, test_loader, device)
        meta_test["prob"] = probs
        trial_df = (
            meta_test.groupby(["user_id","trial_id"], as_index=False)
                     .agg(prob=("prob","mean"))
        )
        fold_best_p = fold_p_values[fold_i]
        trial_df["hard"] = get_hard_labels_by_p(trial_df, fold_best_p, user_col="user_id", prob_col="prob")
        
        merged = submission_template.merge(
            trial_df[["user_id","trial_id","hard"]],
            on=["user_id","trial_id"], how="left"
        )
        
        if vote_acc is None:
            vote_acc = merged["hard"].fillna(0).values
        else:
            vote_acc += merged["hard"].fillna(0).values
            
        n_valid_folds += 1
        del m; gc.collect()
        print(f" Fold {fold_i+1}/{n_users} 硬标签推断完成 (p={fold_best_p:.3f})")

    submission_template["vote_sum"] = vote_acc
    # 如果超半数模型认为是1，则设为1
    submission_template["Emotion_label"] = (
        (vote_acc / max(n_valid_folds, 1)) >= 0.5
    ).astype(int)
    
    submission_template[["user_id","trial_id","Emotion_label"]].to_csv(
        os.path.join(save_dir, "submission.csv"), index=False
    )
    print(f"✅ 投票集成完成，已保存至 {save_dir}/submission.csv")
    print(f" 参与投票折数: {n_valid_folds}")
    print(f" 各折所用前置 p 比例: {[f'{p:.3f}' for p in fold_p_values]}")

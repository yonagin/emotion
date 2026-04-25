import argparse
import copy
import os
import random
from timeit import default_timer as timer

import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from datasets import faced_dataset
from finetune_evaluator import Evaluator
from models.DSAINet import DSAINet


def str2bool(v):
    # argparse's `type=bool` is almost always wrong because bool("False") is True.
    if isinstance(v, bool):
        return v
    if v is None:
        return True
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v!r}")


def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class DSAINetAdapter(nn.Module):
    """
    FACED sample shape from LMDB is (B, 32, 10, 200).
    DSAINet expects (B, 1, C, T), so we reshape to (B, 1, 32, 2000).
    """

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.model = DSAINet(
            n_classes=params.num_of_classes,
            Chans=params.chans,
            Samples=params.samples,
            emb_size=params.emb_size,
            heads=params.heads,
            attn_depth=params.attn_depth,
            attn_dropout=params.attn_dropout,
            eeg1_f1=params.eeg1_f1,
            eeg1_kernel_size=params.eeg1_kernel_size,
            eeg1_D=params.eeg1_D,
            eeg1_pooling_size1=params.eeg1_pooling_size1,
            eeg1_pooling_size2=params.eeg1_pooling_size2,
            eeg1_dropout_rate=params.eeg1_dropout_rate,
            branch_1_kernels=params.branch_1_kernels,
            branch_2_kernels=params.branch_2_kernels,
            conv_expansion=params.conv_expansion,
            conv_dropout=params.conv_dropout,
            intra_ffn_expansion=params.intra_ffn_expansion,
            inter_ffn_expansion=params.inter_ffn_expansion,
            big_residual=params.big_residual,
            big_residual_learnable=params.big_residual_learnable,
            cls_dropout=params.dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,32,10,200) or (B,32,T)
        if x.dim() == 4:
            # (B, C, S, P) -> (B, C, T)
            x = x.reshape(x.shape[0], x.shape[1], -1)
        if x.dim() != 3:
            raise ValueError(f"Unexpected input shape: {tuple(x.shape)}")
        x = x.unsqueeze(1)  # (B,1,C,T)
        return self.model(x)


class TrainerDSAINet(object):
    def __init__(self, params, data_loader, model):
        self.params = params
        self.data_loader = data_loader

        self.val_eval = Evaluator(params, self.data_loader["val"])
        self.test_eval = Evaluator(params, self.data_loader["test"])

        self.model = model.cuda()
        self.criterion = CrossEntropyLoss(label_smoothing=self.params.label_smoothing).cuda()

        self.best_model_states = None

        if self.params.optimizer == "AdamW":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.params.lr, weight_decay=self.params.weight_decay
            )
        else:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.params.lr, momentum=0.9, weight_decay=self.params.weight_decay
            )

        self.data_length = len(self.data_loader["train"])
        self.optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.params.epochs * self.data_length, eta_min=1e-6
        )
        print(self.model)

    def train(self):
        f1_best = 0
        kappa_best = 0
        acc_best = 0
        cm_best = None
        best_epoch = 0

        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []

            for x, y in tqdm(self.data_loader["train"], mininterval=10):
                self.optimizer.zero_grad()
                x = x.cuda()
                y = y.long().cuda()

                pred = self.model(x)
                loss = self.criterion(pred, y)

                loss.backward()
                losses.append(loss.data.cpu().numpy())
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                self.optimizer.step()
                self.optimizer_scheduler.step()

            optim_state = self.optimizer.state_dict()

            with torch.no_grad():
                acc, kappa, f1, cm = self.val_eval.get_metrics_for_multiclass(self.model)
                print(
                    "Epoch {} : Training Loss: {:.5f}, acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}, LR: {:.6f}, Time elapsed {:.2f} mins".format(
                        epoch + 1,
                        float(np.mean(losses)),
                        acc,
                        kappa,
                        f1,
                        optim_state["param_groups"][0]["lr"],
                        (timer() - start_time) / 60,
                    )
                )
                print(cm)

                # Keep the same selection logic as the existing finetune_trainer: track best kappa.
                if kappa > kappa_best:
                    print("kappa increasing....saving weights !! ")
                    print("Val Evaluation: acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}".format(acc, kappa, f1))
                    best_epoch = epoch + 1
                    acc_best = acc
                    kappa_best = kappa
                    f1_best = f1
                    cm_best = cm
                    self.best_model_states = copy.deepcopy(self.model.state_dict())

        if self.best_model_states is not None:
            self.model.load_state_dict(self.best_model_states)

        with torch.no_grad():
            print("***************************Test************************")
            acc, kappa, f1, cm = self.test_eval.get_metrics_for_multiclass(self.model)
            print("***************************Test results************************")
            print("Test Evaluation: acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}".format(acc, kappa, f1))
            print(cm)

            if not os.path.isdir(self.params.model_dir):
                os.makedirs(self.params.model_dir)
            model_path = os.path.join(
                self.params.model_dir,
                "dsainet_epoch{}_acc_{:.5f}_kappa_{:.5f}_f1_{:.5f}.pth".format(best_epoch, acc, kappa, f1),
            )
            torch.save(self.model.state_dict(), model_path)
            print("model save in " + model_path)

        # Print best val summary for quick reference
        if cm_best is not None:
            print("***************************Best Val************************")
            print("Best Val: epoch {}, acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}".format(best_epoch, acc_best, kappa_best, f1_best))
            print(cm_best)


def parse_args():
    parser = argparse.ArgumentParser(description="DSAINet training on FACED (LMDB processed)")

    # basic training
    parser.add_argument("--seed", type=int, default=3407, help="random seed")
    parser.add_argument("--cuda", type=int, default=0, help="cuda number")
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-2, help="weight decay")
    parser.add_argument("--optimizer", type=str, default="AdamW", help="optimizer (AdamW, SGD)")
    parser.add_argument("--clip_value", type=float, default=1.0, help="clip grad norm (<=0 to disable)")
    parser.add_argument("--num_workers", type=int, default=0, help="num_workers (DataLoader in dataset file does not set this)")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="label_smoothing")

    # dataset
    parser.add_argument("--datasets_dir", type=str, default="/data/datasets/BigDownstream/Faced/processed", help="LMDB dir")
    parser.add_argument("--num_of_classes", type=int, default=2, help="number of classes for FACED (0-8)")
    parser.add_argument("--model_dir", type=str, default="runs/dsainet_faced", help="output dir to save weights")

    # FACED to DSAINet shape
    parser.add_argument("--chans", type=int, default=17, help="number of channels")
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


def main():
    params = parse_args()
    print(params)

    setup_seed(params.seed)
    torch.cuda.set_device(params.cuda)

    # Reuse the repo's FACED LMDB loader.
    load_dataset = faced_dataset.LoadDataset(params)
    data_loader = load_dataset.get_data_loader()

    # Overwrite batch_size in loader if user changed it after dataset init,
    # because faced_dataset.LoadDataset reads params.batch_size only at construction time.
    # (The typical usage is passing params in once, so this is just defensive.)
    for split in ["train", "val", "test"]:
        if hasattr(data_loader[split], "batch_size") and data_loader[split].batch_size != params.batch_size:
            data_loader[split].batch_size = params.batch_size

    model = DSAINetAdapter(params)
    t = TrainerDSAINet(params, data_loader, model)
    t.train()

    print("Done!!!!!")


if __name__ == "__main__":
    main()


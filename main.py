import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import hydra
import hydra.core.hydra_config
import numpy as np
import torch
import torch.backends.cudnn
import torch.nn.functional as F
import torch.utils.data
import wandb
from omegaconf import DictConfig
from termcolor import cprint
from torchmetrics import Accuracy
from tqdm import tqdm

from src.datasets import ThingsMEGDataset
from src.models import NewConvClassifier
from src.utils import set_seed


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    # 以下の記事に従って変更を加えた
    # https://qiita.com/sugulu_Ogawa_ISID/items/62f5f7adee083d96a587
    torch.backends.cudnn.benchmark = True

    # only for debugging
    # torch.autograd.set_detect_anomaly(True)

    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    scaler = torch.cuda.amp.GradScaler()

    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="MEG-classification")

    # ------------------
    #    Dataloader
    # ------------------
    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": True,
    }

    train_set = ThingsMEGDataset("train", args.data_dir)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)
    val_set = ThingsMEGDataset("val", args.data_dir)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_args)
    test_set = ThingsMEGDataset("test", args.data_dir)
    test_loader = torch.utils.data.DataLoader(test_set, shuffle=False, **loader_args)

    # ------------------
    #       Model
    # ------------------
    model = NewConvClassifier(
        train_set.num_classes, train_set.seq_len, train_set.num_channels
    ).to(args.device)

    # ------------------
    #     Optimizer
    # ------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ------------------
    #   Start training
    # ------------------
    max_val_acc = 0
    accuracy = Accuracy(
        task="multiclass", num_classes=train_set.num_classes, top_k=10
    ).to(args.device)

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")

        train_loss, train_acc, val_loss, val_acc = [], [], [], []

        model.train()
        for X, y, subject_idxs in tqdm(train_loader, desc="Train"):
            X, y = X.to(args.device, non_blocking=True), y.to(
                args.device, non_blocking=True
            )

            # Runs the forward pass with autocasting.
            with torch.cuda.amp.autocast():
                y_pred = model(X)
                loss = F.cross_entropy(y_pred, y)

                alpha = 0.01  # 正則化パラメータ
                # L2正則化
                l2 = torch.tensor(0.0, requires_grad=True)
                for w in model.parameters():
                    l2 = l2 + torch.norm(w) ** 2
                loss = loss + alpha * l2

            train_loss.append(loss.item())

            optimizer.zero_grad(set_to_none=True)

            # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
            scaler.scale(loss).backward()

            # scaler.step() first unscales the gradients of the optimizer's assigned params.
            scaler.step(optimizer)

            # Updates the scale for next iteration.
            scaler.update()

            acc = accuracy(y_pred, y)
            train_acc.append(acc.item())

        model.eval()
        for X, y, subject_idxs in tqdm(val_loader, desc="Validation"):
            X, y = X.to(args.device), y.to(args.device)

            with torch.no_grad():
                y_pred = model(X)

            val_loss.append(F.cross_entropy(y_pred, y).item())
            val_acc.append(accuracy(y_pred, y).item())

        print(
            f"Epoch {epoch+1}/{args.epochs} | train loss: {np.mean(train_loss):.3f} | train acc: {np.mean(train_acc):.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {np.mean(val_acc):.3f}"
        )
        torch.save(model.state_dict(), os.path.join(logdir, "model_last.pt"))
        if args.use_wandb:
            wandb.log(
                {
                    "train_loss": np.mean(train_loss),
                    "train_acc": np.mean(train_acc),
                    "val_loss": np.mean(val_loss),
                    "val_acc": np.mean(val_acc),
                }
            )

        if np.mean(val_acc) > max_val_acc:
            cprint("New best.", "cyan")
            torch.save(model.state_dict(), os.path.join(logdir, "model_best.pt"))
            max_val_acc = np.mean(val_acc)

    # ----------------------------------
    #  Start evaluation with best model
    # ----------------------------------
    model.load_state_dict(
        torch.load(os.path.join(logdir, "model_best.pt"), map_location=args.device)
    )

    preds = []
    model.eval()
    for X, subject_idxs in tqdm(test_loader, desc="Validation"):
        preds.append(model(X.to(args.device)).detach().cpu())

    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(logdir, "submission"), preds)
    cprint(f"Submission {preds.shape} saved at {logdir}", "cyan")


if __name__ == "__main__":
    run()

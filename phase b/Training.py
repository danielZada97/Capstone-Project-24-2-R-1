import random
import math
import os
import re
import numpy as np
import pandas as pd
import timm
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import KFold
from collections import OrderedDict
from torch_optimizer import Lookahead
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

import albumentations as A
from transformers import get_cosine_schedule_with_warmup

# Import the extra metrics we need
from sklearn.metrics import precision_score, f1_score, recall_score
from torch.nn import CrossEntropyLoss
# 1. Your dataset class import
from MyDataset import Mydataset

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    # ------------------------------
    # Define constants & hyperparams
    # ------------------------------
    SEED = 8620
    NOT_DEBUG = True

    # Lower resolution from 512→256 (helps memory a lot).
    IMG_SIZE = [256, 256]

    # If your GPU is only 4GB, start small:
    BATCH_SIZE = 32
    GRAD_ACC = 8
    N_WORKERS = 2
    PIN_MEMORY = False

    # Reduced input channels from 30→15
    IN_CHANS = 15

    # Model & training config
    MODEL_NAME = "densenet121"
    DROPOUT = 0.5
    N_LABELS = 25
    N_CLASSES = 3 * N_LABELS  # 25 labels × 3 classes each = 75

    # Paths
    DATA_PATH = "C:/Users/danie/OneDrive/Desktop/פרויקט סוף/part b/cvt_png"
    OUTPUT_DIR = "C:/Users/danie/OneDrive/Desktop/פרויקט סוף/part b/rsna24-results"
    rd = "C:/Users/danie/OneDrive/Desktop/פרויקט סוף/rsna-2024-lumbar-spine-degenerative-classification"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    EPOCHS = 50 if NOT_DEBUG else 2
    LR = 1e-7
    WD = 1e-2
    MAX_GRAD_NORM = 1
    USE_AMP = True

    # KFold config
    SELECTED_FOLDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    N_FOLDS = 11 if NOT_DEBUG else 2

    # Aug settings
    AUG = True
    AUG_PROB = 0.75

    def set_random_seed(seed: int = 2222, deterministic: bool = False):
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = deterministic

    set_random_seed(SEED)

    def initialize_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # ------------------------------
    # Load & preprocess DataFrame
    # ------------------------------
    df = pd.read_csv(f"{rd}/train.csv")
    df = df.fillna(-100)  # missing labels become -100
    label2id = {"Normal/Mild": 0, "Moderate": 1, "Severe": 2}
    df = df.replace(label2id)

    print("DataFrame head:\n", df.head())
    print("Total samples:", len(df))

    # --------------------------------
    # Albumentations transforms
    # --------------------------------
    transforms_train = A.Compose([
        A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussianBlur(p=0.3),
        A.ElasticTransform(p=0.2),
        A.GridDistortion(p=0.2),
        A.Normalize(mean=[0.5]*IN_CHANS, std=[0.5]*IN_CHANS),

    ])
    transforms_val = A.Compose([
        A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
        A.Normalize(mean=[0.5]*IN_CHANS, std=[0.5]*IN_CHANS)
    ])

    if not AUG:
        transforms_train = transforms_val

    # ------------------------------
    # Define a simple DenseNet
    # ------------------------------
    class DenseNet(nn.Module):
        def __init__(self, model_name, in_c=15, n_classes=75, pretrained=True):
            super().__init__()
            self.model = timm.create_model(
                model_name,
                pretrained=pretrained,
                in_chans=in_c,
                num_classes=n_classes,
                global_pool="avg"
            )
            self.dropout = nn.Dropout(p=DROPOUT)

        def forward(self, x):
            out = self.model(x)
            out = self.dropout(out)
            return out

    def compute_accuracy(predictions, targets):
        """
        predictions: (B, 3) for each sub-label
        targets: (B,) with values [0,1,2] or ignore_index=-100
        """
        pred_classes = predictions.argmax(dim=1)
        correct = (pred_classes == targets).sum().item()
        total = targets.numel()
        return correct / total if total > 0 else 0.0

    skf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    best_accuracy = 0.0

    all_fold_train_losses = []
    all_fold_val_losses = []
    all_fold_train_accuracies = []
    all_fold_val_accuracies = []
    all_fold_train_precisions = []
    all_fold_val_precisions = []
    all_fold_train_f1s = []
    all_fold_val_f1s = []
    # Add containers for recall:
    all_fold_train_recalls = []
    all_fold_val_recalls = []

    # For AMP
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP, init_scale=4096)

    for fold, (trn_idx, val_idx) in enumerate(skf.split(df)):
        if fold not in SELECTED_FOLDS:
            print(f"Skipping fold {fold}")
            continue

        print(f"\n=== Fold {fold} ===")

        df_train = df.iloc[trn_idx].reset_index(drop=True)
        df_valid = df.iloc[val_idx].reset_index(drop=True)

        print(
            f"Training samples: {len(df_train)}, Validation samples: {len(df_valid)}")

        train_ds = Mydataset(df_train, phase='train',
                             transform=transforms_train, data_path=DATA_PATH)
        valid_ds = Mydataset(df_valid, phase='valid',
                             transform=transforms_val, data_path=DATA_PATH)

        train_dl = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            pin_memory=PIN_MEMORY,
            drop_last=True,
            num_workers=N_WORKERS
        )
        valid_dl = DataLoader(
            valid_ds,
            batch_size=BATCH_SIZE * 2,
            shuffle=False,
            pin_memory=PIN_MEMORY,
            drop_last=False,
            num_workers=N_WORKERS
        )

        # Create model
        model = DenseNet(MODEL_NAME, IN_CHANS, N_CLASSES,
                         pretrained=True).to(device)

        # Optimizer & Scheduler
        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WD)
        optimizer = Lookahead(optimizer, k=5, alpha=0.5)

        warmup_steps = int(EPOCHS / 10 * len(train_dl) // GRAD_ACC)
        total_steps = int(EPOCHS * len(train_dl) // GRAD_ACC)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            num_cycles=0.475,
        )

        class FocalLoss(nn.Module):

            def __init__(self, alpha=1, gamma=2, reduction='mean'):
                super(FocalLoss, self).__init__()
                self.alpha = alpha
                self.gamma = gamma
                self.ce = CrossEntropyLoss(reduction='none')

            def forward(self, inputs, targets):
                logpt = -self.ce(inputs, targets)
                pt = torch.exp(logpt)
                loss = -((1-pt)**self.gamma) * logpt
                if self.alpha is not None:
                    loss = self.alpha * loss
                return loss.mean()

        criterion = FocalLoss()

        weights = torch.tensor([1, 2.0, 4.0], device=device)
       # criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)

        fold_train_losses = []
        fold_val_losses = []
        fold_train_accuracies = []
        fold_val_accuracies = []
        fold_train_precisions = []
        fold_val_precisions = []
        fold_train_f1s = []
        fold_val_f1s = []
        fold_train_recalls = []  # to store recall values for each epoch
        fold_val_recalls = []    # to store recall values for each epoch

        for epoch in range(1, EPOCHS + 1):
            print(f"\n[Fold {fold}] Epoch {epoch}/{EPOCHS}")

            # -------------------------
            # TRAINING
            # -------------------------
            model.train()

            total_loss = 0.0
            total_accuracy = 0.0

            # Collect predictions & targets to compute Precision/F1/Recall
            train_preds_all = []
            train_gts_all = []

            optimizer.zero_grad(set_to_none=True)

            for step, (x, t) in enumerate(tqdm(train_dl, desc="Train", leave=False)):
                x, t = x.to(device), t.to(device)

                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    logits = model(x)

                    batch_loss = 0.0
                    batch_acc = 0.0

                    # We will gather predictions & ground truths for each sub-label
                    sub_pred_list = []
                    sub_gt_list = []

                    for col in range(N_LABELS):
                        pred = logits[:, col * 3: col * 3 + 3]
                        gt = t[:, col]

                        if (gt < -100).any() or (gt > 2).any():
                            # invalid label, skip entire batch
                            batch_loss = None
                            break

                        sub_loss = criterion(pred, gt)
                        if not torch.isfinite(sub_loss):
                            batch_loss = None
                            break

                        batch_loss += sub_loss / N_LABELS
                        batch_acc += compute_accuracy(pred, gt) / N_LABELS

                        valid_mask = (gt >= 0) & (gt <= 2)
                        if valid_mask.sum() > 0:
                            sub_pred = pred[valid_mask].argmax(dim=1)
                            sub_gt = gt[valid_mask]
                            sub_pred_list.append(
                                sub_pred.detach().cpu().numpy())
                            sub_gt_list.append(sub_gt.detach().cpu().numpy())

                if batch_loss is None:
                    optimizer.zero_grad(set_to_none=True)
                    continue

                total_loss += batch_loss.item()
                total_accuracy += batch_acc

                if len(sub_pred_list) > 0:
                    sub_pred_list = np.concatenate(sub_pred_list)
                    sub_gt_list = np.concatenate(sub_gt_list)
                    train_preds_all.append(sub_pred_list)
                    train_gts_all.append(sub_gt_list)

                # Gradient Accumulation
                (scaler.scale(batch_loss) / GRAD_ACC).backward()

                if (step + 1) % GRAD_ACC == 0:
                    scaler.unscale_(optimizer)
                    if MAX_GRAD_NORM:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), MAX_GRAD_NORM)

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    if scheduler is not None:
                        scheduler.step()

            train_epoch_loss = total_loss / len(train_dl)
            train_epoch_acc = total_accuracy / len(train_dl)

            if len(train_preds_all) > 0:
                all_preds_np = np.concatenate(train_preds_all)
                all_gts_np = np.concatenate(train_gts_all)

                train_precision = precision_score(
                    all_gts_np, all_preds_np, average="macro", zero_division=0
                )
                train_f1 = f1_score(
                    all_gts_np, all_preds_np, average="macro", zero_division=0
                )
                # ---- Compute recall as well ----
                train_recall = recall_score(
                    all_gts_np, all_preds_np, average="macro", zero_division=0
                )
            else:
                train_precision = 0.0
                train_f1 = 0.0
                train_recall = 0.0

            print(
                f"  Train Loss = {train_epoch_loss:.4f},"
                f" Train Acc = {train_epoch_acc:.4f},"
                f" Precision = {train_precision:.4f},"
                f" Recall = {train_recall:.4f},"
                f" F1 = {train_f1:.4f}"
            )

            fold_train_losses.append(train_epoch_loss)
            fold_train_accuracies.append(train_epoch_acc)
            fold_train_precisions.append(train_precision)
            fold_train_f1s.append(train_f1)
            fold_train_recalls.append(train_recall)

            # -------------------------
            # VALIDATION
            # -------------------------
            model.eval()
            val_loss_accum = 0.0
            val_acc_accum = 0.0

            val_preds_all = []
            val_gts_all = []

            with torch.no_grad():
                for x_val, t_val in tqdm(valid_dl, desc="Valid", leave=False):
                    x_val, t_val = x_val.to(device), t_val.to(device)
                    with torch.cuda.amp.autocast(enabled=USE_AMP):
                        logits_val = model(x_val)

                        v_loss = 0.0
                        v_acc = 0.0
                        sub_pred_list = []
                        sub_gt_list = []

                        for col in range(N_LABELS):
                            pred = logits_val[:, col * 3: col * 3 + 3]
                            gt = t_val[:, col]

                            if (gt < -100).any() or (gt > 2).any():
                                v_loss = None
                                break
                            sub_loss = criterion(pred, gt)
                            if not torch.isfinite(sub_loss):
                                v_loss = None
                                break
                            v_loss += sub_loss / N_LABELS
                            v_acc += compute_accuracy(pred, gt) / N_LABELS

                            valid_mask = (gt >= 0) & (gt <= 2)
                            if valid_mask.sum() > 0:
                                sub_pred = pred[valid_mask].argmax(dim=1)
                                sub_gt = gt[valid_mask]
                                sub_pred_list.append(
                                    sub_pred.detach().cpu().numpy())
                                sub_gt_list.append(
                                    sub_gt.detach().cpu().numpy())

                    if v_loss is not None:
                        val_loss_accum += v_loss.item()
                        val_acc_accum += v_acc

                        if len(sub_pred_list) > 0:
                            sub_pred_list = np.concatenate(sub_pred_list)
                            sub_gt_list = np.concatenate(sub_gt_list)
                            val_preds_all.append(sub_pred_list)
                            val_gts_all.append(sub_gt_list)

                if len(valid_dl) > 0:
                    val_epoch_loss = val_loss_accum / len(valid_dl)
                    val_epoch_acc = val_acc_accum / len(valid_dl)
                else:
                    val_epoch_loss = 999.0
                    val_epoch_acc = 0.0

                if len(val_preds_all) > 0:
                    all_preds_np = np.concatenate(val_preds_all)
                    all_gts_np = np.concatenate(val_gts_all)
                    val_precision = precision_score(
                        all_gts_np, all_preds_np, average="macro", zero_division=0
                    )
                    val_f1 = f1_score(
                        all_gts_np, all_preds_np, average="macro", zero_division=0
                    )
                    # ---- Compute recall as well ----
                    val_recall = recall_score(
                        all_gts_np, all_preds_np, average="macro", zero_division=0
                    )
                else:
                    val_precision = 0.0
                    val_f1 = 0.0
                    val_recall = 0.0

            print(
                f"  Valid Loss = {val_epoch_loss:.4f},"
                f" Valid Acc = {val_epoch_acc:.4f},"
                f" Precision = {val_precision:.4f},"
                f" Recall = {val_recall:.4f},"
                f" F1 = {val_f1:.4f}"
            )

            fold_val_losses.append(val_epoch_loss)
            fold_val_accuracies.append(val_epoch_acc)
            fold_val_precisions.append(val_precision)
            fold_val_f1s.append(val_f1)
            fold_val_recalls.append(val_recall)

            # Save best model based on accuracy (or F1 if you prefer)
            if val_epoch_acc > best_accuracy:
                print(
                    f"  [Improved] Accuracy: {best_accuracy:.4f} -> {val_epoch_acc:.4f}"
                )
                best_accuracy = val_epoch_acc
                best_model_path = (
                    f"{OUTPUT_DIR}/best_model_fold{fold}_LR={LR}_acc={best_accuracy:.4f}.pt"
                )
                torch.save(model.state_dict(), best_model_path)
                print(f"  Saved best model to {best_model_path}")

        print(f"\nFold {fold} done.")

        # -------------------------
        # Save metrics for fold
        # -------------------------
        df_fold_results = pd.DataFrame({
            "epoch": range(1, EPOCHS + 1),
            "train_loss": fold_train_losses,
            "val_loss": fold_val_losses,
            "train_acc": fold_train_accuracies,
            "val_acc": fold_val_accuracies,
            "train_precision": fold_train_precisions,
            "val_precision": fold_val_precisions,
            "train_recall": fold_train_recalls,
            "val_recall": fold_val_recalls,
            "train_f1": fold_train_f1s,
            "val_f1": fold_val_f1s,
        })
        csv_path = f"{OUTPUT_DIR}/loss_acc_scores_fold_{fold}_lr={LR}.csv"
        df_fold_results.to_csv(csv_path, index=False)
        print(f"Saved fold {fold} metrics -> {csv_path}")

        all_fold_train_losses.append(fold_train_losses)
        all_fold_val_losses.append(fold_val_losses)
        all_fold_train_accuracies.append(fold_train_accuracies)
        all_fold_val_accuracies.append(fold_val_accuracies)
        all_fold_train_precisions.append(fold_train_precisions)
        all_fold_val_precisions.append(fold_val_precisions)
        all_fold_train_f1s.append(fold_train_f1s)
        all_fold_val_f1s.append(fold_val_f1s)
        all_fold_train_recalls.append(fold_train_recalls)
        all_fold_val_recalls.append(fold_val_recalls)

    print("\nAll done!")

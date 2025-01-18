import random
import math
from glob import glob
from PIL import Image
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from collections import OrderedDict
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
import timm
from transformers import get_cosine_schedule_with_warmup
import albumentations as A

# Import your custom dataset class
from MyDataset import Mydataset

# Adjust to point to your CSV file directory
rd = "C:/Users/danie/OneDrive/Desktop/פרויקט סוף/rsna-2024-lumbar-spine-degenerative-classification"

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        return [atoi(c) for c in re.split(r'(\d+)', text)]

    # ------------------------------
    # Define constants & hyperparams
    # ------------------------------
    NOT_DEBUG = True
    DATA_PATH = 'C:/Users/danie/OneDrive/Desktop/פרויקט סוף/part b/cvt_png'
    OUTPUT_DIR = "C:/Users/danie/OneDrive/Desktop/פרויקט סוף/part b/rsna24-results"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # On Windows, keep single-process data loading to avoid issues
    N_WORKERS = 0  # or 6 if it works for you without issues

    USE_AMP = True
    SEED = 8620

    # Image & label configuration
    IMG_SIZE = [512, 512]
    IN_CHANS = 30
    N_LABELS = 25
    N_CLASSES = 3 * N_LABELS  # 25 labels × 3 classes each = 75
    DROPOUT = 0.2

    # Training config
    best_accuracy = 0.0

    AUG_PROB = 0.75
    SELECTED_FOLDS = [0, 1, 2]
    N_FOLDS = 3 if NOT_DEBUG else 2
    EPOCHS = 50 if NOT_DEBUG else 2
    MODEL_NAME = "densenet121"
    GRAD_ACC = 2
    TGT_BATCH_SIZE = 32

    BATCH_SIZE = 4  # Keep batch size small to avoid OOM

    MAX_GRAD_NORM = None
    # EARLY_STOPPING_EPOCH = 3

    LR = 1e-4 * TGT_BATCH_SIZE / 32
    WD = 1e-2
    AUG = True

    CONDITIONS = [
        'Spinal Canal Stenosis',
        'Left Neural Foraminal Narrowing',
        'Right Neural Foraminal Narrowing',
        'Left Subarticular Stenosis',
        'Right Subarticular Stenosis'
    ]

    LEVELS = [
        'L1/L2',
        'L2/L3',
        'L3/L4',
        'L4/L5',
        'L5/S1',
    ]

    def set_random_seed(seed: int = 2222, deterministic: bool = False):
        """Set seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = deterministic  # type: ignore

    # Set seeds
    set_random_seed(SEED)
    # -------------------------------
    # start training and validating loop, announce each run what are the parameters here:
    print(
        f"parameters:\n LEARNING RATE={LR}\n DROPOUT= {DROPOUT}\n EPOCHS={EPOCHS}")

    # -------------------------------
    # ------------------------------
    # Load & preprocess DataFrame
    # ------------------------------
    df = pd.read_csv(f'{rd}/train.csv')
    df = df.fillna(-100)  # missing labels become -100
    label2id = {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}
    df = df.replace(label2id)
    print("DataFrame head:\n", df.head())
    print("Total samples:", len(df))

    # If debugging, optionally sample a subset:
    # df = df.sample(50, random_state=SEED).reset_index(drop=True)

    # ------------------------------
    # Albumentations transforms
    # ------------------------------
    transforms_train = A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=AUG_PROB),
        A.OneOf([
            A.MotionBlur(blur_limit=5),
            A.MedianBlur(blur_limit=5),
            A.GaussianBlur(blur_limit=5),
            A.GaussNoise(var_limit=(5.0, 30.0)),
        ], p=AUG_PROB),
        A.OneOf([
            A.OpticalDistortion(distort_limit=1.0),
            A.GridDistortion(num_steps=5, distort_limit=1.),
            A.ElasticTransform(alpha=3),
        ], p=AUG_PROB),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1,
                           rotate_limit=15, border_mode=0, p=AUG_PROB),
        A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
        A.CoarseDropout(max_holes=16, max_height=64, max_width=64,
                        min_holes=1, min_height=8, min_width=8, p=AUG_PROB),
        A.Normalize(mean=0.5, std=0.5)
    ])

    transforms_val = A.Compose([
        A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
        A.Normalize(mean=0.5, std=0.5)
    ])

    if not NOT_DEBUG or not AUG:
        transforms_train = transforms_val

    # Quick check: Single-batch load
    tmp_ds = Mydataset(df, phase='train', transform=transforms_train,
                       data_path=DATA_PATH)
    tmp_dl = DataLoader(
        tmp_ds,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=0
    )
    for i, (x, t) in enumerate(tmp_dl):
        if i == 2:
            break
        print(
            f"Sample {i} | x shape: {x.shape} | x min: {x.min()}, x max: {x.max()}")
        print(f"Labels shape: {t.shape}, labels: {t}")
    plt.close()
    del tmp_ds, tmp_dl

    # ------------------------------
    # Define the DenseNet model
    # ------------------------------
    class DenseNet(nn.Module):
        def __init__(self, model_name, in_c=30, n_classes=75,
                     pretrained=True, features_only=False):
            super().__init__()
            self.model = timm.create_model(
                model_name,
                pretrained=pretrained,
                features_only=features_only,
                in_chans=in_c,
                num_classes=n_classes,
                global_pool='avg'
            )
            self.dropout = nn.Dropout(p=DROPOUT)

        def forward(self, x):
            return self.model(x)

    # Quick check: forward pass
    test_model = DenseNet(MODEL_NAME, in_c=IN_CHANS,
                          n_classes=N_CLASSES, pretrained=False)
    test_input = torch.randn(2, IN_CHANS, 512, 512)
    test_out = test_model(test_input)
    print("test_out shape:", test_out.shape)
    del test_model, test_input, test_out

    # ------------------------------
    # AMP & GradScaler
    # ------------------------------
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP, init_scale=4096)

    # ------------------------------
    # Define accuracy function
    # ------------------------------
    def compute_accuracy(predictions, targets):
        """
        predictions: (B, 3) for each sub-label
        targets: (B,) with values [0,1,2] or ignore_index=-100
        """
        pred_classes = predictions.argmax(dim=1)  # shape (B,)
        correct = (pred_classes == targets).sum().item()
        total = targets.numel()
        return correct / total if total > 0 else 0

    # ------------------------------
    # Cross-validation
    # ------------------------------
    skf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # We'll store *all* folds' epoch-by-epoch metrics here (optional):
    all_fold_train_losses = []
    all_fold_val_losses = []
    all_fold_train_accuracies = []
    all_fold_val_accuracies = []

    for fold, (trn_idx, val_idx) in enumerate(skf.split(range(len(df)))):
        print(f"\n=== Fold {fold} ===")
        if fold not in SELECTED_FOLDS:
            print(f"Jump fold {fold}")
            continue

        df_train = df.iloc[trn_idx].reset_index(drop=True)
        df_valid = df.iloc[val_idx].reset_index(drop=True)

        print(
            f"Training samples: {len(df_train)}, Validation samples: {len(df_valid)}")

        train_ds = Mydataset(df_train, phase='train',
                             transform=transforms_train, data_path=DATA_PATH)
        valid_ds = Mydataset(df_valid, phase='valid', transform=transforms_val,
                             data_path=DATA_PATH)

        train_dl = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=N_WORKERS
        )
        valid_dl = DataLoader(
            valid_ds,
            batch_size=BATCH_SIZE * 2,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=N_WORKERS
        )

        model = DenseNet(MODEL_NAME, IN_CHANS, N_CLASSES,
                         pretrained=True).to(device)
        fname = f'{OUTPUT_DIR}/best_wll_model_fold-{fold}lr={LR}-acc={best_accuracy:.4f}.pt'

        # If there's a previously saved model, load it
        if os.path.exists(fname):
            model = DenseNet(MODEL_NAME, IN_CHANS, N_CLASSES,
                             pretrained=False).to(device)
            model.load_state_dict(torch.load(fname))
            print(f"Loaded existing model weights from {fname}")

        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WD)
        warmup_steps = int(EPOCHS / 10 * len(train_dl) // GRAD_ACC)
        num_total_steps = int(EPOCHS * len(train_dl) // GRAD_ACC)
        num_cycles = 0.475
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_total_steps,
            num_cycles=num_cycles
        )

        # Weighted CrossEntropyLoss with ignore_index for -100
        weights = torch.tensor([1.0, 2.0, 4.0], device=device)
        criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)

        best_loss = float('inf')

        # Per-fold metrics we will save after training completes
        fold_train_losses = []
        fold_val_losses = []
        fold_train_accuracies = []
        fold_val_accuracies = []

        # ------------------------------
        # Training Loop
        # ------------------------------
        for epoch in range(1, EPOCHS + 1):
            print(f"\nStart epoch {epoch}/{EPOCHS}, Fold {fold}")
            model.train()
            total_loss = 0.0
            total_accuracy = 0.0

            optimizer.zero_grad(set_to_none=True)

            with tqdm(train_dl, desc=f"Train Epoch {epoch}", leave=True) as pbar:
                for idx, (x, t) in enumerate(pbar):
                    x = x.to(device)
                    t = t.to(device)

                    with torch.cuda.amp.autocast(enabled=USE_AMP):
                        logits = model(x)
                        batch_loss = 0.0
                        batch_accuracy = 0.0

                        # Summation over the 25 sub-labels
                        for col in range(N_LABELS):
                            # shape (B,3)
                            pred = logits[:, col * 3: col * 3 + 3]
                            # shape (B,)
                            gt = t[:, col]

                            # If ignore index is found or invalid labels outside [0..2]
                            if (gt < -100).any() or (gt > 2).any():
                                batch_loss = None
                                break

                            sub_loss = criterion(pred, gt)
                            if not torch.isfinite(sub_loss):
                                batch_loss = None
                                break

                            batch_loss += sub_loss / N_LABELS
                            batch_accuracy += compute_accuracy(
                                pred, gt) / N_LABELS

                    if batch_loss is None:
                        print(
                            f"[Warning] Invalid sub-loss in batch {idx}, skipping.")
                        optimizer.zero_grad(set_to_none=True)
                        continue

                    total_loss += batch_loss.item()
                    total_accuracy += batch_accuracy

                    # Gradient accumulation
                    loss_for_backward = batch_loss / GRAD_ACC
                    scaler.scale(loss_for_backward).backward()

                    if (idx + 1) % GRAD_ACC == 0:
                        scaler.unscale_(optimizer)
                        if MAX_GRAD_NORM:
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), MAX_GRAD_NORM
                            )

                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)

                        if scheduler is not None:
                            scheduler.step()

                    pbar.set_postfix(OrderedDict(
                        loss=f"{batch_loss.item():.6f}",
                        acc=f"{batch_accuracy:.4f}"
                    ))

            epoch_loss = total_loss / len(train_dl)
            epoch_accuracy = total_accuracy / len(train_dl)
            torch.cuda.empty_cache()
            print(
                f"Train loss: {epoch_loss:.6f}, Train accuracy: {epoch_accuracy:.4f}")

            # Store training metrics for this epoch
            fold_train_losses.append(epoch_loss)
            fold_train_accuracies.append(epoch_accuracy)

            # ------------------------------
            # Validation Loop
            # ------------------------------
            model.eval()
            val_total = 0.0
            val_accuracy_total = 0.0

            with torch.no_grad():
                with tqdm(valid_dl, desc="Valid", leave=True) as pbar:
                    for x, t in pbar:
                        x, t = x.to(device), t.to(device)

                        with torch.cuda.amp.autocast(enabled=USE_AMP):
                            logits = model(x)
                            v_loss = 0.0
                            v_accuracy = 0.0
                            for col in range(N_LABELS):
                                pred = logits[:, col * 3: col * 3 + 3]
                                gt = t[:, col]
                                if (gt < -100).any() or (gt > 2).any():
                                    v_loss = None
                                    break
                                sub_loss = criterion(pred, gt)
                                if not torch.isfinite(sub_loss):
                                    v_loss = None
                                    break
                                v_loss += sub_loss / N_LABELS
                                v_accuracy += compute_accuracy(
                                    pred, gt) / N_LABELS

                        if v_loss is not None:
                            val_total += v_loss.item()
                            val_accuracy_total += v_accuracy

            if len(valid_dl) > 0:
                val_loss = val_total / len(valid_dl)
                val_accuracy = val_accuracy_total / len(valid_dl)
            else:
                val_loss = 999.0
                val_accuracy = 0.0

            print(
                f"Val loss: {val_loss:.6f}, Val accuracy: {val_accuracy:.4f}")

            # Store validation metrics for this epoch
            fold_val_losses.append(val_loss)
            fold_val_accuracies.append(val_accuracy)

            # ------------------------------
            # Check for improvement
            # ------------------------------
            if val_accuracy > best_accuracy:
                print(
                    f"Val accuracy improved from {best_accuracy:.4f} to {val_accuracy:.4f}")
                best_accuracy = val_accuracy
                fname = f'{OUTPUT_DIR}/best_wll_model_fold-{fold}lr={LR}-acc={best_accuracy:.4f}.pt'

                torch.save(model.state_dict(), fname)
                print(f"Model with best accuracy saved: {fname}")

        # ------------------------------
        # End of all epochs for this fold
        # ------------------------------
        print(f"\nFold {fold} training complete!\n")

        # Save the per-epoch scores into a CSV file for this fold
        df_fold_results = pd.DataFrame({
            "epoch": range(1, EPOCHS + 1),
            "train_loss": fold_train_losses,
            "val_loss": fold_val_losses,
            "train_acc": fold_train_accuracies,
            "val_acc": fold_val_accuracies
        })
        csv_path = f"{OUTPUT_DIR}/loss_acc_scores_fold_{fold}lr={LR}.csv"
        df_fold_results.to_csv(csv_path, index=False)
        print(f"Per-epoch scores saved to {csv_path}")

        # Optionally store them into global lists-of-lists
        all_fold_train_losses.append(fold_train_losses)
        all_fold_val_losses.append(fold_val_losses)
        all_fold_train_accuracies.append(fold_train_accuracies)
        all_fold_val_accuracies.append(fold_val_accuracies)

    print("All selected folds done!")
    print("\nSummary of folds (train losses):", all_fold_train_losses)
    print("Summary of folds (val losses):", all_fold_val_losses)
    print("Summary of folds (train accuracies):", all_fold_train_accuracies)
    print("Summary of folds (val accuracies):", all_fold_val_accuracies)

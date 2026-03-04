"""
trainer.py

Training / validation / evaluation logic for the MIL model.

Assumptions (matching the pipeline):
- DataLoader returns a "batch" that is a list with one dict (because batch_size=1 and case_collate_fn returns batch)
  So we access case_data = batch[0].
- model(stain_slices) returns logits of shape (num_classes,) (no batch dim).
- label is a scalar LongTensor (or shape (1,)).

Key corrections / improvements:
- Never reference non-existent DATA_PATHS['checkpoint_dir'].
- Always save predictions/confusion matrix into a provided output_dir (typically run_dir).
- Create output directories as needed.
- Robustness: handle empty loaders, consistent device handling, safer averaging, clearer early stopping logic.
"""

import time
import os
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
from sklearn.metrics import recall_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import TRAINING_CONFIG, DEVICE

def attention_entropy(w: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Entropy of attention weights AFTER softmax.
    w can be shape (..., M). Returns mean entropy (scalar).
    """
    w = w.clamp_min(eps)
    H = -(w * w.log()).sum(dim=-1)   # sum over attention dimension
    return H.mean()

class MILTrainer:
    """
    Trainer for MIL model.
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
    ):
        self.model = model
        self.device = device if device is not None else DEVICE
        self.checkpoint_dir = checkpoint_dir

        self.model.to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=TRAINING_CONFIG["learning_rate"],
            weight_decay=TRAINING_CONFIG["weight_decay"],
        )

        # Loss with class weights
        class_weights = torch.tensor(TRAINING_CONFIG["class_weights"], dtype=torch.float32, device=self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # LR scheduler
        self.scheduler = None
        if TRAINING_CONFIG.get("use_scheduler", False):
            scheduler_type = TRAINING_CONFIG.get("scheduler_type", "reduce_on_plateau")
            if scheduler_type == "reduce_on_plateau":
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode="min",
                    factor=TRAINING_CONFIG.get("scheduler_factor", 0.5),
                    patience=TRAINING_CONFIG.get("scheduler_patience", 3),
                    min_lr=TRAINING_CONFIG.get("scheduler_min_lr", 1e-6),
                )
            elif scheduler_type == "cosine":
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=TRAINING_CONFIG["epochs"],
                    eta_min=TRAINING_CONFIG.get("scheduler_min_lr", 1e-6),
                )
            else:
                raise ValueError(f"Unknown scheduler_type: {scheduler_type}")

        # Early stopping
        self.use_early_stopping = TRAINING_CONFIG.get("early_stopping", False)
        self.early_stopping_patience = TRAINING_CONFIG.get("early_stopping_patience", 7)
        self.early_stopping_min_delta = TRAINING_CONFIG.get("early_stopping_min_delta", 0.001)
        self.early_stopping_min_epochs = TRAINING_CONFIG.get("early_stopping_min_epochs", 0)

        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0

        # History
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.learning_rates = []

    # ----------------------------
    # Core helpers
    # ----------------------------
    def _calculate_entropy(self, weights: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
        """
        Task 3 ：H(w) = -sum(p * log(p + epsilon))
        """
        return -torch.sum(weights * torch.log(weights + epsilon), dim=-1).mean()

    def _get_all_entropies(self, attn: Dict[str, Any]) -> torch.Tensor:
        """
        Case -> Stain -> Slice -> Patch
        """
        entropies = []
        
        # 1. Case-level
        if "case_weights" in attn:
            entropies.append(self._calculate_entropy(attn["case_weights"]))
            
        # 2.Stain & Slice-level
        for _, d in attn.get("stain_weights", {}).items():
            if "slice_weights" in d:
                entropies.append(self._calculate_entropy(d["slice_weights"]))
            # 3.  Patch-level
            for pw in d.get("patch_weights", []):
                entropies.append(self._calculate_entropy(pw))
                
        return torch.stack(entropies).mean() if entropies else torch.tensor(0.0, device=self.device)

    def _forward_one_case(
        self,
        case_data: Dict[str, Any],
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Returns (logits_with_batch, label_with_batch, attn_dict_or_None)
        logits_with_batch: (1, num_classes)
        label_with_batch:  (1,)
        attn: dict of attention weights if return_attn=True else None
        """
        stain_slices = case_data["stain_slices"]
        label = case_data["label"].to(self.device)

        if return_attn:
            out = self.model(stain_slices, return_attn_weights=True)
            
            # if not isinstance(out, tuple) or len(out) != 2:
            #     raise RuntimeError("return_attn=True model didn't return (logits, attn_dict)")
            
            logits, attn = out
        else:
            logits, attn = self.model(stain_slices), None

        # if logits.dim() != 1:
        #     raise ValueError(f"Expected model to return (num_classes,), got {logits.shape}")
        
        logits = logits.unsqueeze(0)  # (1, num_classes)
        label = label.unsqueeze(0) if label.dim() == 0 else label # (1,)

        return logits, label, attn

    def _ensure_dir(self, path: Optional[str]) -> str:
        if path is None:
            raise ValueError("output_dir/checkpoint_dir must be provided (got None).")
        os.makedirs(path, exist_ok=True)
        return path

    # ----------------------------
    # Train / Validate
    # ----------------------------
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train on every epoch, including Task 3 Entropy logic
        """
        self.model.train()
        
        if len(train_loader) == 0:
            print("[WARN] train_loader is empty. Returning loss=0.0")
            self.train_losses.append(0.0)
            return 0.0

        running_loss = 0.0
        num_batches = 0

        # Get lambda
        lam = float(TRAINING_CONFIG.get("entropy_lambda", 0.0))
        want_attn = lam > 0.0

        for batch in tqdm(train_loader, desc="Training", leave=False):
            if not batch:
                continue
            
            case_data = batch[0]
            
            logits, label, attn = self._forward_one_case(case_data, return_attn=want_attn)
            
            # Cross Entropy
            base_loss = self.criterion(logits, label)

            # Task 3
            if want_attn and attn is not None:
                entropy_term = self._get_all_entropies(attn)
                # Loss = CE - lambda * Entropy 
                loss = base_loss - lam * entropy_term
            else:
                loss = base_loss

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            if want_attn and num_batches == 0:
                print(f"[ER DEBUG] lam={lam:.3e} base_loss={base_loss.item():.6f} "
                      f"entropy={entropy_term.item():.6f} lam*H={(lam*entropy_term).item():.6f}")

            running_loss += float(loss.item())
            num_batches += 1

        avg_loss = running_loss / max(num_batches, 1)
        self.train_losses.append(avg_loss)
        
        return avg_loss

    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate the model.
        Returns (avg_loss, accuracy).
        """
        self.model.eval()

        if len(val_loader) == 0:
            print("[WARN] val_loader is empty. Returning loss=0.0, acc=0.0")
            self.val_losses.append(0.0)
            self.val_accuracies.append(0.0)
            return 0.0, 0.0

        total_loss = 0.0
        correct = 0
        n = 0

        with torch.inference_mode():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                if not batch:
                    continue
                case_data = batch[0]

                logits, label, _ = self._forward_one_case(case_data, return_attn=False)
                loss = self.criterion(logits, label)
                total_loss += float(loss.item())

                pred = torch.argmax(logits, dim=1)  # (1,)
                correct += int((pred == label).sum().item())
                n += 1

        avg_loss = total_loss / max(n, 1)
        acc = correct / max(n, 1)

        self.val_losses.append(avg_loss)
        self.val_accuracies.append(acc)
        return avg_loss, acc

    # ----------------------------
    # Checkpointing
    # ----------------------------
    def save_checkpoint(
        self,
        epoch: int,
        arch: str = "HierarchicalAttnMIL",
        checkpoint_dir: Optional[str] = None,
        is_best: bool = False,
    ) -> str:
        """
        Save checkpoint and return filename.
        """
        ckpt_dir = checkpoint_dir if checkpoint_dir is not None else self.checkpoint_dir
        ckpt_dir = self._ensure_dir(ckpt_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(ckpt_dir, f"{timestamp}_{arch}_epoch{epoch}.pth")

        checkpoint = {
            "arch": arch,
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_accuracies": self.val_accuracies,
            "learning_rates": self.learning_rates,
            "best_val_loss": self.best_val_loss,
            "config": TRAINING_CONFIG,
        }
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, filename)
        print(f"Checkpoint saved: {filename}")

        if is_best:
            best_path = os.path.join(ckpt_dir, "best.pth")
            torch.save(checkpoint, best_path)
            print(f"Best checkpoint updated: {best_path}")

        return filename

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load model checkpoint.
        Returns: epoch number loaded (int).
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])
        self.val_accuracies = checkpoint.get("val_accuracies", [])
        self.learning_rates = checkpoint.get("learning_rates", [])
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        epoch = int(checkpoint.get("epoch", 0))
        print(f"Checkpoint loaded: {checkpoint_path} (epoch={epoch})")
        return epoch

    # ----------------------------
    # Full training loop
    # ----------------------------
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: Optional[int] = None,
        start_epoch: int = 0,
        save_every: int = 1,
        arch: str = "HierarchicalAttnMIL",
    ):
        """
        Full training loop with scheduler + early stopping.
        """
        if epochs is None:
            epochs = TRAINING_CONFIG["epochs"]

        print(f"Starting training from epoch {start_epoch + 1} to {epochs}")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        if self.scheduler is not None:
            print(f"LR scheduler: {type(self.scheduler).__name__}")
        if self.use_early_stopping:
            print(
                "Early stopping enabled "
                f"(patience={self.early_stopping_patience}, "
                f"min_delta={self.early_stopping_min_delta}, "
                f"min_epochs={self.early_stopping_min_epochs})"
            )

        for epoch in range(start_epoch, epochs):
            current_lr = float(self.optimizer.param_groups[0]["lr"])
            print(f"\nEpoch {epoch + 1}/{epochs} (LR: {current_lr:.2e})")
            epoch_start_time = time.time()

            train_loss = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)

            # Record LR used this epoch
            self.learning_rates.append(current_lr)

            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch Time: {epoch_time:.2f}s")
            # Scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Check best / early stop
            is_best = False
            if val_loss < (self.best_val_loss - self.early_stopping_min_delta):
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                is_best = True
                print(f"New best validation loss: {val_loss:.4f}")
            else:
                self.epochs_without_improvement += 1
                print(f"No improvement for {self.epochs_without_improvement} epoch(s)")

            # Save checkpoint periodically
            if save_every > 0 and ((epoch + 1) % save_every == 0):
                self.save_checkpoint(epoch + 1, arch=arch, is_best=is_best)

            # Early stopping
            if self.use_early_stopping:
                if (epoch + 1) >= self.early_stopping_min_epochs and self.epochs_without_improvement >= self.early_stopping_patience:
                    print(f"\nEarly stopping triggered at epoch {epoch + 1}. Best val loss: {self.best_val_loss:.4f}")
                    break

        print("\nTraining completed!")

    # ----------------------------
    # Evaluate + Outputs
    # ----------------------------
    def evaluate(
        self,
        test_loader: DataLoader,
        save_predictions: bool = True,
        output_dir: Optional[str] = None,
        checkpoint_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate model on test set.
        Returns dict with metrics + predictions.
        """
        self.model.eval()

        if len(test_loader) == 0:
            print("[WARN] test_loader is empty. Returning zeros.")
            results = {
                "test_loss": 0.0,
                "test_accuracy": 0.0,
                "predictions": [],
                "true_labels": [],
                "case_ids": [],
                "prediction_probs": [],
                "num_samples": 0,
            }
            return results

        if save_predictions:
            out_dir = self._ensure_dir(output_dir)
        else:
            out_dir = output_dir  # unused

        total_loss = 0.0
        correct = 0
        n = 0

        predictions = []
        true_labels = []
        case_ids = []
        prediction_probs = []

        with torch.inference_mode():
            for batch in tqdm(test_loader, desc="Evaluating"):
                if not batch:
                    continue
                case_data = batch[0]
                case_id = case_data.get("case_id", None)

                logits, label, _ = self._forward_one_case(case_data, return_attn=False)
                loss = self.criterion(logits, label)
                total_loss += float(loss.item())

                probs = torch.softmax(logits, dim=1)  # (1, num_classes)
                pred = torch.argmax(logits, dim=1)    # (1,)

                correct += int((pred == label).sum().item())
                n += 1

                case_ids.append(case_id)
                predictions.append(int(pred.cpu().item()))
                true_labels.append(int(label.cpu().item()))
                prediction_probs.append(probs.cpu().numpy()[0])  # [p0, p1, ...]

        avg_loss = total_loss / max(n, 1)
        acc = correct / max(n, 1)
        h_recall = recall_score(true_labels, predictions)
        b_recall = recall_score(true_labels, predictions, pos_label=0)

        results = {
            "test_loss": avg_loss,
            "test_accuracy": acc,
            "test_high_grade_recall": h_recall, 
            "test_benign_recall": b_recall, 
            "predictions": predictions,
            "true_labels": true_labels,
            "case_ids": case_ids,
            "prediction_probs": prediction_probs,
            "num_samples": n,
        }

        print("\nTest Results:")
        print(f"  Test Loss: {avg_loss:.4f}")
        print(f"  Test Acc:  {acc:.4f}")
        print(f"  Samples:   {n}")

        if save_predictions:
            self._save_predictions_csv(results, out_dir, checkpoint_name)
            self._save_confusion_matrix(results, out_dir)

        return results

    def _save_predictions_csv(
        self,
        results: Dict[str, Any],
        output_dir: str,
        checkpoint_name: Optional[str] = None,
    ) -> str:
        """
        Save per-case predictions to CSV in output_dir.
        """
        import pandas as pd

        self._ensure_dir(output_dir)

        csv_filename = "predictions.csv" if checkpoint_name is None else f"predictions_{os.path.basename(checkpoint_name)}.csv"
        # sanitize filename a bit
        csv_filename = csv_filename.replace(".pth", "").replace(" ", "_")
        csv_path = os.path.join(output_dir, csv_filename)

        # Works for binary, but also generalizes
        probs_arr = results["prediction_probs"]
        if len(probs_arr) > 0:
            num_classes = len(probs_arr[0])
        else:
            num_classes = 0

        df_data = {
            "case_id": results["case_ids"],
            "true_label": results["true_labels"],
            "predicted_label": results["predictions"],
            "correct": [t == p for t, p in zip(results["true_labels"], results["predictions"])],
        }
        # add probability columns
        for c in range(num_classes):
            df_data[f"prob_class{c}"] = [float(p[c]) for p in probs_arr]

        df = pd.DataFrame(df_data)
        df.to_csv(csv_path, index=False)

        print(f"Predictions saved to: {csv_path}")
        return csv_path

    def _save_confusion_matrix(self, results: Dict[str, Any], output_dir: str) -> str:
        """
        Save confusion matrix as PNG using seaborn.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix

        self._ensure_dir(output_dir)

        y_true = results["true_labels"]
        y_pred = results["predictions"]

        if len(y_true) == 0:
            print("[WARN] No samples; skipping confusion matrix.")
            return ""

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.title("Confusion Matrix - Test Set")

        cm_path = os.path.join(output_dir, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Confusion matrix saved to: {cm_path}")
        return cm_path


def count_patches_by_class(case_dict: Dict, label_map: Dict, split_name: str):
    """
    Count patches by class for analysis (based on patch path lists in case_dict).
    """
    from collections import defaultdict

    class_patch_counts = defaultdict(int)

    for case_id, stains in case_dict.items():
        if case_id not in label_map:
            continue

        label = label_map[case_id]
        total_patches = 0

        for stain_data in stains.values():
            for slice_patches in stain_data:
                total_patches += len(slice_patches)

        class_patch_counts[label] += total_patches

    print(f"\nPatch count by class for {split_name}:")
    print(f"  Benign (0):     {class_patch_counts[0]} patches")
    print(f"  High-grade (1): {class_patch_counts[1]} patches")

    return class_patch_counts
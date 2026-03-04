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
from typing import Dict, Any, Tuple, Optional, List #added list here

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import TRAINING_CONFIG, DEVICE

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


        # -------------------------------------------------------
        # ROMA - Mandatory changes addition
        # Entropy regularization lambda (set to 0.0 to disable)
        self.entropy_lambda = TRAINING_CONFIG.get("entropy_lambda", 0.0) 
        # -------------------------------------------------------

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

    # -------------------------------------------------------
    # ROMA - Mandatory changes replacing forward_one_case
    def _forward_one_case(self, case_data: Dict[str, Any], return_attn_weights: bool = False):
        """
        Returns (logits_with_batch, label_with_batch) or
                (logits_with_batch, label_with_batch, attn_info) if return_attn_weights=True.
        logits_with_batch: shape (1, num_classes)
        label_with_batch:  shape (1,)
        """
        stain_slices = case_data["stain_slices"]
        label = case_data["label"].to(self.device)

        if return_attn_weights:
            logits, attn_info = self.model(stain_slices, return_attn_weights=True)
        else:
            logits = self.model(stain_slices)
            attn_info = None

        if logits.dim() != 1:
            raise ValueError(f"Expected model to return (num_classes,), got {tuple(logits.shape)}")

        logits = logits.unsqueeze(0)  # (1, num_classes)

        if label.dim() == 0:
            label = label.unsqueeze(0)  # (1,)

        if return_attn_weights:
            return logits, label, attn_info
        return logits, label
    # end of edited code
    # -------------------------------------------------------

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
        Train for one epoch.
        Returns average training loss.
        """
        self.model.train()

        if len(train_loader) == 0:
            print("[WARN] train_loader is empty. Returning loss=0.0")
            self.train_losses.append(0.0)
            return 0.0
    # -------------------------------------------------------
    # ROMA - Mandatory changes, edited code
        running_loss = 0.0
        num_batches = 0
        use_entropy = self.entropy_lambda > 0.0

        for batch in tqdm(train_loader, desc="Training", leave=False):
            if not batch:
                continue
            case_data = batch[0]

            if use_entropy:
                logits, label, attn_info = self._forward_one_case(case_data, return_attn_weights=True)
            else:
                logits, label = self._forward_one_case(case_data)
                attn_info = None

            ce_loss = self.criterion(logits, label)

            # ---------------------------
            # Entropy regularization
            # ---------------------------
            entropy_loss = 0.0
            if use_entropy and attn_info is not None:
                # Case-level entropy (across stains)
                if "case_entropy" in attn_info:
                    entropy_loss += attn_info["case_entropy"].mean()
                # Stain & patch level entropy
                for stain_name in attn_info.get("stain_weights", {}):
                    info = attn_info["stain_weights"][stain_name]
                    if "slice_entropy" in info:
                        entropy_loss += info["slice_entropy"].mean()
                    for patch_entropy in info.get("patch_entropies", []):
                        entropy_loss += patch_entropy.mean()

            loss = ce_loss + self.entropy_lambda * entropy_loss

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            running_loss += float(loss.item())
            num_batches += 1

        avg_loss = running_loss / max(num_batches, 1)
        self.train_losses.append(avg_loss)
        return avg_loss
    # end of edited code
    # -------------------------------------------------------

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

                logits, label = self._forward_one_case(case_data)
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

        # commented out these lines from Roma's mandatory changes code
        #if is_best:
        #    best_path = os.path.join(ckpt_dir, "best.pth")
        #    torch.save(checkpoint, best_path)
        #    print(f"Best checkpoint updated: {best_path}")

        return filename

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load model checkpoint.
        Returns: epoch number loaded (int).
        """
        # -------------------------------------------------------
        # ROMA - Mandatory changes, edited code
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        # end of edited code
        # -------------------------------------------------------

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

        start_epoch = int(checkpoint.get("epoch", 0))
        print(f"Resumed from checkpoint: {checkpoint_path} (epoch {start_epoch})") #edited from Roma's mandatory changes code
        return start_epoch

    # ----------------------------
    # Full training loop
    # ----------------------------
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int, # edit from roma's code
        start_epoch: int = 0,
        save_every: int = 5,  # edit from roma's code
        arch: str = "HierarchicalAttnMIL",
    ):
        # ROMA addition from mandatory change's code 
        print(f"Starting training from epoch {start_epoch + 1} to {epochs}")
        print(f"Entropy regularization lambda: {self.entropy_lambda}")
        # -------------------------------------------------------
        """
        Full training loop with scheduler + early stopping.
        """
# commented out from Roma's mandatory changes code
 #       if epochs is None:
 #           epochs = TRAINING_CONFIG["epochs"]

 #       print(f"Starting training from epoch {start_epoch + 1} to {epochs}")
 #       print(f"Device: {self.device}")
 #       print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
 #       if self.scheduler is not None:
 #           print(f"LR scheduler: {type(self.scheduler).__name__}")
 #       if self.use_early_stopping:
 #           print(
 #               "Early stopping enabled "
 #               f"(patience={self.early_stopping_patience}, "
 #               f"min_delta={self.early_stopping_min_delta}, "
 #               f"min_epochs={self.early_stopping_min_epochs})"
 #           )

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

                logits, label = self._forward_one_case(case_data)
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

        results = {
            "test_loss": avg_loss,
            "test_accuracy": acc,
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
    
    # -------------------------------------------------------
    # PAIGE - optional task 3 added code
    def evaluate_patches(
        self,
        test_case_dict: Dict,
        label_map: Dict,
        embeddings_dir: str,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Patch-level inference after training. 
        For each test case, loads each patch .pt file, runs it through patch_projector and classifier, 
        records predictions per patch.
        """
        self.model.eval()

        all_case_ids = []
        all_patch_paths = []
        all_probs = []
        all_preds = []
        all_labels = []

        with torch.inference_mode():
            for case_id, stain_map in tqdm(test_case_dict.items(), desc="Patch-level inference"):
                if case_id not in label_map:
                    continue
                true_label = label_map[case_id]

                # Collect all patch paths for this case across all stains and slices, processing each patch individually
                for stain_patches in stain_map.values():
                    for slice_patches in stain_patches:
                        for patch_path in slice_patches:

                            # Load the raw 4096-dim embedding 
                            fname = os.path.basename(patch_path)
                            feat_path = os.path.join(embeddings_dir, os.path.splitext(fname)[0] + ".pt")

                            try:
                                embedding = torch.load(feat_path, map_location="cpu") 
                                if embedding.dim() != 1:
                                    embedding = embedding.view(-1)
                                embedding = embedding.to(torch.float32).to(self.device)
                            except Exception as e:
                                print(f"[WARN] Could not load {feat_path}: {e}")
                                continue

                            # Run through patch_projector -> classifier -> softmax directly for this individual patch
                            # skip attention pooling 
                            probs = self.model.predict_patches(embedding.unsqueeze(0))  
                            pred = int((probs[0, 1] >= 0.5).item()) #using 0.5 threshold for prediction 

                            all_case_ids.append(case_id)
                            all_patch_paths.append(patch_path)
                            all_probs.append(probs.cpu().numpy()[0])  # probability [benign, high-grade]
                            all_preds.append(pred)
                            all_labels.append(true_label)
        results = {
            "case_ids":    all_case_ids,
            "patch_paths": all_patch_paths,
            "probs":       all_probs,
            "predictions": all_preds,
            "true_labels": all_labels,
        }
        if output_dir is not None:
            self._save_patch_predictions_csv(results, output_dir)

        return results

    def _save_patch_predictions_csv(self, results: Dict[str, Any], output_dir: str,) -> str:
        self._ensure_dir(output_dir)
        # Saves one row per patch with case_id attached so results can be
        # grouped back to the case level in summarize_patch_predictions_by_case
        df = pd.DataFrame({
            "case_id":        results["case_ids"],
            "patch_path":     results["patch_paths"],
            "true_label":     results["true_labels"],
            "predicted":      results["predictions"],
            "prob_benign":    [float(p[0]) for p in results["probs"]],
            "prob_highgrade": [float(p[1]) for p in results["probs"]],
        })
        csv_path = os.path.join(output_dir, "patch_predictions.csv")
        df.to_csv(csv_path, index=False)
        print(f"Patch predictions saved to: {csv_path}")
        return csv_path
    
    def summarize_patch_predictions_by_case(self, patch_results: Dict[str, Any], output_dir: Optional[str] = None,
                                            case_predictions: Optional[Dict] = None, ) -> Dict[str, Any]:
        """
        Groups patch predictions by case and computes percentage of patches predicted as benign vs high-grade for each case
        """
        # grouping patch predictions by case_id
        case_patch_preds = defaultdict(list)
        case_true_labels = {}

        for case_id, pred, true_label in zip(
            patch_results["case_ids"],
            patch_results["predictions"],
            patch_results["true_labels"],
        ):
            case_patch_preds[case_id].append(pred)
            case_true_labels[case_id] = true_label

        # computing percentages per case
        rows = []
        for case_id, preds in case_patch_preds.items():
            total = len(preds)
            # Predictions are 0 or 1 so sum gives count of high-grade patches
            pct_highgrade = 100.0 * sum(preds) / total
            pct_benign = 100.0 - pct_highgrade

            rows.append({
                "case_id":        case_id,
                "true_label":     case_true_labels[case_id],
                # case_predictions comes from test_results returned by trainer.evaluate()
                # joining here so each row shows both case-level and patch-level results
                "predicted_label": case_predictions.get(case_id) if case_predictions else None,
                "total_patches":  total,
                "pct_benign":     round(pct_benign, 2),
                "pct_highgrade":  round(pct_highgrade, 2),
            })

        df = pd.DataFrame(rows)

        print("\nPatch prediction summary by case:")
        print(df.to_string(index=False))

        if output_dir is not None:
            self._ensure_dir(output_dir)
            csv_path = os.path.join(output_dir, "patch_summary_by_case.csv")
            df.to_csv(csv_path, index=False)
            print(f"Patch summary saved to: {csv_path}")

        return df.to_dict(orient="records")
    
    def identify_misclassified_benign_cases(self, patch_summary: List[Dict[str, Any]], 
                                            output_dir: Optional[str] = None,) -> List[Dict[str, Any]]:
        """
        Filters the patch summary to benign cases that were misclassified as high-grade.
        """
        # Filter to false positives: benign cases that were predicted as high-grade
        misclassified = [
            r for r in patch_summary
            if r["true_label"] == 0 and r["predicted_label"] == 1
        ]

        df = pd.DataFrame(misclassified)

        print("\nMisclassified benign cases (true=benign, predicted=high-grade):")
        print(df.to_string(index=False))

        if output_dir is not None:
            self._ensure_dir(output_dir)
            csv_path = os.path.join(output_dir, "misclassified_benign_cases.csv")
            df.to_csv(csv_path, index=False)
            print(f"Misclassified benign cases saved to: {csv_path}")

        return misclassified
    # End of added code
    # -------------------------------------------------------

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
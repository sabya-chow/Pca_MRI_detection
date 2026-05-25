"""Comprehensive evaluation: AUC-ROC, PR-AUC, F1, MCC, QWK, ECE, calibration plots."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
    confusion_matrix, f1_score,
    balanced_accuracy_score, matthews_corrcoef,
    cohen_kappa_score,
)
from src.model import coral_predict


# ── Calibration helper ────────────────────────────────────────────────────────

def _ece(probs: np.ndarray, binary_true: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    n    = len(probs)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        acc  = binary_true[mask].mean()
        conf = probs[mask].mean()
        ece += mask.sum() / n * abs(acc - conf)
    return float(ece)


# ── Core evaluation ───────────────────────────────────────────────────────────

def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = 5,
    cancer_threshold: int = 2,
) -> dict:
    model.eval()
    all_coral, all_preds, all_labels = [], [], []

    with torch.no_grad():
        for x, y in loader:
            out  = model(x.to(device))
            pred = coral_predict(out.cpu())
            all_coral.append(out.cpu().numpy())
            all_preds.append(pred.numpy())
            all_labels.append(y.numpy())

    coral_probs = np.concatenate(all_coral, axis=0)   # (N, K-1)
    preds       = np.concatenate(all_preds)
    labels      = np.concatenate(all_labels)

    prob_cancer = coral_probs[:, cancer_threshold - 1]
    binary_true = (labels >= cancer_threshold).astype(int)
    binary_pred = (preds  >= cancer_threshold).astype(int)

    cm = confusion_matrix(binary_true, binary_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    ppv         = tp / max(tp + fp, 1)
    npv         = tn / max(tn + fn, 1)

    try:
        auc = roc_auc_score(binary_true, prob_cancer)
    except Exception:
        auc = float("nan")

    try:
        ap = average_precision_score(binary_true, prob_cancer)
    except Exception:
        ap = float("nan")

    f1_bin  = f1_score(binary_true, binary_pred, zero_division=0)
    f1_mac  = f1_score(labels, preds, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(binary_true, binary_pred)
    mcc     = matthews_corrcoef(binary_true, binary_pred)

    try:
        qwk = cohen_kappa_score(labels, preds, weights="quadratic")
    except Exception:
        qwk = float("nan")

    mae = float(np.abs(preds - labels).mean())
    acc = float((preds == labels).mean())
    ece = _ece(prob_cancer, binary_true)

    per_class_auc = {}
    for k in range(num_classes):
        col = min(k, coral_probs.shape[1] - 1)
        try:
            per_class_auc[k] = roc_auc_score((labels == k).astype(int), coral_probs[:, col])
        except Exception:
            per_class_auc[k] = float("nan")

    return {
        "auc": auc, "ap": ap,
        "sensitivity": sensitivity, "specificity": specificity,
        "ppv": ppv, "npv": npv,
        "f1_binary": f1_bin, "f1_macro": f1_mac,
        "balanced_accuracy": bal_acc, "mcc": mcc, "qwk": qwk,
        "ordinal_mae": mae, "accuracy": acc, "ece": ece,
        "per_class_auc": per_class_auc,
        "probs": prob_cancer, "coral_probs": coral_probs,
        "preds": preds, "labels": labels,
        "binary_true": binary_true, "binary_pred": binary_pred,
    }


def print_metrics(res: dict, title: str = "Evaluation") -> None:
    print(f"\n{'='*62}")
    print(f"  {title}")
    print(f"{'='*62}")
    print(f"  AUC-ROC          : {res['auc']:.4f}   <- primary metric")
    print(f"  PR-AUC           : {res['ap']:.4f}   <- important w/ class imbalance")
    print(f"  Sensitivity      : {res['sensitivity']:.4f}   (recall, true-positive rate)")
    print(f"  Specificity      : {res['specificity']:.4f}   (true-negative rate)")
    print(f"  PPV (Precision)  : {res['ppv']:.4f}")
    print(f"  NPV              : {res['npv']:.4f}")
    print(f"  F1 Binary        : {res['f1_binary']:.4f}")
    print(f"  F1 Macro         : {res['f1_macro']:.4f}")
    print(f"  Balanced Acc     : {res['balanced_accuracy']:.4f}")
    print(f"  MCC              : {res['mcc']:.4f}   ([-1,1], robust to imbalance)")
    print(f"  QWK              : {res['qwk']:.4f}   (ordinal agreement, 0=random)")
    print(f"  Ordinal MAE      : {res['ordinal_mae']:.4f}  (PIRADS steps avg error)")
    print(f"  Exact Accuracy   : {res['accuracy']:.4f}")
    print(f"  ECE              : {res['ece']:.4f}   (calibration error, lower=better)")
    pac = res.get("per_class_auc", {})
    print(f"  Per-class AUC    : " + "  ".join(f"P{k+1}={pac.get(k,float('nan')):.3f}" for k in range(5)))
    print(f"  n samples        : {len(res['labels'])}")
    print(f"{'='*62}")


# ── SimCLR pre-training curves ────────────────────────────────────────────────

def plot_simclr_curves(history: dict) -> None:
    epochs = range(1, len(history.get("loss", [])) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("SimCLR Pre-training Diagnostics", fontsize=13, fontweight="bold")

    axes[0].plot(epochs, history["loss"], "b-o", ms=3, lw=1.5)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("NT-Xent Loss")
    axes[0].set_title("Contrastive Loss  (lower = richer representations)")
    axes[0].grid(alpha=0.3)
    if len(history["loss"]) > 5:
        smooth = np.convolve(history["loss"], np.ones(5)/5, mode="valid")
        axes[0].plot(range(3, 3+len(smooth)), smooth, "r-", lw=2, label="5-ep MA")
        axes[0].legend()

    if "lr" in history:
        axes[1].plot(epochs, history["lr"], "g-", lw=2)
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Learning Rate")
        axes[1].set_title("Cosine LR Annealing")
        axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
    print(f"  Final NT-Xent loss : {history['loss'][-1]:.4f}")


# ── Supervised training curves ────────────────────────────────────────────────

def plot_training_curves(history: dict, title: str = "Training Diagnostics",
                         phase_a_len: int = 0) -> None:
    epochs = range(1, len(history.get("train_loss", [])) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    vline_kw = dict(color="grey", linestyle="--", lw=1.5, alpha=0.7)

    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss"], "b-", lw=1.5, label="Train")
    ax.plot(epochs, history["val_loss"],   "r-", lw=1.5, label="Val")
    best_ep = int(np.argmin(history["val_loss"])) + 1
    best_v  = min(history["val_loss"])
    ax.axvline(best_ep, color="green", linestyle=":", lw=1.5, label=f"Best val ep={best_ep}")
    if phase_a_len > 0:
        ax.axvline(phase_a_len, **vline_kw, label="Unfreeze encoder")
    ax.annotate(f"{best_v:.3f}", xy=(best_ep, best_v), xytext=(best_ep+0.5, best_v*1.05),
                fontsize=8, color="green")
    ax.set_xlabel("Epoch"); ax.set_ylabel("CORAL Loss")
    ax.set_title("Loss  — train vs val", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Accuracy
    ax = axes[0, 1]
    if "train_acc" in history:
        ax.plot(epochs, history["train_acc"], "b-", lw=1.5, label="Train Acc")
    if "val_acc" in history:
        ax.plot(epochs, history["val_acc"],   "r-", lw=1.5, label="Val Acc")
    if phase_a_len > 0:
        ax.axvline(phase_a_len, **vline_kw)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (exact match)")
    ax.set_title("Accuracy  — train vs val", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # MAE
    ax = axes[1, 0]
    if "train_mae" in history:
        ax.plot(epochs, history["train_mae"], "b-", lw=1.5, label="Train MAE")
    if "val_mae" in history:
        ax.plot(epochs, history["val_mae"],   "r-", lw=1.5, label="Val MAE")
    if phase_a_len > 0:
        ax.axvline(phase_a_len, **vline_kw)
    ax.set_xlabel("Epoch"); ax.set_ylabel("MAE (PIRADS steps)")
    ax.set_title("Ordinal MAE  — train vs val", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # LR + grad norm
    ax = axes[1, 1]
    if "lr" in history and len(history["lr"]) > 0:
        ax.plot(epochs, history["lr"], "g-", lw=2, label="Learning Rate")
        ax.set_ylabel("Learning Rate"); ax.set_xlabel("Epoch")
        ax.set_title("LR Schedule & Gradient Norm", fontweight="bold")
        ax.legend(loc="upper left", fontsize=8)
        if "grad_norm" in history and any(g > 0 for g in history["grad_norm"]):
            ax2 = ax.twinx()
            ax2.plot(epochs, history["grad_norm"], "m--", alpha=0.7, lw=1.5, label="Grad Norm")
            ax2.set_ylabel("Gradient L2 Norm", color="m")
            ax2.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Overfit gap
    if "train_loss" in history and "val_loss" in history:
        gap = [v - t for t, v in zip(history["train_loss"], history["val_loss"])]
        print(f"  Overfit gap (val-train loss): max={max(gap):.4f}  final={gap[-1]:.4f}")
        if gap[-1] > 0.1:
            print("  WARNING: Significant overfitting — consider more dropout or weight decay.")


# ── Full 6-panel evaluation plot ──────────────────────────────────────────────

def plot_results(res: dict, title: str = "Model Evaluation") -> None:
    from sklearn.metrics import roc_curve, precision_recall_curve
    fig = plt.figure(figsize=(18, 15))
    gs  = gridspec.GridSpec(3, 2, hspace=0.42, wspace=0.35)

    # 1. ROC curve
    ax1 = fig.add_subplot(gs[0, 0])
    try:
        fpr, tpr, _ = roc_curve(res["binary_true"], res["probs"])
        ax1.plot(fpr, tpr, "b-", lw=2.5, label=f"AUC={res['auc']:.3f}")
        ax1.fill_between(fpr, tpr, alpha=0.12, color="b")
        ax1.plot([0,1],[0,1],"k--",lw=1,label="Random (AUC=0.50)")
        idx = np.argmin(np.abs(tpr - 0.90))
        ax1.scatter(fpr[idx], tpr[idx], color="red", s=60, zorder=5,
                    label=f"Sens=0.90 -> Spec={1-fpr[idx]:.2f}")
        ax1.set_xlabel("False Positive Rate (1-Specificity)")
        ax1.set_ylabel("True Positive Rate (Sensitivity)")
        ax1.set_title("ROC Curve  —  Cancer vs Non-cancer (PIRADS>=3)", fontweight="bold")
        ax1.legend(fontsize=8); ax1.set_xlim(0,1); ax1.set_ylim(0,1.02)
    except Exception as e:
        ax1.text(0.5,0.5,str(e),ha="center",va="center",transform=ax1.transAxes)

    # 2. Precision-Recall curve
    ax2 = fig.add_subplot(gs[0, 1])
    try:
        prec, rec, _ = precision_recall_curve(res["binary_true"], res["probs"])
        pos_rate = res["binary_true"].mean()
        ax2.plot(rec, prec, "g-", lw=2.5, label=f"PR-AUC={res['ap']:.3f}")
        ax2.fill_between(rec, prec, alpha=0.12, color="g")
        ax2.axhline(pos_rate, color="k", linestyle="--", lw=1,
                    label=f"Random baseline (prev={pos_rate:.2f})")
        ax2.set_xlabel("Recall (Sensitivity)")
        ax2.set_ylabel("Precision (PPV)")
        ax2.set_title("Precision-Recall Curve", fontweight="bold")
        ax2.legend(fontsize=8); ax2.set_xlim(0,1); ax2.set_ylim(0,1.05)
    except Exception as e:
        ax2.text(0.5,0.5,str(e),ha="center",va="center",transform=ax2.transAxes)

    # 3. Ordinal confusion matrix (row-normalised)
    ax3 = fig.add_subplot(gs[1, 0])
    lr = list(range(5))
    cm_ord  = confusion_matrix(res["labels"], res["preds"], labels=lr)
    cm_norm = cm_ord.astype(float) / (cm_ord.sum(axis=1, keepdims=True) + 1e-8)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", ax=ax3,
                xticklabels=[f"P{k+1}" for k in lr],
                yticklabels=[f"P{k+1}" for k in lr],
                vmin=0, vmax=1, cbar_kws={"label": "Row fraction"})
    ax3.set_xlabel("Predicted PIRADS"); ax3.set_ylabel("True PIRADS")
    ax3.set_title("Ordinal Confusion Matrix  (row-normalised)", fontweight="bold")

    # 4. Calibration / reliability diagram
    ax4 = fig.add_subplot(gs[1, 1])
    try:
        n_bins = 8
        bins   = np.linspace(0, 1, n_bins + 1)
        bin_c, bin_a = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (res["probs"] >= lo) & (res["probs"] < hi)
            if mask.sum() > 0:
                bin_c.append(res["probs"][mask].mean())
                bin_a.append(res["binary_true"][mask].mean())
        bin_c, bin_a = np.array(bin_c), np.array(bin_a)
        gap = bin_a - bin_c
        colors_cal = ["#e74c3c" if g > 0 else "#2980b9" for g in gap]
        ax4.bar(bin_c, np.abs(gap), bottom=np.minimum(bin_c, bin_a),
                width=0.09, alpha=0.35, color=colors_cal, label="Gap")
        ax4.plot(bin_c, bin_a, "r-o", lw=2, ms=5, label=f"Model  ECE={res['ece']:.3f}")
        ax4.plot([0,1],[0,1],"k--",lw=1,label="Perfect calibration")
        ax4.set_xlabel("Mean Predicted Probability"); ax4.set_ylabel("Fraction Positive")
        ax4.set_title("Calibration / Reliability Diagram", fontweight="bold")
        ax4.legend(fontsize=8); ax4.set_xlim(-0.05,1.05); ax4.set_ylim(-0.05,1.1)
    except Exception as e:
        ax4.text(0.5,0.5,str(e),ha="center",va="center",transform=ax4.transAxes)

    # 5. Cancer score distribution by PIRADS
    ax5 = fig.add_subplot(gs[2, 0])
    colors5 = ["#27ae60","#8BC34A","#f39c12","#e67e22","#c0392b"]
    for k in range(5):
        mask = res["labels"] == k
        if mask.any():
            ax5.hist(res["probs"][mask], bins=16, alpha=0.55, color=colors5[k],
                     label=f"PIRADS {k+1}  n={mask.sum()}", density=True)
    ax5.axvline(0.5, color="k", linestyle="--", lw=1.5, label="Default thr=0.5")
    ax5.set_xlabel("P(cancer >= PIRADS-3)"); ax5.set_ylabel("Density")
    ax5.set_title("Predicted Cancer Score — by True PIRADS", fontweight="bold")
    ax5.legend(fontsize=8)

    # 6. Summary metrics horizontal bar
    ax6 = fig.add_subplot(gs[2, 1])
    mcc_norm = (res["mcc"] + 1) / 2
    metric_names  = ["AUC-ROC", "PR-AUC", "Sensitivity", "Specificity",
                     "PPV", "NPV", "F1-Binary", "Bal. Acc", "MCC (norm)"]
    metric_values = [res["auc"], res["ap"], res["sensitivity"], res["specificity"],
                     res["ppv"], res["npv"], res["f1_binary"],
                     res["balanced_accuracy"], mcc_norm]
    palette = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(metric_names)))
    bars = ax6.barh(metric_names, metric_values, color=palette, edgecolor="white", height=0.6)
    for bar, val in zip(bars, metric_values):
        ax6.text(min(val + 0.01, 1.0), bar.get_y() + bar.get_height()/2,
                 f"{val:.3f}", va="center", fontsize=9, fontweight="bold")
    ax6.axvline(0.5, color="red", linestyle="--", lw=1, label="Random baseline")
    ax6.set_xlim(0, 1.18); ax6.set_xlabel("Score")
    ax6.set_title("Summary Metrics Dashboard", fontweight="bold")
    ax6.legend(fontsize=8)

    fig.suptitle(title, fontsize=15, fontweight="bold")
    plt.show()


# ── Threshold analysis ────────────────────────────────────────────────────────

def plot_threshold_analysis(res: dict, title: str = "Threshold Sensitivity") -> None:
    thresholds = np.linspace(0.05, 0.95, 80)
    sens_list, spec_list, f1_list, ppv_list = [], [], [], []
    for t in thresholds:
        bp = (res["probs"] >= t).astype(int)
        cm = confusion_matrix(res["binary_true"], bp, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
        sens_list.append(tp / max(tp+fn, 1))
        spec_list.append(tn / max(tn+fp, 1))
        f1_list.append(f1_score(res["binary_true"], bp, zero_division=0))
        ppv_list.append(tp / max(tp+fp, 1))

    J_idx   = int(np.argmax([s+sp-1 for s,sp in zip(sens_list, spec_list)]))
    f1_idx  = int(np.argmax(f1_list))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    axes[0].plot(thresholds, sens_list, "b-",  lw=2, label="Sensitivity")
    axes[0].plot(thresholds, spec_list, "r-",  lw=2, label="Specificity")
    axes[0].plot(thresholds, ppv_list,  "g-",  lw=2, label="PPV (Precision)")
    axes[0].axvline(0.5, color="k", linestyle="--", lw=1, label="Default thr=0.5")
    axes[0].axvline(thresholds[J_idx], color="purple", linestyle=":", lw=2,
                    label=f"Youden J @ {thresholds[J_idx]:.2f}")
    axes[0].set_xlabel("Decision Threshold"); axes[0].set_ylabel("Score")
    axes[0].set_title("Sens / Spec / PPV vs Threshold", fontweight="bold")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

    axes[1].plot(thresholds, f1_list, color="darkorange", lw=2.5)
    axes[1].axvline(thresholds[f1_idx], color="r", linestyle="--",
                    label=f"Best F1={max(f1_list):.3f} @ thr={thresholds[f1_idx]:.2f}")
    axes[1].set_xlabel("Decision Threshold"); axes[1].set_ylabel("F1 Score")
    axes[1].set_title("F1 vs Threshold", fontweight="bold")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"  Youden-optimal threshold: {thresholds[J_idx]:.3f}  "
          f"(Sens={sens_list[J_idx]:.3f}, Spec={spec_list[J_idx]:.3f})")
    print(f"  F1-optimal threshold    : {thresholds[f1_idx]:.3f}  "
          f"(F1={max(f1_list):.3f})")


# ── Per-class ROC (one-vs-rest) ───────────────────────────────────────────────

def plot_per_class_roc(res: dict) -> None:
    colors5 = ["#27ae60","#8BC34A","#f39c12","#e67e22","#c0392b"]
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
    fig.suptitle("Per-class One-vs-Rest ROC  (PIRADS 1-5)", fontsize=13, fontweight="bold")

    for k in range(5):
        ax = axes[k]
        binary = (res["labels"] == k).astype(int)
        n_pos  = binary.sum()
        col    = min(k, res["coral_probs"].shape[1] - 1)
        score  = res["coral_probs"][:, col]
        try:
            fpr, tpr, _ = roc_curve(binary, score)
            auc = roc_auc_score(binary, score)
            ax.plot(fpr, tpr, color=colors5[k], lw=2.5, label=f"AUC={auc:.3f}")
            ax.fill_between(fpr, tpr, alpha=0.15, color=colors5[k])
        except Exception:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
        ax.plot([0,1],[0,1],"k--",lw=1)
        ax.set_title(f"PIRADS {k+1}\n(n={n_pos})", fontweight="bold", color=colors5[k])
        ax.set_xlabel("FPR")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("TPR")
    plt.tight_layout()
    plt.show()


# ── Ablation comparison ───────────────────────────────────────────────────────

def compare_ablations(results_dict: dict) -> None:
    names = list(results_dict.keys())
    def g(n, k): return results_dict[n].get(k, float("nan"))

    metrics_cl = {
        "AUC":      [g(n,"auc")               for n in names],
        "PR-AUC":   [g(n,"ap")                for n in names],
        "Sens.":    [g(n,"sensitivity")       for n in names],
        "Spec.":    [g(n,"specificity")       for n in names],
        "F1":       [g(n,"f1_binary")         for n in names],
        "Bal.Acc":  [g(n,"balanced_accuracy") for n in names],
    }
    colors_cls = ["#2196F3","#00BCD4","#4CAF50","#FF9800","#E91E63","#9C27B0"]
    x = np.arange(len(names))
    w = 0.13

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Ablation Study", fontsize=14, fontweight="bold")

    for i, (met, vals) in enumerate(metrics_cl.items()):
        offset = (i - len(metrics_cl)/2 + 0.5) * w
        axes[0].bar(x + offset, vals, w, label=met, color=colors_cls[i], alpha=0.85)
    axes[0].set_xticks(x); axes[0].set_xticklabels(names, rotation=12, ha="right")
    axes[0].set_ylim(0, 1.2); axes[0].axhline(0.5, color="red", linestyle="--", lw=1)
    axes[0].set_title("Classification Metrics", fontweight="bold"); axes[0].legend(fontsize=8)

    maes = [g(n,"ordinal_mae") for n in names]
    qwks = [g(n,"qwk")        for n in names]
    bw   = 0.35
    axes[1].bar(x - bw/2, maes, bw, label="Ordinal MAE (lower=better)", color="#9C27B0", alpha=0.85)
    ax2 = axes[1].twinx()
    ax2.bar(x + bw/2, qwks, bw, label="QWK (higher=better)", color="#FF5722", alpha=0.85)
    axes[1].set_xticks(x); axes[1].set_xticklabels(names, rotation=12, ha="right")
    axes[1].set_ylabel("MAE (PIRADS steps)"); ax2.set_ylabel("QWK")
    axes[1].set_title("Ordinal Metrics: MAE & QWK", fontweight="bold")
    axes[1].legend(loc="upper left", fontsize=8); ax2.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.show()

    # Print table
    print(f"\n{'Model':<26} {'AUC':>6} {'PR-AUC':>7} {'Sens':>6} {'Spec':>6} "
          f"{'F1':>6} {'Bal.Acc':>8} {'MAE':>6} {'QWK':>6}")
    print("-" * 80)
    for n in names:
        r = results_dict[n]
        print(f"  {n:<24} {g(n,'auc'):>6.3f} {g(n,'ap'):>7.3f} "
              f"{g(n,'sensitivity'):>6.3f} {g(n,'specificity'):>6.3f} "
              f"{g(n,'f1_binary'):>6.3f} {g(n,'balanced_accuracy'):>8.3f} "
              f"{g(n,'ordinal_mae'):>6.3f} {g(n,'qwk'):>6.3f}")

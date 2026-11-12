"""
Visualization Module for Assamese Word Sense Disambiguation.
Generates publication-quality charts and graphs for:
1. Prompting Technique Accuracy Comparison
2. Precision, Recall, and F1-Score Comparison
3. Confusion Matrix
4. Target Word Frequency Distribution
5. Sense Distribution (2-sense vs 3+-sense)
6. Ambiguous Word Detection Performance
7. Word-wise / Complexity Performance Breakdown
All graphs are saved to results/graphs/ and can be embedded into Tkinter.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import matplotlib
# Use Agg backend for non-GUI headless plotting/saving; TkAgg used in GUI
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from config import GRAPHS_DIR, THEME, ORIGINAL_DATASET_PATH

# Set academic plotting style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Nirmala UI", "Vrinda", "Segoe UI", "Arial", "DejaVu Sans", "sans-serif"]
plt.rcParams["axes.edgecolor"] = "#CBD5E1"
plt.rcParams["axes.linewidth"] = 0.8


class WSDVisualizer:
    """
    Produces publication-grade figures for the Assamese WSD project.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or GRAPHS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.colors = [THEME["primary"], THEME["secondary"], THEME["accent"], THEME["warning"]]

    def plot_accuracy_comparison(
        self,
        metrics: Dict[str, Dict[str, float]],
        save_name: str = "1_technique_accuracy_comparison.png",
    ) -> Path:
        """1. Prompting technique accuracy comparison (Conditional vs End-to-End)."""
        techniques = list(metrics.keys())
        short_names = [t.replace(" Prompting", "") for t in techniques]
        cond_acc = [metrics[t]["conditional_accuracy"] * 100 for t in techniques]
        e2e_acc = [metrics[t]["end_to_end_accuracy"] * 100 for t in techniques]

        x = np.arange(len(techniques))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 5))
        rects1 = ax.bar(x - width/2, cond_acc, width, label="Conditional WSD Acc", color=THEME["primary"])
        rects2 = ax.bar(x + width/2, e2e_acc, width, label="End-to-End Acc", color=THEME["accent"])

        ax.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
        ax.set_title("WSD Accuracy Comparison Across Prompting Techniques", fontsize=13, fontweight="bold", pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, fontsize=10)
        ax.set_ylim(0, 105)
        ax.legend(frameon=True, facecolor="white", edgecolor="#E2E8F0")

        # Add data labels
        for rect in rects1:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
        for rect in rects2:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        out_path = self.output_dir / save_name
        plt.savefig(out_path, dpi=200)
        plt.close()
        return out_path

    def plot_precision_recall_f1(
        self,
        metrics: Dict[str, Dict[str, float]],
        save_name: str = "2_precision_recall_f1_comparison.png",
    ) -> Path:
        """2. Precision, Recall, and F1-score comparison."""
        techniques = list(metrics.keys())
        short_names = [t.replace(" Prompting", "") for t in techniques]
        prec = [metrics[t]["macro_precision"] * 100 for t in techniques]
        rec = [metrics[t]["macro_recall"] * 100 for t in techniques]
        f1 = [metrics[t]["macro_f1"] * 100 for t in techniques]

        x = np.arange(len(techniques))
        width = 0.25

        fig, ax = plt.subplots(figsize=(9, 5))
        r1 = ax.bar(x - width, prec, width, label="Macro Precision", color="#1E3A8A")
        r2 = ax.bar(x, rec, width, label="Macro Recall", color="#0284C7")
        r3 = ax.bar(x + width, f1, width, label="Macro F1-Score", color="#0D9488")

        ax.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
        ax.set_title("Precision, Recall, and F1-Score Across Prompting Methods", fontsize=13, fontweight="bold", pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, fontsize=10)
        ax.set_ylim(0, 105)
        ax.legend(frameon=True, facecolor="white", edgecolor="#E2E8F0")

        plt.tight_layout()
        out_path = self.output_dir / save_name
        plt.savefig(out_path, dpi=200)
        plt.close()
        return out_path

    def plot_confusion_matrix(
        self,
        results_df: pd.DataFrame,
        top_k_classes: int = 8,
        save_name: str = "3_wsd_confusion_matrix.png",
    ) -> Path:
        """3. Confusion Matrix for top frequent sense classes."""
        # Filter to valid detections and top classes
        df = results_df[results_df["Is_Detected"] == True].copy()
        top_classes = df["Ground_Truth_Sense_ID"].value_counts().head(top_k_classes).index.tolist()
        df_sub = df[df["Ground_Truth_Sense_ID"].isin(top_classes) & df["Predicted_Sense_ID"].isin(top_classes)]

        if df_sub.empty:
            top_classes = df["Ground_Truth_Sense_ID"].value_counts().head(4).index.tolist()
            df_sub = df[df["Ground_Truth_Sense_ID"].isin(top_classes)]

        cm = pd.crosstab(
            df_sub["Ground_Truth_Sense_ID"],
            df_sub["Predicted_Sense_ID"],
            rownames=["Ground Truth"],
            colnames=["Predicted"],
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True, ax=ax, linewidths=0.5, linecolor="#E2E8F0")
        ax.set_title(f"Confusion Matrix (Top Senses)", fontsize=13, fontweight="bold", pad=12)

        plt.tight_layout()
        out_path = self.output_dir / save_name
        plt.savefig(out_path, dpi=200)
        plt.close()
        return out_path

    def plot_word_distribution(
        self,
        raw_df: pd.DataFrame,
        save_name: str = "4_target_word_distribution.png",
    ) -> Path:
        """4. Target word frequency distribution across the dataset."""
        counts = raw_df["Target_Word"].value_counts().sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(8, 7))
        # Use simple numeric indices with labels to ensure clean display
        y_pos = np.arange(len(counts))
        bars = ax.barh(y_pos, counts.values, color="#1E3A8A", edgecolor="none", height=0.65)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(counts.index, fontsize=10)
        ax.set_xlabel("Sentence Count", fontsize=11, fontweight="bold")
        ax.set_title("Assamese Target Word Frequency Distribution (N=1,395)", fontsize=13, fontweight="bold", pad=12)

        for bar in bars:
            w = bar.get_width()
            ax.annotate(f"{int(w)}", xy=(w, bar.get_y() + bar.get_height() / 2),
                        xytext=(4, 0), textcoords="offset points", ha="left", va="center", fontsize=8.5)

        plt.tight_layout()
        out_path = self.output_dir / save_name
        plt.savefig(out_path, dpi=200)
        plt.close()
        return out_path

    def plot_sense_distribution(
        self,
        senses_map: Dict[str, Dict[str, Any]],
        save_name: str = "5_sense_complexity_distribution.png",
    ) -> Path:
        """5. Distribution of Ambiguity (2-sense vs 3+-sense words)."""
        two_sense_count = sum(1 for w, senses in senses_map.items() if len(senses) == 2)
        three_sense_count = sum(1 for w, senses in senses_map.items() if len(senses) == 3)
        multi_sense_count = sum(1 for w, senses in senses_map.items() if len(senses) > 3)

        categories = ["2 Senses\n(Binary Ambiguity)", "3 Senses\n(Ternary)", "4+ Senses\n(Polysemy)"]
        values = [two_sense_count, three_sense_count, multi_sense_count]
        colors = ["#1E3A8A", "#0284C7", "#0D9488"]

        fig, ax = plt.subplots(figsize=(7, 4.8))
        bars = ax.bar(categories, values, color=colors, width=0.55)
        ax.set_ylabel("Number of Target Words", fontsize=11, fontweight="bold")
        ax.set_title("Sense Distribution by Lexical Complexity", fontsize=13, fontweight="bold", pad=12)
        ax.set_ylim(0, max(values) + 3)

        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{int(h)} words", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

        plt.tight_layout()
        out_path = self.output_dir / save_name
        plt.savefig(out_path, dpi=200)
        plt.close()
        return out_path

    def plot_detection_performance(
        self,
        detection_metrics: Dict[str, float],
        save_name: str = "6_ambiguous_word_detection_performance.png",
    ) -> Path:
        """6. Ambiguous Word Detection Performance (Stage A)."""
        metrics_names = ["Precision", "Recall", "F1-Score", "Accuracy"]
        vals = [
            detection_metrics.get("detection_precision", 0.0) * 100,
            detection_metrics.get("detection_recall", 0.0) * 100,
            detection_metrics.get("detection_f1", 0.0) * 100,
            detection_metrics.get("detection_accuracy", 0.0) * 100,
        ]

        fig, ax = plt.subplots(figsize=(7, 4.8))
        bars = ax.bar(metrics_names, vals, color=["#1E3A8A", "#0284C7", "#0D9488", "#16A34A"], width=0.5)
        ax.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
        ax.set_title("Stage A: Automatic Ambiguous Word Detection Metrics", fontsize=13, fontweight="bold", pad=12)
        ax.set_ylim(0, 110)

        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

        plt.tight_layout()
        out_path = self.output_dir / save_name
        plt.savefig(out_path, dpi=200)
        plt.close()
        return out_path

    def plot_complexity_breakdown(
        self,
        metrics: Dict[str, Dict[str, float]],
        save_name: str = "7_wordwise_complexity_performance.png",
    ) -> Path:
        """7. 2-sense vs 3+-sense disambiguation accuracy across techniques."""
        techniques = list(metrics.keys())
        short_names = [t.replace(" Prompting", "") for t in techniques]
        two_sense = [metrics[t]["two_sense_accuracy"] * 100 for t in techniques]
        multi_sense = [metrics[t]["multi_sense_accuracy"] * 100 for t in techniques]

        x = np.arange(len(techniques))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 5))
        r1 = ax.bar(x - width/2, two_sense, width, label="2-Sense Ambiguity (Binary)", color="#1E3A8A")
        r2 = ax.bar(x + width/2, multi_sense, width, label="3+-Sense Ambiguity (Polysemous)", color="#D97706")

        ax.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
        ax.set_title("Disambiguation Accuracy by Lexical Complexity", fontsize=13, fontweight="bold", pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, fontsize=10)
        ax.set_ylim(0, 105)
        ax.legend(frameon=True, facecolor="white", edgecolor="#E2E8F0")

        for rect in r1:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
        for rect in r2:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        out_path = self.output_dir / save_name
        plt.savefig(out_path, dpi=200)
        plt.close()
        return out_path

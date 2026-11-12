"""
Batch Query Processor and Evaluation Module for Assamese WSD.
Allows users to upload custom query files (.csv, .txt, .tsv), automatically:
1. Detects ambiguous words in each query.
2. Predicts the correct sense and meaning using chosen prompt engineering technique.
3. Computes vector space Cosine Similarity between query context and candidate senses.
4. If ground-truth Sense IDs are present:
   - Computes Precision, Recall, Macro-F1, and Accuracy.
   - Generates and plots the Confusion Matrix.
5. Exports complete row-wise results to CSV.
"""

import os
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from dataset_handler import DatasetHandler
from ambiguous_word_detector import AssameseAmbiguousWordDetector
from wsd_prediction import WSDPredictor
from config import RESULTS_DIR, GRAPHS_DIR, THEME, PROMPT_ZERO_SHOT


class CosineSimilarityEngine:
    """
    Computes vector space Cosine Similarity between input queries and candidate sense profiles.
    Precomputes TF-IDF representations from verified corpus training sentences.
    """

    def __init__(self, dataset_handler: Optional[DatasetHandler] = None):
        self.handler = dataset_handler or DatasetHandler()
        self._sense_corpus: Dict[str, str] = {}
        self._build_sense_corpus()

    def _build_sense_corpus(self):
        """Aggregates all training sentences for each unique Sense_ID."""
        train_df, _ = self.handler.load_train_test()
        grouped = train_df.groupby("Sense_ID")
        for sid, group in grouped:
            sentences = group["Sentence"].tolist()
            # Clean and combine all sentences for this sense
            cleaned = []
            for s in sentences:
                tokens = [w for w in re.findall(r"[\u0980-\u09FF]+", s) if len(w) > 1]
                cleaned.append(" ".join(tokens))
            self._sense_corpus[sid] = " ".join(cleaned)

    def compute_similarity(
        self,
        query_sentence: str,
        target_word: str,
        candidate_senses: List[Dict[str, Any]],
    ) -> Tuple[float, Dict[str, float]]:
        """
        Computes cosine similarity between the query sentence context and each candidate sense profile.
        Returns:
            - Best candidate similarity (float)
            - Dictionary of {sense_id: similarity_score}
        """
        if not candidate_senses:
            return 0.0, {}

        # Remove the ambiguous target word itself to focus on surrounding contextual evidence
        query_tokens = [w for w in re.findall(r"[\u0980-\u09FF]+", query_sentence) if w != target_word]
        clean_query = " ".join(query_tokens)
        if not clean_query.strip():
            clean_query = query_sentence.strip()

        sense_ids = []
        sense_texts = []
        for cs in candidate_senses:
            sid = cs["sense_id"]
            text = self._sense_corpus.get(sid, cs.get("primary_meaning", ""))
            sense_ids.append(sid)
            sense_texts.append(text)

        try:
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
            tfidf_matrix = vectorizer.fit_transform(sense_texts + [clean_query])

            sense_vectors = tfidf_matrix[:-1]
            query_vector = tfidf_matrix[-1:]

            sims = cosine_similarity(query_vector, sense_vectors)[0]
            scores = {sid: round(float(sim), 4) for sid, sim in zip(sense_ids, sims)}
            best_sim = max(scores.values()) if scores else 0.0
            return best_sim, scores
        except Exception:
            # Fallback if vocabulary is too sparse
            return 0.5, {cs["sense_id"]: 0.5 for cs in candidate_senses}


class BatchQueryProcessor:
    """
    Handles uploading, parsing, batch prediction, cosine similarity computation,
    and ground-truth evaluation for custom query files.
    """

    def __init__(
        self,
        dataset_handler: Optional[DatasetHandler] = None,
        predictor: Optional[WSDPredictor] = None,
    ):
        self.handler = dataset_handler or DatasetHandler()
        self.detector = AssameseAmbiguousWordDetector(self.handler)
        self.predictor = predictor or WSDPredictor(self.handler, self.detector)
        self.similarity_engine = CosineSimilarityEngine(self.handler)

    def load_query_file(self, file_path: str) -> pd.DataFrame:
        """
        Loads and standardizes an uploaded file (.csv, .tsv, or .txt).
        Auto-detects Sentence/Query and Ground Truth columns.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        if ext in (".csv", ".tsv"):
            sep = "\t" if ext == ".tsv" else ","
            df = pd.read_csv(file_path, sep=sep, encoding="utf-8-sig")
        elif ext in (".txt", ""):
            # Text file with one query per line
            with open(file_path, "r", encoding="utf-8-sig", errors="ignore") as f:
                lines = [line.strip() for line in f if line.strip()]
            df = pd.DataFrame({"Sentence": lines})
        else:
            raise ValueError(f"Unsupported file format '{ext}'. Please upload a .csv, .tsv, or .txt file.")

        # Identify sentence column
        sentence_col = None
        for col in df.columns:
            if col.strip().lower() in ("sentence", "query", "text", "input", "input_sentence", "question"):
                sentence_col = col
                break
        if sentence_col is None:
            # Pick first string column
            str_cols = df.select_dtypes(include=["object"]).columns
            if len(str_cols) > 0:
                sentence_col = str_cols[0]
            else:
                raise ValueError("Could not find a valid text/sentence column in the uploaded file.")

        df = df.rename(columns={sentence_col: "Sentence"})
        df["Sentence"] = df["Sentence"].astype(str).apply(lambda s: unicodedata.normalize("NFC", s.strip()))

        # Check for optional ground-truth sense column
        gt_col = None
        for col in df.columns:
            if col.strip().lower() in ("sense_id", "sense", "target_sense", "ground_truth", "label", "expected_sense"):
                gt_col = col
                break
        if gt_col:
            df = df.rename(columns={gt_col: "Ground_Truth_Sense_ID"})
            df["Ground_Truth_Sense_ID"] = df["Ground_Truth_Sense_ID"].astype(str).str.strip().str.upper()
        else:
            df["Ground_Truth_Sense_ID"] = None

        # Check for optional target word column
        tw_col = None
        for col in df.columns:
            if col.strip().lower() in ("target_word", "word", "target", "ambiguous_word"):
                tw_col = col
                break
        if tw_col:
            df = df.rename(columns={tw_col: "Annotated_Target_Word"})
        else:
            df["Annotated_Target_Word"] = None

        if "ID" not in df.columns:
            df["ID"] = [f"Q_{i+1:04d}" for i in range(len(df))]

        return df

    def process_queries(
        self,
        file_path: str,
        technique: str = PROMPT_ZERO_SHOT,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Analyzes all queries in the uploaded file:
        - Automatically detects ambiguous words.
        - Predicts intended Sense ID and meaning.
        - Computes Cosine Similarity against candidate senses.
        - Computes Accuracy, Precision, Recall, F1, and Confusion Matrix if Ground Truth is provided.
        """
        df = self.load_query_file(file_path)
        total_queries = len(df)

        output_rows = []
        has_ground_truth = df["Ground_Truth_Sense_ID"].notnull().any() and (df["Ground_Truth_Sense_ID"] != "NONE").any()

        for idx, row in df.iterrows():
            qid = row.get("ID", f"Q_{idx+1}")
            sent = row["Sentence"]
            gt_sense = row.get("Ground_Truth_Sense_ID")
            annotated_word = row.get("Annotated_Target_Word")

            # Step 1: Automatic Detection
            detected = self.detector.detect_ambiguous_words(sent)

            if not detected:
                output_rows.append({
                    "Query_ID": qid,
                    "Sentence": sent,
                    "Detected_Word": "None",
                    "Predicted_Sense_ID": "NO_DETECTION",
                    "Predicted_Meaning": "No ambiguous word detected from current vocabulary",
                    "Cosine_Similarity": 0.0,
                    "All_Candidate_Similarities": "{}",
                    "Explanation": "No vocabulary matching the 22 registered ambiguous words was detected.",
                    "Ground_Truth_Sense_ID": gt_sense or "N/A",
                    "Is_Correct": False if gt_sense else "N/A",
                    "Confidence": 0.0,
                })
            else:
                for det in detected:
                    tw = det["target_word"]
                    candidate_senses = det["candidate_senses"]

                    # Predict sense using selected prompting technique
                    prompt = self.predictor.prompt_generator.build_prompt(
                        technique=technique,
                        sentence=sent,
                        target_word=tw,
                        candidate_senses=candidate_senses,
                    )
                    pred = self.predictor.llm_service.predict_sense(
                        prompt=prompt,
                        target_word=tw,
                        candidate_senses=candidate_senses,
                        sentence=sent,
                    )

                    pred_sid = pred.get("predicted_sense_id", "UNKNOWN")
                    meaning = pred.get("predicted_meaning", "")
                    explanation = pred.get("explanation", "")

                    # Compute Cosine Similarity
                    best_sim, all_sims = self.similarity_engine.compute_similarity(sent, tw, candidate_senses)
                    pred_sim = all_sims.get(pred_sid, best_sim)

                    # Determine correctness if ground truth is present
                    is_correct = "N/A"
                    if gt_sense and str(gt_sense).upper() not in ("NONE", "NAN", ""):
                        is_correct = (pred_sid == gt_sense.strip().upper())

                    output_rows.append({
                        "Query_ID": qid,
                        "Sentence": sent,
                        "Detected_Word": tw,
                        "Predicted_Sense_ID": pred_sid,
                        "Predicted_Meaning": meaning,
                        "Cosine_Similarity": pred_sim,
                        "All_Candidate_Similarities": str(all_sims),
                        "Explanation": explanation,
                        "Ground_Truth_Sense_ID": gt_sense or "N/A",
                        "Is_Correct": is_correct,
                        "Confidence": pred.get("confidence", 0.85),
                    })

            if progress_callback:
                progress_callback(int(((idx + 1) / total_queries) * 100), idx + 1, total_queries)

        results_df = pd.DataFrame(output_rows)

        # Compute Evaluation Metrics if Ground Truth is provided
        metrics: Dict[str, Any] = {
            "total_queries": total_queries,
            "total_predictions": len(results_df),
            "mean_cosine_similarity": round(float(results_df["Cosine_Similarity"].mean()), 4) if not results_df.empty else 0.0,
            "has_ground_truth": bool(has_ground_truth),
        }

        cm_plot_path = None
        if has_ground_truth:
            eval_subset = results_df[results_df["Ground_Truth_Sense_ID"].isin(self.handler.get_all_sense_ids())].copy()
            if not eval_subset.empty:
                y_true = eval_subset["Ground_Truth_Sense_ID"].tolist()
                y_pred = eval_subset["Predicted_Sense_ID"].tolist()

                acc = accuracy_score(y_true, y_pred)
                prec, rec, f1, _ = precision_recall_fscore_support(
                    y_true, y_pred, average="macro", zero_division=0
                )

                metrics["accuracy"] = round(float(acc), 4)
                metrics["precision"] = round(float(prec), 4)
                metrics["recall"] = round(float(rec), 4)
                metrics["f1_score"] = round(float(f1), 4)
                metrics["evaluated_samples"] = len(eval_subset)

                # Generate and save Confusion Matrix plot
                cm_plot_path = self.plot_confusion_matrix(eval_subset)
                metrics["confusion_matrix_path"] = str(cm_plot_path)

        # Save results to CSV
        output_csv_path = RESULTS_DIR / "batch_query_results.csv"
        results_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
        metrics["results_csv_path"] = str(output_csv_path)

        return {
            "metrics": metrics,
            "results_df": results_df,
            "confusion_matrix_plot": cm_plot_path,
        }

    def plot_confusion_matrix(
        self,
        eval_df: pd.DataFrame,
        top_k: int = 8,
        save_name: str = "batch_confusion_matrix.png",
    ) -> Path:
        """Generates and saves a publication-quality Confusion Matrix heatmap."""
        top_classes = eval_df["Ground_Truth_Sense_ID"].value_counts().head(top_k).index.tolist()
        df_sub = eval_df[eval_df["Ground_Truth_Sense_ID"].isin(top_classes) & eval_df["Predicted_Sense_ID"].isin(top_classes)]

        if df_sub.empty:
            df_sub = eval_df

        cm = pd.crosstab(
            df_sub["Ground_Truth_Sense_ID"],
            df_sub["Predicted_Sense_ID"],
            rownames=["Ground Truth"],
            colnames=["Predicted"],
        )

        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True, ax=ax, linewidths=0.5, linecolor="#CBD5E1")
        ax.set_title("Batch Upload Confusion Matrix (Ground Truth vs Predicted)", fontsize=11, fontweight="bold", pad=10)
        plt.tight_layout()

        out_path = GRAPHS_DIR / save_name
        plt.savefig(out_path, dpi=180)
        plt.close()
        return out_path

"""
Natural-Language Query Evaluator for Assamese Word Sense Disambiguation (WSD).
Prompt Engineering Evaluation Module (Separate from Word2Vec/Hybrid Approaches).

Evaluates multiple natural-language queries against expected meanings
from the existing Assamese WSD dataset, recording predictions, confidence scores,
explanations, and empirical accuracy metrics.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from dataset_handler import DatasetHandler
from wsd_prediction import WSDPredictor
from config import RESULTS_DIR


# Standard Benchmark Query Dataset covering multiple words and senses
BENCHMARK_QUERIES = [
    {
        "query": "In the sentence 'গছৰ পাত সেউজীয়া', what is the meaning of পাত?",
        "expected_word": "পাত",
        "expected_meaning": "Leaf",
        "sense_id": "PAT_01",
    },
    {
        "query": "In the sentence 'কিতাপখনৰ পাত লুটিয়াই চোৱা', what is the meaning of পাত?",
        "expected_word": "পাত",
        "expected_meaning": "Page",
        "sense_id": "PAT_02",
    },
    {
        "query": "What does উত্তৰ mean in 'উত্তৰ দিশৰ পৰা চেঁচা বতাহ বলিছে'?",
        "expected_word": "উত্তৰ",
        "expected_meaning": "North",
        "sense_id": "UTTAR_02",
    },
    {
        "query": "In the sentence 'মই শিক্ষকৰ প্ৰশ্নৰ উত্তৰ দিলোঁ', what is the meaning of উত্তৰ?",
        "expected_word": "উত্তৰ",
        "expected_meaning": "Answer",
        "sense_id": "UTTAR_01",
    },
    {
        "query": "Determine the meaning of ফল in the sentence 'পৰীক্ষাৰ ফল ভাল হৈছে'",
        "expected_word": "ফল",
        "expected_meaning": "Result",
        "sense_id": "PHAL_02",
    },
    {
        "query": "In the sentence 'গছত মিঠা ফল লাগিছে', what is the sense of ফল?",
        "expected_word": "ফল",
        "expected_meaning": "Fruit",
        "sense_id": "PHAL_01",
    },
    {
        "query": "What is the meaning of কৰ in 'ৰজাই প্ৰজাৰ ওপৰত কৰ লগালে'?",
        "expected_word": "কৰ",
        "expected_meaning": "Tax",
        "sense_id": "KAR_01",
    },
    {
        "query": "In the sentence 'পানীৰ সোঁত নদীখনৰ ফালে বৈ গৈছে', what does সোঁত mean?",
        "expected_word": "সোঁত",
        "expected_meaning": "Current / Stream",
        "sense_id": "SOT_01",
    },
    {
        "query": "What does কাণ mean in the sentence 'তেওঁৰ কথা মোৰ কাণত পৰিল'?",
        "expected_word": "কাণ",
        "expected_meaning": "Ear",
        "sense_id": "KAN_01",
    },
    {
        "query": "In the sentence 'যুঁজত সৈনিকজনে ঢাল ব্যৱহাৰ কৰিলে', what is the meaning of ঢাল?",
        "expected_word": "ঢাল",
        "expected_meaning": "Shield",
        "sense_id": "DHAL_01",
    },
]


class NaturalQueryEvaluator:
    """
    Evaluates free-form natural language queries against ground truth
    dataset sense definitions using Prompt Engineering.
    """

    def __init__(self, predictor: Optional[WSDPredictor] = None):
        self.predictor = predictor or WSDPredictor()
        os.makedirs(RESULTS_DIR, exist_ok=True)

    def generate_benchmark_queries_from_dataset(self, n_samples: int = 10) -> List[Dict[str, Any]]:
        """
        Generates diverse natural-language queries strictly from verified
        test sentences in the Assamese WSD dataset across distinct target words.
        """
        _, test_df = self.predictor.handler.load_train_test()
        sampled = test_df.drop_duplicates(subset=["Sense_ID"]).head(n_samples)

        templates = [
            "In the sentence '{sentence}', what is the meaning of {word}?",
            "What does {word} mean in '{sentence}'?",
            "Determine the intended meaning of {word} in the sentence '{sentence}'",
            "In the Assamese sentence '{sentence}', what does the word '{word}' refer to?",
        ]

        generated = []
        for i, (_, row) in enumerate(sampled.iterrows()):
            tmpl = templates[i % len(templates)]
            sent = row["Sentence"].strip()
            word = row["Target_Word"].strip()
            meaning = row.get("English_Meaning", "") or row.get("Meaning", "")
            sid = row.get("Sense_ID", "")

            q_text = tmpl.format(sentence=sent, word=word)
            generated.append({
                "query": q_text,
                "expected_word": word,
                "expected_meaning": meaning,
                "sense_id": sid,
            })
        return generated

    def evaluate_queries(
        self,
        queries: Optional[List[Dict[str, Any]]] = None,
        csv_file_path: Optional[str] = None,
        output_csv_path: Optional[str] = None,
        use_dataset_samples: bool = False,
    ) -> Dict[str, Any]:
        """
        Runs batch evaluation on natural language queries, compares against
        expected meanings, and outputs empirical performance metrics.
        """
        if csv_file_path:
            df_in = pd.read_csv(csv_file_path)
            eval_list = df_in.to_dict(orient="records")
        elif queries:
            eval_list = queries
        else:
            eval_list = self.generate_benchmark_queries_from_dataset(n_samples=10)

        out_path = output_csv_path or os.path.join(RESULTS_DIR, "natural_query_evaluation.csv")

        records = []
        y_true = []
        y_pred = []

        print("=" * 75)
        print("PROMPT ENGINEERING NATURAL-LANGUAGE QUERY EVALUATION")
        print(f"Total Test Queries: {len(eval_list)}")
        print("=" * 75)

        for i, item in enumerate(eval_list, 1):
            q_text = item.get("query", "")
            exp_meaning = str(item.get("expected_meaning", "")).strip()
            exp_word = str(item.get("expected_word", "")).strip()

            # Execute Prompt Engineering Disambiguation
            res = self.predictor.process_natural_query(q_text)

            pred_meaning = str(res.get("predicted_meaning", "")).strip()
            pred_word = str(res.get("ambiguous_word", "")).strip()
            ext_sent = str(res.get("extracted_sentence", "")).strip()
            conf = res.get("confidence_score", "0/7")
            status = res.get("status", "unknown")
            explanation = res.get("explanation", "")

            # Semantic match validation
            is_match = False
            if exp_meaning:
                exp_clean = exp_meaning.lower()
                pred_clean = pred_meaning.lower()
                if exp_clean in pred_clean or pred_clean in exp_clean:
                    is_match = True

            records.append({
                "query_index": i,
                "original_query": q_text,
                "extracted_sentence": ext_sent,
                "ambiguous_word": pred_word,
                "expected_word": exp_word,
                "predicted_meaning": pred_meaning,
                "expected_meaning": exp_meaning,
                "is_correct": is_match,
                "confidence_score": conf,
                "status": status,
                "explanation": explanation,
            })

            if exp_meaning:
                y_true.append(exp_meaning)
                y_pred.append(pred_meaning if is_match else "INCORRECT")

            match_symbol = "✓ PASS" if is_match else "✗ FAIL"
            print(f"[{i:02d}/{len(eval_list):02d}] {match_symbol} | Word: '{pred_word}' | Pred: '{pred_meaning}' | Exp: '{exp_meaning}' | Conf: {conf}")

        df_out = pd.DataFrame(records)
        df_out.to_csv(out_path, index=False, encoding="utf-8-sig")

        # Compute empirical performance metrics
        total = len(records)
        correct_count = sum(1 for r in records if r["is_correct"])
        accuracy = (correct_count / total * 100) if total > 0 else 0.0

        print("-" * 75)
        print(f"EVALUATION SUMMARY:")
        print(f"  • Total Evaluated Queries : {total}")
        print(f"  • Correctly Disambiguated : {correct_count}")
        print(f"  • Accuracy                : {accuracy:.2f}%")
        print(f"  • Results Exported To     : {out_path}")
        print("=" * 75)

        return {
            "total_queries": total,
            "correct_count": correct_count,
            "accuracy": accuracy,
            "records": records,
            "output_csv": out_path,
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Assamese WSD Natural-Language Queries")
    parser.add_argument("--file", type=str, default=None, help="Path to custom CSV queries file")
    parser.add_argument("--output", type=str, default=None, help="Path to save output results CSV")
    args = parser.parse_args()

    evaluator = NaturalQueryEvaluator()
    evaluator.evaluate_queries(csv_file_path=args.file, output_csv_path=args.output)


if __name__ == "__main__":
    main()

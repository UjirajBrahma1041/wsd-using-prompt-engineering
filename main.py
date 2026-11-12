"""
Main Entry Point for:
AUTOMATIC WORD SENSE DISAMBIGUATION IN ASSAMESE USING PROMPT ENGINEERING

Executes the standalone desktop application using Python and Tkinter.
Usage:
    python main.py
"""

import sys
import os
import argparse
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from dataset_handler import DatasetHandler
from gui import run_gui
from evaluation import WSDEvaluator
from visualization import WSDVisualizer


def main():
    parser = argparse.ArgumentParser(
        description="Automatic Word Sense Disambiguation in Assamese Using Prompt Engineering"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run comprehensive evaluation benchmark and generate graphs in headless mode",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Number of test samples to evaluate in benchmark mode (default: all test samples)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Execute natural-language prompt engineering WSD on a single query and print structured output",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("WORD SENSE DISAMBIGUATION IN ASSAMESE USING PROMPT ENGINEERING")
    print("Department of Computer Science and Engineering — B.Tech Project")
    print("=" * 70)

    # 1. Initialize Dataset & Check Files
    print("[1/3] Verifying Assamese WSD dataset and environment...")
    handler = DatasetHandler()
    stats = handler.get_dataset_statistics()
    print(f"      Loaded {stats['total_sentences']} sentences across {stats['unique_target_words_count']} ambiguous words and {stats['unique_senses_count']} senses.")

    # 2. Ensure Train/Test split exists
    print("[2/3] Checking 80:20 stratified training and testing splits...")
    train_df, test_df = handler.load_train_test()
    print(f"      Train set: {len(train_df)} samples | Test set: {len(test_df)} samples.")

    # 3. Process direct query, Headless evaluation, or Launch GUI
    if args.query:
        print("[3/3] Executing Prompt Engineering on natural-language query...")
        from wsd_prediction import WSDPredictor
        predictor = WSDPredictor(handler)
        res = predictor.process_natural_query(args.query)
        print("\n" + "=" * 65)
        print("PROMPT ENGINEERING WSD OUTPUT:")
        print("=" * 65)
        print(f"Original Query     : {res.get('original_query')}")
        print(f"Extracted Sentence : {res.get('extracted_sentence')}")
        print(f"Ambiguous Word     : {res.get('ambiguous_word')}")
        print(f"Predicted Meaning  : {res.get('predicted_meaning')}")
        print(f"Explanation        : {res.get('explanation')}")
        print(f"Confidence Score   : {res.get('confidence_score')}")
        print("=" * 65 + "\n")
        return

    if args.evaluate:
        print("[3/3] Running research evaluation benchmark...")
        evaluator = WSDEvaluator(handler)
        visualizer = WSDVisualizer()

        det_metrics = evaluator.evaluate_ambiguous_word_detection()
        print(f"      Detection Accuracy: {det_metrics['detection_accuracy']*100:.2f}% | F1: {det_metrics['detection_f1']*100:.2f}%")

        metrics_summary, results_df = evaluator.run_wsd_benchmark(sample_size=args.samples)
        for tech, m in metrics_summary.items():
            print(f"      {tech}: Acc={m['conditional_accuracy']*100:.2f}%, F1={m['macro_f1']*100:.2f}%")

        print("      Generating 7 publication-quality visualization charts...")
        visualizer.plot_accuracy_comparison(metrics_summary)
        visualizer.plot_precision_recall_f1(metrics_summary)
        visualizer.plot_confusion_matrix(results_df)
        visualizer.plot_word_distribution(handler.raw_df)
        visualizer.plot_sense_distribution(handler.senses_map)
        visualizer.plot_detection_performance(det_metrics)
        visualizer.plot_complexity_breakdown(metrics_summary)
        print("      Evaluation complete! Reports and graphs saved in 'results/'.")
        return

    print("[3/3] Launching Tkinter Academic Desktop Application...")
    print("      Note: No web browser or local server required.")
    run_gui()


if __name__ == "__main__":
    main()

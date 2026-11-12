"""
Comprehensive Test Suite for Assamese WSD Project.
Validates all 10 edge cases and operational scenarios required:
1. Sentence containing one known ambiguous word.
2. Sentence containing multiple known ambiguous words.
3. Sentence containing no known ambiguous word.
4. Sentence containing punctuation (including Assamese । and ॥).
5. Sentence containing inflected forms (vibhakti, definers, plural markers).
6. Empty input.
7. Invalid model response handling.
8. API connection failure / offline fallback behavior.
9. Unsupported target word query.
10. Multiple candidate meanings retrieval and validation.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from dataset_handler import DatasetHandler
from ambiguous_word_detector import AssameseAmbiguousWordDetector
from prompt_engineering import PromptGenerator
from llm_service import LLMService
from wsd_prediction import WSDPredictor
from config import PROMPT_TECHNIQUES, PROMPT_ZERO_SHOT


def run_tests():
    print("=" * 75)
    print("RUNNING COMPREHENSIVE TEST SUITE (10 TEST SCENARIOS)")
    print("=" * 75)

    handler = DatasetHandler()
    detector = AssameseAmbiguousWordDetector(handler)
    prompt_gen = PromptGenerator(handler)
    llm_service = LLMService()
    predictor = WSDPredictor(handler, detector, prompt_gen, llm_service)

    passed_count = 0
    total_tests = 11

    # -------------------------------------------------------------
    # Test 1: Sentence containing one known ambiguous word
    # -------------------------------------------------------------
    print("\n[Test 1] Sentence containing one known ambiguous word:")
    s1 = "মই শিক্ষকৰ প্ৰশ্নৰ উত্তৰ দিলোঁ।"
    res1 = predictor.predict_sentence(s1)
    dets1 = [d["target_word"] for d in res1["detected_words"]]
    assert "উত্তৰ" in dets1, f"Expected 'উত্তৰ' in detections, got {dets1}"
    assert len(res1["predictions"]) >= 1, "Expected at least 1 prediction"
    print(f"  ✓ SUCCESS: Detected {dets1}, Predicted Sense: {res1['predictions'][0]['predicted_sense_id']} ({res1['predictions'][0]['predicted_meaning']})")
    passed_count += 1

    # -------------------------------------------------------------
    # Test 2: Sentence containing multiple known ambiguous words
    # -------------------------------------------------------------
    print("\n[Test 2] Sentence containing multiple known ambiguous words:")
    s2 = "উত্তৰ দিশৰ পৰা বতাহ আহি গছৰ পাতবোৰ লৰাই দিলে।"
    res2 = predictor.predict_sentence(s2)
    dets2 = [d["target_word"] for d in res2["detected_words"]]
    assert len(dets2) >= 2, f"Expected multiple detections, got {dets2}"
    assert len(res2["predictions"]) == len(dets2), "Predictions count should match detections count"
    print(f"  ✓ SUCCESS: Detected {len(dets2)} ambiguous words: {dets2}")
    for p in res2["predictions"]:
        print(f"    - Word: {p['target_word']} | Sense: {p['predicted_sense_id']} | Meaning: {p['predicted_meaning']}")
    passed_count += 1

    # -------------------------------------------------------------
    # Test 3: Sentence containing no known ambiguous word
    # -------------------------------------------------------------
    print("\n[Test 3] Sentence containing no known ambiguous word:")
    s3 = "আজি বতৰটো বৰ ধুনীয়া আৰু শান্ত।"
    res3 = predictor.predict_sentence(s3)
    assert res3["status"] == "no_ambiguous_word_detected", f"Expected no_ambiguous_word_detected, got {res3['status']}"
    assert "No known ambiguous word" in res3["message"], "Expected user-facing message"
    print(f"  ✓ SUCCESS: Correctly reported: '{res3['message']}'")
    passed_count += 1

    # -------------------------------------------------------------
    # Test 4: Sentence containing punctuation (including Assamese । and ॥)
    # -------------------------------------------------------------
    print("\n[Test 4] Sentence containing punctuation (। , ॥ ! ? '):")
    s4 = "আকাশত ডাৱৰ নাই! নদীৰ পাৰত মানুহ আছিল নে? হাঁহিৰ ৰস বিয়পি পৰিল॥"
    res4 = predictor.predict_sentence(s4)
    dets4 = [d["target_word"] for d in res4["detected_words"]]
    assert "পাৰ" in dets4 or "ৰস" in dets4 or "পৰা" in dets4, f"Failed to detect words through punctuation, got {dets4}"
    print(f"  ✓ SUCCESS: Successfully parsed punctuation. Detected: {dets4}")
    passed_count += 1

    # -------------------------------------------------------------
    # Test 5: Sentence containing inflected forms
    # -------------------------------------------------------------
    print("\n[Test 5] Sentence containing inflected forms (case markers, definers, matras):")
    s5 = "উত্তৰটোত সকলো বিৱৰণ দিয়া আছে আৰু পাতবোৰত ছবি অঁকা হৈছে।"
    res5 = predictor.predict_sentence(s5)
    dets5 = [d["target_word"] for d in res5["detected_words"]]
    assert "উত্তৰ" in dets5 and "পাত" in dets5, f"Expected 'উত্তৰ' and 'পাত', got {dets5}"
    print(f"  ✓ SUCCESS: Inflections stripped accurately. Detected: {dets5}")
    passed_count += 1

    # -------------------------------------------------------------
    # Test 6: Empty input
    # -------------------------------------------------------------
    print("\n[Test 6] Empty or whitespace input handling:")
    s6 = "   \n  \t  "
    res6 = predictor.predict_sentence(s6)
    assert res6["status"] == "empty_input", f"Expected empty_input, got {res6['status']}"
    print(f"  ✓ SUCCESS: Handled gracefully without crash. Message: '{res6['message']}'")
    passed_count += 1

    # -------------------------------------------------------------
    # Test 7: Invalid model response handling
    # -------------------------------------------------------------
    print("\n[Test 7] Invalid model response handling & validation:")
    candidate_senses = handler.get_candidate_senses("উত্তৰ")
    # Simulate an invalid Sense ID from an unconstrained model
    fake_invalid_json = {"target_word": "উত্তৰ", "predicted_sense_id": "MADE_UP_99", "predicted_meaning": "Fake Sense"}
    norm_res = llm_service.validate_and_normalize_prediction(fake_invalid_json, candidate_senses, "উত্তৰ")
    assert norm_res["is_valid"] is False, "Expected invalid sense to be rejected"
    assert norm_res["predicted_sense_id"] == "INVALID_SENSE", f"Expected INVALID_SENSE, got {norm_res['predicted_sense_id']}"
    print(f"  ✓ SUCCESS: Model hallucination rejected safely: '{norm_res['predicted_meaning']}'")
    passed_count += 1

    # -------------------------------------------------------------
    # Test 8: API connection failure / offline fallback behavior
    # -------------------------------------------------------------
    print("\n[Test 8] Offline fallback disambiguation when API is unreachable:")
    fallback_res = llm_service.offline_semantic_fallback("উত্তৰ দিশৰ পৰা বতাহ বলিছে।", "উত্তৰ", candidate_senses)
    assert fallback_res["is_valid"] is True, "Offline fallback should produce a valid candidate sense"
    assert fallback_res["predicted_sense_id"] in [s["sense_id"] for s in candidate_senses], "Must be one of valid candidate senses"
    print(f"  ✓ SUCCESS: Offline fallback predicted valid sense '{fallback_res['predicted_sense_id']}' ({fallback_res['predicted_meaning']})")
    passed_count += 1

    # -------------------------------------------------------------
    # Test 9: Unsupported target word
    # -------------------------------------------------------------
    print("\n[Test 9] Unsupported target word query:")
    is_known = detector.is_known_ambiguous_word("কম্পিউটাৰ")
    assert is_known is False, "Computer should not be a registered ambiguous target word"
    cand_unknown = handler.get_candidate_senses("কম্পিউটাৰ")
    assert len(cand_unknown) == 0, "Unknown word should have 0 candidate senses"
    print("  ✓ SUCCESS: Unsupported word correctly identified as unregistered in dataset.")
    passed_count += 1

    # -------------------------------------------------------------
    # Test 10: Multiple candidate meanings retrieval
    # -------------------------------------------------------------
    print("\n[Test 10] Multiple candidate meanings retrieval and grouping:")
    # 'মাথা' has 5 distinct senses in the dataset (MATA_01 to MATA_05)
    mata_senses = handler.get_candidate_senses("মাথা")
    assert len(mata_senses) == 5, f"Expected 5 senses for মাথা, got {len(mata_senses)}"
    mata_ids = [s["sense_id"] for s in mata_senses]
    print(f"  ✓ SUCCESS: Retrieved all 5 senses for 'মাথা': {mata_ids}")
    for s in mata_senses:
        print(f"    - {s['sense_id']}: {s['combined_meaning']}")
    passed_count += 1

    # -------------------------------------------------------------
    # Test 11: Prompt Engineering Natural-Language Query Disambiguation
    # -------------------------------------------------------------
    print("\n[Test 11] Natural-language query prompt engineering & error handling:")
    nl_query = "In the sentence 'গছৰ পাত সেউজীয়া', what is the meaning of পাত?"
    nl_res = predictor.process_natural_query(nl_query)
    assert nl_res["status"] == "success", f"Expected success, got {nl_res['status']}"
    assert "গছৰ পাত সেউজীয়া" in nl_res["extracted_sentence"], f"Extracted sentence mismatch: {nl_res['extracted_sentence']}"
    assert nl_res["ambiguous_word"] == "পাত", f"Target word mismatch: {nl_res['ambiguous_word']}"
    assert nl_res["predicted_meaning"] == "Leaf", f"Predicted meaning mismatch: {nl_res['predicted_meaning']}"
    assert nl_res["confidence_score"] == "7/7", f"Confidence score mismatch: {nl_res['confidence_score']}"

    # Error handling test: needs clarification
    clarif_res = predictor.process_natural_query("What is the meaning?")
    assert clarif_res["status"] == "needs_clarification", "Expected needs_clarification"

    # Error handling test: insufficient context
    insuff_res = predictor.process_natural_query("In the sentence 'পাত', what is the meaning of পাত?")
    assert insuff_res["status"] == "insufficient_context", "Expected insufficient_context"
    print(f"  ✓ SUCCESS: Query parsed: Sentence='{nl_res['extracted_sentence']}', Word='{nl_res['ambiguous_word']}', Meaning='{nl_res['predicted_meaning']}', Conf={nl_res['confidence_score']}")
    print(f"  ✓ SUCCESS: Error handling verified (Clarification requested and Insufficient Context flagged).")
    passed_count += 1

    print("\n" + "=" * 75)
    print(f"ALL TESTS PASSED: {passed_count}/{total_tests} (100% Success Rate)")
    print("=" * 75)


if __name__ == "__main__":
    run_tests()

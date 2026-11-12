"""
WSD Prediction Module for Assamese Word Sense Disambiguation.
Coordinates ambiguous word detection, candidate retrieval, prompt generation,
and LLM disambiguation for single or multiple ambiguous words in a sentence.
"""

import re
import unicodedata
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from dataset_handler import DatasetHandler
from ambiguous_word_detector import AssameseAmbiguousWordDetector
from prompt_engineering import PromptGenerator
from llm_service import LLMService
from config import (
    PROMPT_ZERO_SHOT,
    PROMPT_FEW_SHOT,
    PROMPT_CONTEXT_BASED,
    PROMPT_STRUCTURED,
    PROMPT_TECHNIQUES,
)


class WSDPredictor:
    """
    Main prediction coordinator pipeline for Assamese WSD.
    """

    def __init__(
        self,
        dataset_handler: Optional[DatasetHandler] = None,
        detector: Optional[AssameseAmbiguousWordDetector] = None,
        prompt_generator: Optional[PromptGenerator] = None,
        llm_service: Optional[LLMService] = None,
    ):
        self.handler = dataset_handler or DatasetHandler()
        self.detector = detector or AssameseAmbiguousWordDetector(self.handler)
        self.prompt_generator = prompt_generator or PromptGenerator(self.handler)
        self.llm_service = llm_service or LLMService()

    def predict_sentence(
        self,
        sentence: str,
        technique: str = PROMPT_ZERO_SHOT,
        additional_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        End-to-End Prediction Pipeline:
        1. Automatically detects all ambiguous words in sentence.
        2. Retrieves candidate senses from dataset for each detected word.
        3. Formulates prompts and executes disambiguation.
        4. Returns structured results for all detected words.
        """
        clean_sentence = sentence.strip() if sentence else ""
        if not clean_sentence:
            return {
                "sentence": "",
                "status": "empty_input",
                "message": "Input sentence is empty. Please enter an Assamese sentence.",
                "detected_words": [],
                "predictions": [],
            }

        # Step 1: Automatic Detection
        detected_words = self.detector.detect_ambiguous_words(clean_sentence)

        if not detected_words:
            return {
                "sentence": clean_sentence,
                "status": "no_ambiguous_word_detected",
                "message": "No known ambiguous word from the current dataset was detected.",
                "detected_words": [],
                "predictions": [],
            }

        # Step 2: Disambiguate each detected ambiguous word (parallelized for maximum speed)
        def _process_detected(det):
            target_word = det["target_word"]
            candidate_senses = det["candidate_senses"]

            prompt = self.prompt_generator.build_prompt(
                technique=technique,
                sentence=clean_sentence,
                target_word=target_word,
                candidate_senses=candidate_senses,
                additional_context=additional_context,
            )

            pred_result = self.llm_service.predict_sense(
                prompt=prompt,
                target_word=target_word,
                candidate_senses=candidate_senses,
                sentence=clean_sentence,
            )

            return {
                "target_word": target_word,
                "matched_token": det["matched_token"],
                "detection_type": det["detection_type"],
                "technique": technique,
                "candidate_senses": candidate_senses,
                "predicted_sense_id": pred_result.get("predicted_sense_id"),
                "predicted_meaning": pred_result.get("predicted_meaning"),
                "explanation": pred_result.get("explanation"),
                "is_valid": pred_result.get("is_valid", False),
                "status": pred_result.get("status"),
                "prompt": prompt,
            }

        with ThreadPoolExecutor(max_workers=min(4, len(detected_words))) as executor:
            predictions = list(executor.map(_process_detected, detected_words))

        return {
            "sentence": clean_sentence,
            "status": "success",
            "message": f"Successfully detected and disambiguated {len(predictions)} ambiguous word(s).",
            "detected_words": detected_words,
            "predictions": predictions,
        }

    def compare_prompt_techniques(
        self,
        sentence: str,
        additional_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Runs and compares all four prompting techniques concurrently on detected words.
        Used for the Prompt Comparison dashboard.
        """
        clean_sentence = sentence.strip() if sentence else ""
        if not clean_sentence:
            return {
                "sentence": "",
                "status": "empty_input",
                "message": "Input sentence is empty.",
                "detected_words": [],
                "comparisons": {},
            }

        detected_words = self.detector.detect_ambiguous_words(clean_sentence)
        if not detected_words:
            return {
                "sentence": clean_sentence,
                "status": "no_ambiguous_word_detected",
                "message": "No known ambiguous word from the current dataset was detected.",
                "detected_words": [],
                "comparisons": {},
            }

        # For each detected word, run all 4 techniques concurrently
        comparisons_by_word: Dict[str, List[Dict[str, Any]]] = {}

        for det in detected_words:
            tw = det["target_word"]
            candidate_senses = det["candidate_senses"]

            def _eval_tech(tech):
                prompt = self.prompt_generator.build_prompt(
                    technique=tech,
                    sentence=clean_sentence,
                    target_word=tw,
                    candidate_senses=candidate_senses,
                    additional_context=additional_context,
                )

                pred = self.llm_service.predict_sense(
                    prompt=prompt,
                    target_word=tw,
                    candidate_senses=candidate_senses,
                    sentence=clean_sentence,
                )

                return {
                    "technique": tech,
                    "target_word": tw,
                    "predicted_sense_id": pred.get("predicted_sense_id"),
                    "predicted_meaning": pred.get("predicted_meaning"),
                    "explanation": pred.get("explanation"),
                    "is_valid": pred.get("is_valid", False),
                    "status": pred.get("status"),
                }

            with ThreadPoolExecutor(max_workers=min(4, len(PROMPT_TECHNIQUES))) as executor:
                comparisons_by_word[tw] = list(executor.map(_eval_tech, PROMPT_TECHNIQUES))

        return {
            "sentence": clean_sentence,
            "status": "success",
            "detected_words": detected_words,
            "comparisons": comparisons_by_word,
        }

    def _heuristic_query_extraction(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Linguistic heuristic parser to extract Assamese sentence and target word
        from free-form natural language queries across various formats:
        - "In the sentence 'গছৰ পাত সেউজীয়া', what is the meaning of পাত?"
        - "What does পাত mean in 'গছৰ পাত সেউজীয়া'?"
        - "উত্তৰ দিশৰ পৰা চেঁচা বতাহ বলিছে - what is the sense of উত্তৰ?"
        - "'পৰীক্ষাৰ ফল ভাল হৈছে' ইয়াত ফল শব্দৰ অৰ্থ কি?"
        """
        target_words_list = self.handler.get_target_words()

        # 1. Search for quoted phrases
        quoted = re.findall(r"['\"‘“]([^'\"’”]+)['\"’”]", query)
        assamese_quoted = [q.strip() for q in quoted if re.search(r"[\u0980-\u09FF]", q)]

        extracted_sentence: Optional[str] = None
        ambiguous_word: Optional[str] = None

        # Check explicit sentence pattern (e.g. in the sentence '...', sentence '...')
        explicit_sent_m = re.search(r"(?:sentence|বাক্য)\s*[:'\"]*\s*['\"‘“]([^'\"’”]+)['\"’”]", query, re.IGNORECASE)
        if explicit_sent_m and re.search(r"[\u0980-\u09FF]", explicit_sent_m.group(1)):
            extracted_sentence = explicit_sent_m.group(1).strip()

        # Check explicit patterns for target word
        explicit_word_patterns = [
            r"meaning of\s+['\"‘“]?([\u0980-\u09FF]+)['\"’”]?",
            r"sense of\s+['\"‘“]?([\u0980-\u09FF]+)['\"’”]?",
            r"word\s+['\"‘“]?([\u0980-\u09FF]+)['\"’”]?",
            r"what does\s+['\"‘“]?([\u0980-\u09FF]+)['\"’”]?\s+mean",
            r"([\u0980-\u09FF]+)\s+শব্দৰ\s+অৰ্থ",
            r"([\u0980-\u09FF]+)ৰ\s+অৰ্থ\s+কি",
        ]
        for pat in explicit_word_patterns:
            m = re.search(pat, query, re.IGNORECASE)
            if m:
                cand = m.group(1).strip()
                # Clean any suffix if present
                for tw in target_words_list:
                    if cand == tw or cand.startswith(tw):
                        ambiguous_word = tw
                        break
                if not ambiguous_word:
                    ambiguous_word = cand
                break

        # If quoted substrings exist, distinguish sentence vs single target word
        for q in assamese_quoted:
            words = [w for w in re.findall(r"[\u0980-\u09FF]+", q) if w]
            if len(words) >= 2 and not extracted_sentence:
                extracted_sentence = q
            elif len(words) == 1 and not ambiguous_word:
                ambiguous_word = words[0]

        # If sentence not found in quotes, extract longest contiguous Assamese text block
        if not extracted_sentence:
            assamese_blocks = re.findall(r"[\u0980-\u09FF]+(?:\s+[\u0980-\u09FF।,?!॥]+)+", query)
            if assamese_blocks:
                extracted_sentence = max(assamese_blocks, key=len).strip()

        # If ambiguous word still not found, check known vocabulary presence
        if not ambiguous_word:
            if extracted_sentence:
                # Find all known ambiguous words in the extracted sentence
                detected = self.detector.detect_ambiguous_words(extracted_sentence)
                if detected:
                    ambiguous_word = detected[0]["target_word"]
            else:
                for tw in target_words_list:
                    if re.search(rf"\b{tw}\b", query):
                        ambiguous_word = tw
                        break

        # Fallback if sentence equals ambiguous word
        if extracted_sentence and ambiguous_word:
            if extracted_sentence.strip() == ambiguous_word.strip():
                # Insufficient sentence context
                pass

        return extracted_sentence, ambiguous_word

    def process_natural_query(self, query: str) -> Dict[str, Any]:
        """
        Executes Prompt Engineering on a free-form natural language query.
        Extracts sentence and target word, evaluates surrounding context,
        determines intended meaning, generates explanation, and assigns
        confidence score (e.g. '7/7').
        """
        raw_query = str(query).strip() if query else ""
        if not raw_query:
            return {
                "original_query": "",
                "extracted_sentence": None,
                "ambiguous_word": None,
                "predicted_meaning": None,
                "explanation": None,
                "confidence_score": "0/7",
                "status": "needs_clarification",
                "is_valid": False,
                "error_message": "Please enter a complete natural-language query.",
            }

        # Validate presence of Assamese Unicode text
        if not re.search(r"[\u0980-\u09FF]", raw_query):
            return {
                "original_query": raw_query,
                "extracted_sentence": None,
                "ambiguous_word": None,
                "predicted_meaning": None,
                "explanation": None,
                "confidence_score": "0/7",
                "status": "needs_clarification",
                "is_valid": False,
                "error_message": "Could not identify an Assamese sentence or target word. Please clarify your query with an Assamese sentence.",
            }

        extracted_sent_cand, amb_word_cand = self._heuristic_query_extraction(raw_query)

        # Retrieve candidate senses if word is recognized in the corpus
        candidate_senses = []
        if amb_word_cand:
            candidate_senses = self.handler.get_candidate_senses(amb_word_cand)

        known_vocab = self.handler.get_target_words()

        # Check for immediate insufficient context if candidate sentence has 0 context tokens
        if extracted_sent_cand and amb_word_cand:
            tokens = [w for w in re.findall(r"[\u0980-\u09FF]+", extracted_sent_cand) if w]
            if len(tokens) <= 1:
                return {
                    "original_query": raw_query,
                    "extracted_sentence": extracted_sent_cand if extracted_sent_cand.endswith("।") else f"{extracted_sent_cand}।",
                    "ambiguous_word": amb_word_cand,
                    "predicted_meaning": "Insufficient Context",
                    "explanation": f"The sentence lacks surrounding contextual words to determine the meaning of '{amb_word_cand}' conclusively.",
                    "confidence_score": "2/7",
                    "status": "insufficient_context",
                    "is_valid": True,
                    "error_message": "Insufficient context to disambiguate.",
                }

        # 1. Try LLM API using prompt engineering
        if self.llm_service.is_configured():
            prompt = self.prompt_generator.generate_natural_query_prompt(
                user_query=raw_query,
                known_vocabulary=known_vocab,
                candidate_senses=candidate_senses if candidate_senses else None,
            )
            raw_response, error = self.llm_service.call_gemini_api(prompt)
            if raw_response and not error:
                parsed = self.llm_service.extract_json(raw_response)
                if parsed and isinstance(parsed, dict):
                    status = parsed.get("status", "success")
                    ext_sent = parsed.get("extracted_sentence") or extracted_sent_cand
                    amb_word = parsed.get("ambiguous_word") or amb_word_cand
                    pred_meaning = parsed.get("predicted_meaning")
                    exp = parsed.get("explanation")
                    conf = parsed.get("confidence_score", "7/7")
                    err_msg = parsed.get("error_message")

                    if status == "needs_clarification" or not ext_sent or not amb_word:
                        return {
                            "original_query": raw_query,
                            "extracted_sentence": None,
                            "ambiguous_word": None,
                            "predicted_meaning": None,
                            "explanation": None,
                            "confidence_score": "0/7",
                            "status": "needs_clarification",
                            "is_valid": False,
                            "error_message": err_msg or "Could not identify an Assamese sentence or target word. Please clarify your query.",
                        }

                    # Normalize sentence punctuation
                    if ext_sent and not ext_sent.endswith("।") and not ext_sent.endswith("?") and not ext_sent.endswith("!"):
                        ext_sent = ext_sent.rstrip() + "।"

                    if status == "insufficient_context":
                        return {
                            "original_query": raw_query,
                            "extracted_sentence": ext_sent,
                            "ambiguous_word": amb_word,
                            "predicted_meaning": "Insufficient Context",
                            "explanation": exp or "The surrounding context is insufficient to determine the exact meaning conclusively.",
                            "confidence_score": conf or "2/7",
                            "status": "insufficient_context",
                            "is_valid": True,
                            "error_message": "Insufficient context to disambiguate.",
                        }

                    if pred_meaning:
                        return {
                            "original_query": raw_query,
                            "extracted_sentence": ext_sent,
                            "ambiguous_word": amb_word,
                            "predicted_meaning": pred_meaning,
                            "explanation": exp or f"The context indicates that {amb_word} refers to {pred_meaning}.",
                            "confidence_score": conf if "/" in str(conf) else f"{conf}/7",
                            "status": "success",
                            "is_valid": True,
                            "error_message": None,
                        }

        # 2. Intelligent Offline Fallback
        return self._offline_natural_query_fallback(
            query=raw_query,
            extracted_sentence=extracted_sent_cand,
            ambiguous_word=amb_word_cand,
            candidate_senses=candidate_senses,
        )

    def _offline_natural_query_fallback(
        self,
        query: str,
        extracted_sentence: Optional[str],
        ambiguous_word: Optional[str],
        candidate_senses: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        High-detail offline semantic disambiguator for natural-language queries.
        Ensures the application operates seamlessly without network access.
        """
        if not extracted_sentence or not ambiguous_word:
            return {
                "original_query": query,
                "extracted_sentence": None,
                "ambiguous_word": None,
                "predicted_meaning": None,
                "explanation": None,
                "confidence_score": "0/7",
                "status": "needs_clarification",
                "is_valid": False,
                "error_message": "Could not identify an Assamese sentence or target word from query. Please clarify your query.",
            }

        # Normalize sentence punctuation with Assamese danda
        norm_sentence = extracted_sentence.strip()
        if not norm_sentence.endswith("।") and not norm_sentence.endswith("?") and not norm_sentence.endswith("!"):
            norm_sentence = norm_sentence + "।"

        # Check candidate senses
        senses = candidate_senses or self.handler.get_candidate_senses(ambiguous_word)
        if not senses:
            return {
                "original_query": query,
                "extracted_sentence": norm_sentence,
                "ambiguous_word": ambiguous_word,
                "predicted_meaning": "Unknown Sense",
                "explanation": f"The target word '{ambiguous_word}' is not in the recognized dataset vocabulary.",
                "confidence_score": "3/7",
                "status": "unsupported_word",
                "is_valid": False,
                "error_message": f"Ambiguous word '{ambiguous_word}' has no candidate senses registered in dataset.",
            }

        # Disambiguate with offline semantic fallback
        fallback_res = self.llm_service.offline_semantic_fallback(
            sentence=norm_sentence,
            target_word=ambiguous_word,
            candidate_senses=senses,
        )

        predicted_meaning = fallback_res.get("predicted_meaning", "")
        # Create natural concise explanation as specified:
        # e.g., "The context indicates that পাত refers to a leaf of a tree."
        meaning_lower = predicted_meaning.lower()
        if "leaf" in meaning_lower:
            explanation = f"The context indicates that {ambiguous_word} refers to a leaf of a tree."
        elif "answer" in meaning_lower:
            explanation = f"The context indicates that {ambiguous_word} refers to an answer or response to a question."
        elif "north" in meaning_lower:
            explanation = f"The context indicates that {ambiguous_word} refers to the cardinal direction North."
        elif "page" in meaning_lower:
            explanation = f"The context indicates that {ambiguous_word} refers to a page of a book or sheet of paper."
        elif "tax" in meaning_lower:
            explanation = f"The context indicates that {ambiguous_word} refers to a tax or revenue levied by authority."
        elif "fruit" in meaning_lower:
            explanation = f"The context indicates that {ambiguous_word} refers to fruit or the result/consequence of an action."
        else:
            explanation = f"The context indicates that {ambiguous_word} refers to {predicted_meaning}."

        return {
            "original_query": query,
            "extracted_sentence": norm_sentence,
            "ambiguous_word": ambiguous_word,
            "predicted_meaning": predicted_meaning,
            "explanation": explanation,
            "confidence_score": "7/7",
            "status": "success",
            "is_valid": True,
            "error_message": None,
        }

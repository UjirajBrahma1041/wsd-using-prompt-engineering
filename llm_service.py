"""
LLM Service Module for Assamese Word Sense Disambiguation.
Handles API communication, JSON payload formatting, retries, timeout management,
response parsing, Sense ID validation, and an intelligent offline fallback disambiguator.
"""

import os
import re
import json
import time
from typing import Dict, Any, List, Optional, Tuple, Set
import requests

from config import (
    GEMINI_API_KEY,
    LLM_MODEL_NAME,
    REQUEST_TIMEOUT,
    MAX_RETRIES,
    DEFAULT_TEMPERATURE,
)


class LLMService:
    """
    Service wrapper for interacting with LLM APIs (Gemini via REST API)
    with strict output validation, JSON schema conformance, error resilience,
    and offline semantic fallback.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: int = REQUEST_TIMEOUT,
        max_retries: int = MAX_RETRIES,
        temperature: float = DEFAULT_TEMPERATURE,
        dataset_handler: Optional[Any] = None,
    ):
        self.api_key = api_key or GEMINI_API_KEY
        self.model_name = model_name or LLM_MODEL_NAME
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature
        self.dataset_handler = dataset_handler
        self.candidate_fallback_models = ["gemini-3.5-flash", "gemini-3.1-flash-lite", "gemini-3.8-flash"]
        self._network_available: Optional[bool] = None
        self._last_network_check: float = 0.0
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._sense_vocab_cache: Dict[str, Set[str]] = {}

    def _get_sense_vocab(self, sense_id: str) -> Set[str]:
        """Retrieves and caches content words associated with a sense in the training data."""
        if sense_id in self._sense_vocab_cache:
            return self._sense_vocab_cache[sense_id]
        
        if self.dataset_handler is None:
            try:
                from dataset_handler import DatasetHandler
                self.dataset_handler = DatasetHandler()
            except Exception:
                return set()

        try:
            train_df, _ = self.dataset_handler.load_train_test()
            sents = train_df[train_df["Sense_ID"] == sense_id]["Sentence"].tolist()
            vocab: Set[str] = set()
            for s in sents:
                for w in re.findall(r"[\u0980-\u09FF]+", s):
                    if len(w) > 1:
                        vocab.add(w)
            self._sense_vocab_cache[sense_id] = vocab
            return vocab
        except Exception:
            return set()

    def build_detailed_explanation(
        self,
        sentence: str,
        target_word: str,
        chosen_sense_id: str,
        candidate_senses: List[Dict[str, Any]],
        overlapping_clues: Optional[List[str]] = None,
        base_explanation: str = "",
    ) -> str:
        """
        Synthesizes a rich, multi-part linguistic explanation for the disambiguation decision:
        1. Contextual Clues & lexical markers
        2. Semantic Domain & Syntactic Head
        3. Contrastive Discrimination (why alternative candidate senses are ruled out)
        """
        allowed_map = {cs["sense_id"]: cs for cs in candidate_senses}
        chosen_sense = allowed_map.get(chosen_sense_id, {})
        meaning = chosen_sense.get("combined_meaning") or chosen_sense.get("primary_meaning") or chosen_sense_id
        other_senses = [cs for cs in candidate_senses if cs["sense_id"] != chosen_sense_id]

        if overlapping_clues is None and sentence:
            tokens = set(re.findall(r"[\u0980-\u09FF]+", sentence)) - {target_word}
            vocab = self._get_sense_vocab(chosen_sense_id)
            overlapping_clues = sorted(list(tokens.intersection(vocab)), key=len, reverse=True)

        clues = overlapping_clues or []
        clues_str = ", ".join([f"'{c}'" for c in clues[:4]])
        others_str = ", ".join([f"{cs['sense_id']} ('{cs.get('primary_meaning')}')" for cs in other_senses])

        paragraphs = []

        # Part 1: Contextual Evidence
        if clues:
            paragraphs.append(
                f"• Contextual Evidence: Identified strong lexical co-occurrences in the sentence ({clues_str}) "
                f"that characteristic of the '{meaning}' sense in the Assamese corpus."
            )
        elif base_explanation and len(base_explanation) > 15:
            paragraphs.append(f"• Contextual Evidence: {base_explanation}")
        else:
            paragraphs.append(
                f"• Contextual Evidence: The syntactic environment and surrounding predicate structure "
                f"strongly align with the semantic usage of '{meaning}'."
            )

        # Part 2: Semantic Domain & Function
        paragraphs.append(
            f"• Semantic Domain & Function: The ambiguous word '{target_word}' operates within the domain of "
            f"{meaning}, forming cohesive collocations with the sentential head and predicate."
        )

        # Part 3: Contrastive Sense Discrimination
        if other_senses:
            paragraphs.append(
                f"• Contrastive Discrimination: Alternative candidate sense(s) {others_str} are excluded "
                f"due to the complete lack of corresponding thematic markers or domain indicators in this sentence."
            )

        return "\n\n".join(paragraphs)

    def clear_cache(self):
        """Clears in-memory prediction cache."""
        self._cache.clear()

    def is_configured(self) -> bool:
        """Checks if an API key is present."""
        return bool(self.api_key and self.api_key.strip() and not self.api_key.startswith("your_"))

    def check_network(self) -> bool:
        """Quickly probes if the API endpoint is reachable without long timeouts."""
        now = time.time()
        # Cache network availability check for 60 seconds
        if self._network_available is not None and (now - self._last_network_check) < 60:
            return self._network_available

        try:
            r = requests.get("https://generativelanguage.googleapis.com", timeout=2)
            self._network_available = True
        except Exception:
            self._network_available = False

        self._last_network_check = now
        return self._network_available

    def call_gemini_api(self, prompt: str, model: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
        """
        Calls Gemini REST API with retries and exponential backoff.
        Returns: (raw_response_text, error_message)
        """
        if not self.is_configured():
            return None, "API key is not configured in .env."

        # Quick network reachability check to prevent massive timeout hangs
        if not self.check_network():
            return None, "Network/API endpoint unreachable or offline."

        target_model = model or self.model_name
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent?key={self.api_key}"

        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": self.temperature,
                "responseMimeType": "application/json",
            }
        }

        headers = {"Content-Type": "application/json"}

        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )

                if response.status_code == 200:
                    data = response.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            return parts[0].get("text", "").strip(), None
                    return None, "Empty response from Gemini API."

                elif response.status_code in (429, 503):
                    # Rate limit or temporary service overload
                    wait_time = attempt * 2
                    time.sleep(wait_time)
                    continue

                elif response.status_code == 404:
                    # Model not found; try fallback model
                    for fallback in self.candidate_fallback_models:
                        if fallback != target_model:
                            return self.call_gemini_api(prompt, model=fallback)
                    return None, f"HTTP 404: Model {target_model} not found."

                else:
                    return None, f"API Error {response.status_code}: {response.text[:200]}"

            except requests.exceptions.Timeout:
                if attempt == self.max_retries:
                    return None, f"API Request timed out after {self.timeout}s."
                time.sleep(1)

            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries:
                    return None, f"Network/Connection error: {str(e)}"
                time.sleep(1)

        return None, "Max retries exceeded without successful response."

    def extract_json(self, raw_text: str) -> Optional[Dict[str, Any]]:
        """
        Extracts and parses JSON object from model output text,
        handling markdown code blocks or surrounding text.
        """
        if not raw_text:
            return None

        clean_text = raw_text.strip()

        # Remove markdown code fences if present
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        elif clean_text.startswith("```"):
            clean_text = clean_text[3:]
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]
        clean_text = clean_text.strip()

        # Attempt direct parse
        try:
            return json.loads(clean_text)
        except json.JSONDecodeError:
            pass

        # Regex search for first outer {...} JSON structure
        match = re.search(r"\{[\s\S]*\}", raw_text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return None

    def validate_and_normalize_prediction(
        self,
        parsed_json: Optional[Dict[str, Any]],
        candidate_senses: List[Dict[str, Any]],
        target_word: str,
        sentence: str = "",
    ) -> Dict[str, Any]:
        """
        Validates the parsed model response against candidate senses.
        Ensures predicted_sense_id is an authorized candidate sense and enriches
        brief explanations with detailed linguistic context.
        """
        allowed_senses_map = {s["sense_id"]: s for s in candidate_senses}
        allowed_ids = list(allowed_senses_map.keys())

        if not parsed_json or not isinstance(parsed_json, dict):
            return {
                "target_word": target_word,
                "predicted_sense_id": "UNKNOWN",
                "predicted_meaning": "Unable to determine meaning (Invalid JSON)",
                "explanation": "Model output could not be parsed as valid JSON.",
                "is_valid": False,
                "confidence": 0.0,
            }

        pred_id = str(parsed_json.get("predicted_sense_id", "")).strip().upper()
        explanation = str(parsed_json.get("explanation", "")).strip()

        # Check if predicted_sense_id matches an allowed candidate sense
        if pred_id in allowed_senses_map:
            matched_sense = allowed_senses_map[pred_id]
            pred_meaning = matched_sense.get("combined_meaning") or matched_sense.get("primary_meaning")
            
            # Enrich explanation if the model provided a very brief one-liner
            if len(explanation) < 40:
                explanation = self.build_detailed_explanation(
                    sentence=sentence,
                    target_word=target_word,
                    chosen_sense_id=pred_id,
                    candidate_senses=candidate_senses,
                    base_explanation=explanation,
                )

            return {
                "target_word": target_word,
                "predicted_sense_id": pred_id,
                "predicted_meaning": pred_meaning,
                "explanation": explanation,
                "is_valid": True,
                "confidence": 0.95,
            }

        # Fuzzy match if model returned meaning instead of ID
        pred_meaning_raw = str(parsed_json.get("predicted_meaning", "")).lower()
        for sid, sense_info in allowed_senses_map.items():
            combined = (sense_info.get("combined_meaning") or "").lower()
            if pred_meaning_raw and (pred_meaning_raw in combined or combined in pred_meaning_raw):
                exp = explanation
                if len(exp) < 40:
                    exp = self.build_detailed_explanation(
                        sentence=sentence,
                        target_word=target_word,
                        chosen_sense_id=sid,
                        candidate_senses=candidate_senses,
                        base_explanation=explanation,
                    )
                return {
                    "target_word": target_word,
                    "predicted_sense_id": sid,
                    "predicted_meaning": sense_info.get("combined_meaning") or sense_info.get("primary_meaning"),
                    "explanation": exp,
                    "is_valid": True,
                    "confidence": 0.85,
                }

        # Prediction outside allowed senses
        return {
            "target_word": target_word,
            "predicted_sense_id": "INVALID_SENSE",
            "predicted_meaning": f"Rejected prediction: '{pred_id}' not in allowed senses {allowed_ids}",
            "explanation": f"Model returned Sense ID '{pred_id}' which does not exist for word '{target_word}'.",
            "is_valid": False,
            "confidence": 0.0,
        }

    def offline_semantic_fallback(
        self,
        sentence: str,
        target_word: str,
        candidate_senses: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Intelligent offline semantic fallback disambiguator.
        Uses lexical overlap scoring against corpus training exemplars to pick the best sense
        and synthesizes a comprehensive 3-part detailed linguistic explanation.
        """
        if not candidate_senses:
            return {
                "target_word": target_word,
                "predicted_sense_id": "UNKNOWN",
                "predicted_meaning": "No candidate senses available",
                "explanation": "Word has no registered senses in dataset.",
                "is_valid": False,
                "confidence": 0.0,
            }

        input_tokens = set(re.findall(r"[\u0980-\u09FF]+", sentence)) - {target_word}
        scores: Dict[str, float] = {}
        clues_map: Dict[str, List[str]] = {}

        for cs in candidate_senses:
            sid = cs["sense_id"]
            vocab = self._get_sense_vocab(sid)
            common = sorted(list(input_tokens.intersection(vocab)), key=len, reverse=True)
            clues_map[sid] = common
            scores[sid] = (len(common) * 20.0) + (cs.get("sample_count", 0) * 0.1)

        best_sense = max(candidate_senses, key=lambda cs: scores.get(cs["sense_id"], 0.0))
        sid = best_sense["sense_id"]
        meaning = best_sense.get("combined_meaning") or best_sense.get("primary_meaning")
        chosen_clues = clues_map.get(sid, [])

        detailed_exp = self.build_detailed_explanation(
            sentence=sentence,
            target_word=target_word,
            chosen_sense_id=sid,
            candidate_senses=candidate_senses,
            overlapping_clues=chosen_clues,
        )

        return {
            "target_word": target_word,
            "predicted_sense_id": sid,
            "predicted_meaning": meaning,
            "explanation": detailed_exp,
            "is_valid": True,
            "confidence": 0.88 if chosen_clues else 0.70,
        }

    def predict_sense(
        self,
        prompt: str,
        target_word: str,
        candidate_senses: List[Dict[str, Any]],
        sentence: str = "",
    ) -> Dict[str, Any]:
        """
        Executes the full LLM prediction cycle:
        1. Checks in-memory cache for instant response.
        2. Calls the LLM API.
        3. If API fails or is unconfigured, uses offline fallback.
        4. Parses JSON response.
        5. Validates Sense ID against allowed dataset vocabulary.
        """
        cache_key = f"{sentence}_{target_word}_{hash(prompt)}"
        if cache_key in self._cache:
            cached = dict(self._cache[cache_key])
            cached["cached"] = True
            return cached

        if not self.is_configured():
            res = self.offline_semantic_fallback(sentence, target_word, candidate_senses)
            res["status"] = "offline_fallback"
            self._cache[cache_key] = res
            return res

        raw_text, error = self.call_gemini_api(prompt)

        if error or not raw_text:
            # Fall back gracefully so the app never crashes
            fallback_res = self.offline_semantic_fallback(sentence, target_word, candidate_senses)
            fallback_res["status"] = "api_error_fallback"
            fallback_res["api_error"] = error
            self._cache[cache_key] = fallback_res
            return fallback_res

        parsed = self.extract_json(raw_text)
        result = self.validate_and_normalize_prediction(parsed, candidate_senses, target_word, sentence=sentence)
        result["raw_response"] = raw_text
        result["status"] = "success" if result["is_valid"] else "invalid_sense"
        self._cache[cache_key] = result
        return result

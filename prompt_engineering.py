"""
Prompt Engineering Module for Assamese Word Sense Disambiguation (WSD).
Implements four distinct prompt engineering paradigms:
1. Zero-Shot Prompting
2. Few-Shot Prompting
3. Context-Based Prompting
4. Structured Prompting
"""

import json
from typing import Dict, List, Any, Optional
from dataset_handler import DatasetHandler
from config import (
    PROMPT_ZERO_SHOT,
    PROMPT_FEW_SHOT,
    PROMPT_CONTEXT_BASED,
    PROMPT_STRUCTURED,
)


class PromptGenerator:
    """
    Constructs research-grade prompts for Assamese Word Sense Disambiguation
    under different prompting paradigms.
    """

    def __init__(self, dataset_handler: Optional[DatasetHandler] = None):
        self.handler = dataset_handler or DatasetHandler()

    def format_candidate_senses(self, candidate_senses: List[Dict[str, Any]]) -> str:
        """
        Formats candidate senses into a clean bulleted list for prompt injection.
        """
        lines = []
        for sense in candidate_senses:
            sid = sense["sense_id"]
            meaning = sense.get("combined_meaning") or sense.get("primary_meaning")
            lines.append(f"- Sense ID: {sid} | Meaning: {meaning}")
        return "\n".join(lines)

    def generate_zero_shot_prompt(
        self,
        sentence: str,
        target_word: str,
        candidate_senses: List[Dict[str, Any]],
    ) -> str:
        """
        Technique 1: Zero-Shot Prompting
        Evaluates model's pre-trained semantic understanding without demonstrations.
        """
        senses_text = self.format_candidate_senses(candidate_senses)
        allowed_ids = [s["sense_id"] for s in candidate_senses]

        prompt = f"""You are an expert Assamese linguist and Natural Language Processing specialist.
Your task is Word Sense Disambiguation (WSD) for the Assamese language.

Given an Assamese sentence containing the ambiguous word "{target_word}", select the most appropriate Sense ID and meaning from the provided candidate senses.

### INPUT SENTENCE:
"{sentence}"

### TARGET AMBIGUOUS WORD:
"{target_word}"

### CANDIDATE SENSES:
{senses_text}

### INSTRUCTIONS:
1. Analyze the context of the sentence to determine the exact meaning of "{target_word}".
2. You MUST choose ONLY from the allowed Sense IDs: {allowed_ids}.
3. Provide a detailed, comprehensive explanation (3-4 sentences) covering:
   a) Specific lexical clues and keywords present in the sentence.
   b) Thematic/domain justification for selecting this Sense ID.
   c) Explicit contrast explaining why the alternative candidate senses do not fit.
4. Output your answer STRICTLY as a JSON object with the following schema:
{{
  "target_word": "{target_word}",
  "predicted_sense_id": "<ONE_OF_ALLOWED_SENSE_IDS>",
  "predicted_meaning": "<CORRESPONDING_ENGLISH_MEANING>",
  "explanation": "<DETAILED_EXPLANATION_CITING_CLUES_AND_CONTRAST>"
}}
Do NOT output any markdown commentary outside the JSON block.
"""
        return prompt.strip()

    def generate_few_shot_prompt(
        self,
        sentence: str,
        target_word: str,
        candidate_senses: List[Dict[str, Any]],
        num_examples_per_sense: int = 1,
    ) -> str:
        """
        Technique 2: Few-Shot Prompting
        Provides exemplar demonstrations strictly from the training dataset.
        """
        senses_text = self.format_candidate_senses(candidate_senses)
        allowed_ids = [s["sense_id"] for s in candidate_senses]

        # Retrieve verified examples from training data
        examples = self.handler.get_few_shot_examples(
            target_word=target_word,
            max_examples_per_sense=num_examples_per_sense,
            exclude_sentence=sentence,
        )

        examples_text = ""
        if examples:
            ex_lines = []
            for i, ex in enumerate(examples, 1):
                ex_lines.append(
                    f"Example {i}:\n"
                    f"Sentence: \"{ex['sentence']}\"\n"
                    f"Target Word: \"{ex['target_word']}\"\n"
                    f"Correct Sense ID: {ex['sense_id']}\n"
                    f"Meaning: {ex['meaning']}\n"
                )
            examples_text = "### REFERENCE TRAINING DEMONSTRATIONS:\n" + "\n".join(ex_lines) + "\n"

        prompt = f"""You are an expert Assamese linguist and Natural Language Processing specialist.
Your task is Word Sense Disambiguation (WSD) for the Assamese language using in-context exemplar learning.

{examples_text}
### TARGET SENTENCE TO EVALUATE:
"{sentence}"

### TARGET AMBIGUOUS WORD:
"{target_word}"

### CANDIDATE SENSES:
{senses_text}

### INSTRUCTIONS:
1. Compare the target sentence with the reference demonstrations above.
2. Select the correct Sense ID strictly from: {allowed_ids}.
3. Provide a detailed, comprehensive explanation (3-4 sentences) that compares the syntactic context with the training demonstrations, identifies key lexical indicators, and explains why competing senses do not fit.
4. Output your response STRICTLY as a JSON object:
{{
  "target_word": "{target_word}",
  "predicted_sense_id": "<ONE_OF_ALLOWED_SENSE_IDS>",
  "predicted_meaning": "<CORRESPONDING_ENGLISH_MEANING>",
  "explanation": "<DETAILED_EXPLANATION_WITH_DEMONSTRATION_COMPARISON>"
}}
Do NOT output any text outside the JSON block.
"""
        return prompt.strip()

    def generate_context_based_prompt(
        self,
        sentence: str,
        target_word: str,
        candidate_senses: List[Dict[str, Any]],
        additional_context: Optional[str] = None,
    ) -> str:
        """
        Technique 3: Context-Based Prompting
        Focuses on surrounding lexical cues, domain constraints, and optional user context.
        """
        senses_text = self.format_candidate_senses(candidate_senses)
        allowed_ids = [s["sense_id"] for s in candidate_senses]

        context_section = ""
        if additional_context and additional_context.strip():
            context_section = f"""### ADDITIONAL CONTEXT / DOMAIN INFO:
"{additional_context.strip()}"
"""

        prompt = f"""You are an advanced NLP system specializing in Assamese contextual semantics and lexical disambiguation.

### INPUT SENTENCE:
"{sentence}"

### TARGET AMBIGUOUS WORD:
"{target_word}"
{context_section}
### CANDIDATE SENSES:
{senses_text}

### CONTEXTUAL DISAMBIGUATION GUIDELINES:
1. Examine the syntactic head, verb modifiers, case markers, and collocated nouns in the sentence.
2. Identify semantic clues that rule out incorrect senses and support the intended sense.
3. You MUST select ONE Sense ID strictly from: {allowed_ids}.
4. Provide a thorough, multi-sentence explanation detailing:
   a) The exact contextual trigger words and collocations in the sentence.
   b) The specific semantic domain and grammatical function.
   c) Explicit justification for excluding alternative candidate senses.
5. Return your prediction as a JSON object:
{{
  "target_word": "{target_word}",
  "predicted_sense_id": "<ONE_OF_ALLOWED_SENSE_IDS>",
  "predicted_meaning": "<CORRESPONDING_ENGLISH_MEANING>",
  "explanation": "<DETAILED_CONTEXTUAL_AND_CONTRASTIVE_EXPLANATION>"
}}
"""
        return prompt.strip()

    def generate_structured_prompt(
        self,
        sentence: str,
        target_word: str,
        candidate_senses: List[Dict[str, Any]],
    ) -> str:
        """
        Technique 4: Structured Prompting
        Enforces a rigorous schema specification with explicit validation criteria.
        """
        senses_list = [
            {"sense_id": s["sense_id"], "meaning": s.get("combined_meaning") or s.get("primary_meaning")}
            for s in candidate_senses
        ]
        allowed_ids = [s["sense_id"] for s in candidate_senses]

        prompt = f"""=== ASSAMESE WSD SPECIFICATION SCHEMA ===
[TASK]
Identify the intended sense of the ambiguous Assamese word "{target_word}" in the given sentence.

[SENTENCE]
{sentence}

[TARGET_WORD]
{target_word}

[ALLOWED_SENSES]
{json.dumps(senses_list, ensure_ascii=False, indent=2)}

[OUTPUT_REQUIREMENTS]
- Return a valid JSON document conforming to this exact TypeScript interface:
  interface WSDResult {{
    target_word: string;
    predicted_sense_id: {json.dumps(allowed_ids)};
    predicted_meaning: string;
    explanation: string; // Detailed 3-4 sentence explanation identifying sentence keywords, semantic domain justification, and reasons for excluding competing senses.
  }}
- The predicted_sense_id MUST be one of {allowed_ids}.
- No additional Markdown headers, formatting, or conversational text.
"""
        return prompt.strip()

    def build_prompt(
        self,
        technique: str,
        sentence: str,
        target_word: str,
        candidate_senses: List[Dict[str, Any]],
        additional_context: Optional[str] = None,
    ) -> str:
        """
        Unified dispatch method for generating any prompting technique.
        """
        if technique == PROMPT_ZERO_SHOT:
            return self.generate_zero_shot_prompt(sentence, target_word, candidate_senses)
        elif technique == PROMPT_FEW_SHOT:
            return self.generate_few_shot_prompt(sentence, target_word, candidate_senses)
        elif technique == PROMPT_CONTEXT_BASED:
            return self.generate_context_based_prompt(sentence, target_word, candidate_senses, additional_context)
        elif technique == PROMPT_STRUCTURED:
            return self.generate_structured_prompt(sentence, target_word, candidate_senses)
        else:
            # Default to Zero-Shot
            return self.generate_zero_shot_prompt(sentence, target_word, candidate_senses)

    def generate_natural_query_prompt(
        self,
        user_query: str,
        known_vocabulary: Optional[List[str]] = None,
        candidate_senses: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Specialized Prompt Engineering for Natural-Language Query WSD.
        Accepts unstructured natural-language queries (e.g.
        "In the sentence 'গছৰ পাত সেউজীয়া', what is the meaning of পাত?")
        and instructs the LLM to:
        1. Extract the Assamese sentence and target ambiguous word.
        2. Analyze the surrounding context without hardcoding answers.
        3. Determine the intended meaning.
        4. Synthesize a clear explanation and assign a confidence score (e.g. '7/7').
        5. Signal clarification or insufficient context when appropriate.
        """
        candidate_text = ""
        if candidate_senses:
            candidate_text = (
                "\n### CANDIDATE SENSES FROM DATASET:\n"
                + self.format_candidate_senses(candidate_senses)
                + "\n"
            )

        vocab_hint = ""
        if known_vocabulary:
            vocab_hint = f"\n### REFERENCE POLYSEMOUS WORDS IN ASSAMESE CORPUS:\n{', '.join(known_vocabulary)}\n"

        prompt = f"""You are an expert computational linguist specializing in Assamese Word Sense Disambiguation (WSD) and Natural Language Processing.

The user has submitted an open natural-language query asking to disambiguate a word in an Assamese sentence.
Analyze the complete query, extract the sentence and ambiguous word, evaluate the surrounding context, and determine the exact intended meaning.
Do NOT hardcode answers; reason strictly from the contextual cues in the extracted sentence.

### USER QUERY:
"{user_query}"
{candidate_text}{vocab_hint}
### INSTRUCTIONS & OUTPUT REQUIREMENTS:
1. EXTRACTION:
   - Extract the complete Assamese sentence containing Unicode Assamese characters. Ensure proper punctuation (e.g., '।').
   - Identify the exact target ambiguous word being queried.
   - If the query does NOT contain an identifiable Assamese sentence or target word, set "status": "needs_clarification".

2. CONTEXTUAL REASONING:
   - Examine the surrounding semantic cues, modifiers, and verbs in the extracted sentence.
   - If the sentence lacks sufficient context to distinguish the sense conclusively, set "status": "insufficient_context" and "predicted_meaning": "Insufficient Context".
   - Otherwise, select or formulate the accurate English meaning of the ambiguous word in this context.

3. CONFIDENCE SCORE:
   - Provide a confidence score formatted as "X/7" (e.g., "7/7" for high certainty with strong context, "6/7" for moderate certainty, or "2/7" for insufficient context).

4. EXPLANATION:
   - Provide a clear, natural explanation explaining how the context indicates this meaning (e.g., "The context indicates that পাত refers to a leaf of a tree.").

5. OUTPUT FORMAT:
   Return STRICTLY a JSON object with this exact structure:
{{
  "status": "success",
  "extracted_sentence": "<EXTRACTED_ASSAMESE_SENTENCE>",
  "ambiguous_word": "<TARGET_AMBIGUOUS_WORD>",
  "predicted_meaning": "<PREDICTED_MEANING>",
  "explanation": "<CLEAR_CONTEXTUAL_EXPLANATION>",
  "confidence_score": "7/7",
  "error_message": null
}}

If clarification is required (missing Assamese text or ambiguous word):
{{
  "status": "needs_clarification",
  "extracted_sentence": null,
  "ambiguous_word": null,
  "predicted_meaning": null,
  "explanation": null,
  "confidence_score": "0/7",
  "error_message": "Could not identify an Assamese sentence or target word. Please clarify your query."
}}

If context is ambiguous or insufficient:
{{
  "status": "insufficient_context",
  "extracted_sentence": "<EXTRACTED_ASSAMESE_SENTENCE>",
  "ambiguous_word": "<TARGET_AMBIGUOUS_WORD>",
  "predicted_meaning": "Insufficient Context",
  "explanation": "The surrounding context is insufficient to determine the exact meaning conclusively.",
  "confidence_score": "2/7",
  "error_message": "Insufficient context to disambiguate."
}}

Return ONLY the raw JSON document. No markdown wrappers or extra commentary.
"""
        return prompt.strip()

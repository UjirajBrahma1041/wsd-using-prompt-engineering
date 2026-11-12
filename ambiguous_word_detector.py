"""
Ambiguous Word Detector Module for Assamese WSD.
Automatically detects known ambiguous words from Assamese input sentences without manual user selection.
Supports single and multiple ambiguous words, handles Assamese morphology, case markers, definers,
and verb inflections, while strictly avoiding false positives in unrelated words.
"""

import re
import unicodedata
from typing import List, Dict, Any, Set, Tuple, Optional
from dataset_handler import DatasetHandler

# Common Assamese inflectional suffixes, case markers (বিভক্তি),
# definiteness markers (নিৰ্দিষ্টতাবাচক প্ৰত্যয়), and plural markers (বহুবচন প্ৰত্যয়)
ASSAMESE_SUFFIXES = sorted([
    # Complex combined suffixes
    "বোৰলৈকে", "বিলাকলৈকে", "হঁতলৈকে",
    "বোৰৰে", "বোৰেও", "বোৰেৰে", "বোৰলৈ", "বোৰৰ", "বোৰত", "বোৰে", "বোৰ",
    "বিলাকৰে", "বিলাকলৈ", "বিলাকৰ", "বিলাকত", "বিলাকে", "বিলাক",
    "হঁতৰে", "হঁতলৈ", "হঁতৰ", "হঁতত", "হঁতক", "হঁত",
    "খনেৰে", "খনলৈ", "খনতো", "খনৰ", "খনত", "খনে", "খনো", "খন",
    "টোৱেৰে", "টোলৈকে", "টোলৈ", "টোৰ", "টোত", "টোৱে", "টোক", "টোও", "টো",
    "যোৰেৰে", "যোৰলৈ", "যোৰৰ", "যোৰত", "যোৰে", "যোৰ",
    "ডালৰ", "ডালত", "ডালেৰে", "ডালে", "ডাল",
    "জোপাৰ", "জোপাত", "জোপালৈ", "জোপাই", "জোপা",
    "গৰাকীৰ", "গৰাকীত", "গৰাকীক", "গৰাকী",
    # Single and standard case endings
    "লৈকে", "লৈ", "তে", "ৰে", "এৰে", "ৰ", "ত", "ক", "এ", "ই", "ও", "নে", "হে", "তু", "য়ে", "ৱে"
], key=len, reverse=True)

# Assamese vowel signs (matras) that attach to consonant endings
ASSAMESE_MATRAS = ["ে", "ৈ", "ো", "ৌ", "ী", "ি", "ু", "ূ", "ৃ"]

# Known verbal forms / inflections mapped to canonical target words
VERBAL_INFLECTIONS: Dict[str, Dict[str, str]] = {
    "পৰা": {
        "পৰিল": "verb_past",
        "পৰিছিল": "verb_past_perfect",
        "পৰিলে": "verb_conditional",
        "পৰি": "verb_participle",
        "পৰে": "verb_present",
        "পৰিব": "verb_future",
        "পৰাত": "verb_locative",
        "পৰাকৈ": "verb_adverbial",
        "পৰিছে": "verb_continuous",
        "পৰাটো": "verb_nominal",
    },
    "ধৰা": {
        "ধৰিলে": "verb_conditional",
        "ধৰিছিল": "verb_past_perfect",
        "ধৰি": "verb_participle",
        "ধৰে": "verb_present",
        "ধৰিব": "verb_future",
        "ধৰাকৈ": "verb_adverbial",
        "ধৰাত": "verb_locative",
        "ধৰিছে": "verb_continuous",
        "ধৰাটো": "verb_nominal",
    },
    "বোৱা": {
        "বয়": "verb_present_3p",
        "ব'বলৈ": "verb_infinitive_colloquial",
        "ববলৈ": "verb_infinitive",
        "ব'লে": "verb_conditional_colloquial",
        "বলে": "verb_conditional",
        "বৈছে": "verb_continuous",
        "বোৱাই": "verb_causative",
        "বোৱাৰ": "verb_genitive",
        "বোৱাত": "verb_locative",
        "বাই": "verb_participle",
        "বাইছে": "verb_continuous",
        "বয়নে": "verb_interrogative",
    },
    "ঢাল": {
        "ঢালিলে": "verb_conditional",
        "ঢালিছিল": "verb_past_perfect",
        "ঢালি": "verb_participle",
        "ঢালে": "verb_present",
        "ঢালিব": "verb_future",
        "ঢালিলেহে": "verb_conditional_emphasis",
        "ঢালিছে": "verb_continuous",
        "ঢালক": "verb_imperative",
    },
    "গুণ": {
        "গুণাগুণ": "compound_reduplication",
    },
    "কাণ": {
        "কাণফুলি": "compound_ornament",
    },
    "উত্তৰ": {
        "উত্তৰমুখী": "compound_direction",
        "উত্তৰপত্ৰ": "compound_noun",
        "উত্তৰপত্ৰত": "compound_locative",
    }
}

# Known non-target words that contain target words as substrings but are distinct roots (FALSE POSITIVE BLACKLIST)
# e.g., মানুহ is NOT মান; সমান is NOT মান; কিমান is NOT মান; আনাৰস is NOT ৰস
FALSE_POSITIVE_WORDS: Set[str] = {
    # Unrelated words containing মান
    "মানুহ", "মানুহে", "মানুহৰ", "মানুহত", "মানুহক", "মানুহজন", "মানুহজনী", "মানুহবোৰ", "মানুহবিলাক",
    "সমান", "সমানকৈ", "সমানভাৱে", "কিমান", "সিমান", "ইমান", "যিমান", "সন্মান", "অপমান",
    "বিমান", "ভাৰতৰত্ন", "প্ৰমাণ", "অনুপাত",
    # Unrelated words containing ৰস
    "আনাৰস", "আনাৰসৰ", "আনাৰসত", "ভাৰসাম্য",
    # Unrelated words containing ফল
    "সফল", "সফলতা", "সফলভাৱে", "বিফল", "বিফলতা", "প্ৰতিফল", "প্ৰতিফলন",
    # Unrelated words containing ভাগ
    "বিভাগ", "বিভাগৰ", "বিভাগত", "বিভাগে", "দুৰ্ভাগ্য", "সৌভাগ্য", "অংশগ্ৰহণ",
    # Unrelated words containing পাত
    "গছজোপা", "গছজোপাত", "মানুহজোপা", "আঁচনি",
    # Unrelated words containing ওপৰ
    "ওপৰঞ্চি",
    # Unrelated words containing আদি
    "আদিত্য", "আদিম", "আদৰ", "আদৰণি",
}


class AssameseAmbiguousWordDetector:
    """
    Detector for automatic identification of ambiguous Assamese words in sentences.
    Uses tokenization, morphological affix stripping, verbal inflection mapping,
    and false-positive avoidance.
    """

    def __init__(self, dataset_handler: Optional[DatasetHandler] = None):
        self.handler = dataset_handler or DatasetHandler()
        self.vocabulary = set(self.handler.get_target_words())
        self.token_pattern = re.compile(r"[\u0980-\u09FF\u0964\u0965']+")

    def get_vocabulary(self) -> List[str]:
        """Returns the sorted list of target words known in the current dataset."""
        return sorted(list(self.vocabulary))

    def is_known_ambiguous_word(self, word: str) -> bool:
        """Checks if a normalized word is in the ambiguous vocabulary."""
        norm = unicodedata.normalize("NFC", word.strip())
        return norm in self.vocabulary

    def detect_ambiguous_words(self, sentence: str) -> List[Dict[str, Any]]:
        """
        Scans an Assamese sentence and automatically detects all occurrences of ambiguous target words.
        Returns a list of structured detection records.
        """
        if not sentence or not sentence.strip():
            return []

        norm_sentence = unicodedata.normalize("NFC", sentence)
        matches: List[Dict[str, Any]] = []
        seen_targets: Set[str] = set()

        # Find word tokens in the sentence along with their start and end positions
        for token_match in re.finditer(r"[\u0980-\u09FF\u0964\u0965']+", norm_sentence):
            raw_token = token_match.group()
            # Clean trailing punctuation from token
            clean_token = raw_token.strip("।॥,.-!?\"' ")
            if not clean_token:
                continue

            start_char, end_char = token_match.span()

            # Check blacklist for known false-positive words
            if clean_token in FALSE_POSITIVE_WORDS:
                continue

            detected_target: Optional[str] = None
            detection_type: str = "exact"

            # 1. Exact Match against target vocabulary
            if clean_token in self.vocabulary:
                detected_target = clean_token
                detection_type = "exact"

            # 2. Check verbal / compound inflections mapping
            if not detected_target:
                for root, inflections in VERBAL_INFLECTIONS.items():
                    if root in self.vocabulary and clean_token in inflections:
                        detected_target = root
                        detection_type = f"inflected_{inflections[clean_token]}"
                        break

            # 3. Suffix Stripping (case markers, definers, plural markers)
            if not detected_target:
                for suf in ASSAMESE_SUFFIXES:
                    if clean_token.endswith(suf) and len(clean_token) > len(suf):
                        stem = clean_token[:-len(suf)]
                        if stem in self.vocabulary and stem not in FALSE_POSITIVE_WORDS:
                            detected_target = stem
                            detection_type = f"suffix_stripped (-{suf})"
                            break
                        # Handle matra attached to stem before suffix
                        for m in ASSAMESE_MATRAS:
                            if stem.endswith(m):
                                sub_stem = stem[:-len(m)]
                                if sub_stem in self.vocabulary and sub_stem not in FALSE_POSITIVE_WORDS:
                                    detected_target = sub_stem
                                    detection_type = f"matra_suffix_stripped (-{m}{suf})"
                                    break
                        if detected_target:
                            break

            # 4. Standalone Matra attached directly to root (e.g. কাণে, মুখে, ৰসে, ঢালে)
            if not detected_target:
                for m in ASSAMESE_MATRAS:
                    if clean_token.endswith(m):
                        stem = clean_token[:-len(m)]
                        if stem in self.vocabulary and stem not in FALSE_POSITIVE_WORDS:
                            detected_target = stem
                            detection_type = f"matra_inflected (-{m})"
                            break

            # 5. Compound / Root-prefix matching
            if not detected_target:
                for tw in self.vocabulary:
                    if clean_token.startswith(tw) and len(clean_token) > len(tw):
                        remainder = clean_token[len(tw):]
                        
                        # Case A: Known compounds/reduplications (e.g. উত্তৰমুখী, গুণাগুণ, কাণফুলি)
                        if clean_token in VERBAL_INFLECTIONS.get(tw, {}):
                            detected_target = tw
                            detection_type = "known_compound"
                            break

                        # Case B: Remainder is strictly a valid suffix from ASSAMESE_SUFFIXES
                        if remainder in ASSAMESE_SUFFIXES:
                            detected_target = tw
                            detection_type = f"suffix_stripped (-{remainder})"
                            break

                        # Case C: Remainder starts with a matra followed strictly by a valid suffix or nothing
                        for m in ASSAMESE_MATRAS:
                            if remainder.startswith(m):
                                after_matra = remainder[len(m):]
                                if after_matra == "" or after_matra in ASSAMESE_SUFFIXES:
                                    detected_target = tw
                                    detection_type = f"matra_suffix_stripped (-{m}{after_matra})"
                                    break
                        if detected_target:
                            break

            # Record detection if a valid ambiguous target word was found
            if detected_target and detected_target in self.vocabulary:
                candidate_senses = self.handler.get_candidate_senses(detected_target)
                
                # Check if this exact target word has already been detected in this sentence
                if detected_target not in seen_targets:
                    seen_targets.add(detected_target)
                    matches.append({
                        "target_word": detected_target,
                        "matched_token": clean_token,
                        "start_char": start_char,
                        "end_char": end_char,
                        "detection_type": detection_type,
                        "candidate_senses": candidate_senses,
                        "num_candidate_senses": len(candidate_senses)
                    })

        return matches

"""
Dataset Handler Module for Assamese Word Sense Disambiguation (WSD).
Responsible for loading, validating, analyzing, splitting, and managing candidate senses
from the original Assamese WSD dataset without modifying or overwriting original files.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import unicodedata
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from config import (
    ORIGINAL_DATASET_PATH,
    TRAIN_DATASET_PATH,
    TEST_DATASET_PATH,
    RANDOM_SEED,
)

REQUIRED_COLUMNS = ["ID", "Sentence", "Target_Word", "Sense_ID", "English_Meaning"]


class DatasetHandler:
    def __init__(self, dataset_path: Optional[Path] = None):
        self.dataset_path = dataset_path or ORIGINAL_DATASET_PATH
        self.raw_df: Optional[pd.DataFrame] = None
        self.train_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        self.senses_map: Dict[str, Dict[str, Dict[str, Any]]] = {}
        
        # Load and initialize
        self.load_dataset()

    def load_dataset(self) -> pd.DataFrame:
        """
        Loads the original dataset CSV, validates columns, and applies NFC Unicode normalization.
        """
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Original dataset not found at: {self.dataset_path}")

        df = pd.read_csv(self.dataset_path, encoding="utf-8-sig")
        
        # Validate columns
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Dataset is missing required columns: {missing_cols}")

        # Normalize text to NFC Unicode to prevent combining character mismatches
        df["Sentence"] = df["Sentence"].astype(str).apply(lambda s: unicodedata.normalize("NFC", s.strip()))
        df["Target_Word"] = df["Target_Word"].astype(str).apply(lambda s: unicodedata.normalize("NFC", s.strip()))
        df["Sense_ID"] = df["Sense_ID"].astype(str).apply(lambda s: s.strip().upper())
        df["English_Meaning"] = df["English_Meaning"].astype(str).apply(lambda s: s.strip())

        self.raw_df = df
        self._build_senses_map()
        return self.raw_df

    def _build_senses_map(self) -> None:
        """
        Builds a comprehensive dictionary mapping each Target_Word to its candidate Sense_IDs,
        canonical meanings, and all observed synonym variants from the CSV.
        """
        if self.raw_df is None:
            return

        self.senses_map = {}
        grouped = self.raw_df.groupby(["Target_Word", "Sense_ID"])
        
        for (target_word, sense_id), group in grouped:
            if target_word not in self.senses_map:
                self.senses_map[target_word] = {}

            # Meanings observed in the dataset for this sense
            meanings = group["English_Meaning"].unique().tolist()
            # Primary meaning is the most frequent or canonical representation
            primary_meaning = group["English_Meaning"].mode().iloc[0] if not group.empty else meanings[0]
            
            # Formulate a descriptive combined label if multiple synonyms exist
            combined_meaning = " / ".join(meanings) if len(meanings) > 1 else primary_meaning

            self.senses_map[target_word][sense_id] = {
                "sense_id": sense_id,
                "primary_meaning": primary_meaning,
                "combined_meaning": combined_meaning,
                "all_meanings": meanings,
                "sample_count": len(group)
            }

    def get_candidate_senses(self, target_word: str) -> List[Dict[str, Any]]:
        """
        Retrieves candidate senses for a given target word.
        Returns a list of sense objects with sense_id, primary_meaning, and combined_meaning.
        """
        normalized_word = unicodedata.normalize("NFC", target_word.strip())
        word_senses = self.senses_map.get(normalized_word, {})
        return list(word_senses.values())

    def get_target_words(self) -> List[str]:
        """Returns sorted list of all unique target words present in dataset."""
        if self.raw_df is None:
            self.load_dataset()
        return sorted(self.raw_df["Target_Word"].unique().tolist())

    def get_all_sense_ids(self) -> List[str]:
        """Returns sorted list of all unique Sense IDs present in dataset."""
        if self.raw_df is None:
            self.load_dataset()
        return sorted(self.raw_df["Sense_ID"].unique().tolist())

    def get_dataset_statistics(self) -> Dict[str, Any]:
        """
        Calculates and returns thorough dataset statistics.
        """
        if self.raw_df is None:
            self.load_dataset()

        df = self.raw_df
        word_counts = df["Target_Word"].value_counts().to_dict()
        sense_counts = df["Sense_ID"].value_counts().to_dict()
        
        # Calculate senses per word count
        senses_per_word = {word: len(senses) for word, senses in self.senses_map.items()}
        
        # Missing values and duplicates
        missing_values = df.isnull().sum().to_dict()
        duplicates_count = df.duplicated(subset=["Sentence", "Target_Word", "Sense_ID"]).sum()

        return {
            "total_sentences": len(df),
            "unique_target_words_count": df["Target_Word"].nunique(),
            "unique_senses_count": df["Sense_ID"].nunique(),
            "target_words": self.get_target_words(),
            "word_counts": word_counts,
            "sense_counts": sense_counts,
            "senses_per_word": senses_per_word,
            "missing_values": missing_values,
            "duplicate_records": int(duplicates_count),
            "senses_detail": self.senses_map
        }

    def prepare_train_test_split(
        self,
        test_size: float = 0.2,
        random_state: int = RANDOM_SEED,
        save: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits dataset into 80% train and 20% test sets using stratified sampling on
        (Target_Word + Sense_ID) to preserve class distributions and prevent data leakage.
        Saves to data/train.csv and data/test.csv without modifying the original dataset.
        """
        if self.raw_df is None:
            self.load_dataset()

        df = self.raw_df.copy()
        # Stratification key combining target word and sense ID
        df["strat_key"] = df["Target_Word"] + "_" + df["Sense_ID"]
        
        # Check rare classes
        class_counts = df["strat_key"].value_counts()
        rare_classes = class_counts[class_counts < 2].index.tolist()

        if rare_classes:
            # If any class has only 1 sample, separate it into train to avoid stratify failure
            train_rare = df[df["strat_key"].isin(rare_classes)]
            df_strat = df[~df["strat_key"].isin(rare_classes)]
            train_sub, test_sub = train_test_split(
                df_strat,
                test_size=test_size,
                random_state=random_state,
                stratify=df_strat["strat_key"]
            )
            train_df = pd.concat([train_sub, train_rare]).sample(frac=1, random_state=random_state).reset_index(drop=True)
            test_df = test_sub.sample(frac=1, random_state=random_state).reset_index(drop=True)
        else:
            train_df, test_df = train_test_split(
                df,
                test_size=test_size,
                random_state=random_state,
                stratify=df["strat_key"]
            )
            train_df = train_df.reset_index(drop=True)
            test_df = test_df.reset_index(drop=True)

        # Drop temporary stratification column
        train_df = train_df.drop(columns=["strat_key"])
        test_df = test_df.drop(columns=["strat_key"])

        self.train_df = train_df
        self.test_df = test_df

        if save:
            TRAIN_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
            train_df.to_csv(TRAIN_DATASET_PATH, index=False, encoding="utf-8-sig")
            test_df.to_csv(TEST_DATASET_PATH, index=False, encoding="utf-8-sig")

        return train_df, test_df

    def load_train_test(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads existing train.csv and test.csv, or generates them if not present.
        """
        if os.path.exists(TRAIN_DATASET_PATH) and os.path.exists(TEST_DATASET_PATH):
            self.train_df = pd.read_csv(TRAIN_DATASET_PATH, encoding="utf-8-sig")
            self.test_df = pd.read_csv(TEST_DATASET_PATH, encoding="utf-8-sig")
        else:
            self.prepare_train_test_split(save=True)
            
        return self.train_df, self.test_df

    def get_few_shot_examples(
        self,
        target_word: str,
        max_examples_per_sense: int = 1,
        exclude_sentence: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Retrieves clean few-shot training examples strictly from the training dataset.
        Guarantees that test sentences are never leaked.
        Selects balanced representative sentences for each available sense of the target word.
        """
        if self.train_df is None:
            self.load_train_test()

        normalized_word = unicodedata.normalize("NFC", target_word.strip())
        word_train = self.train_df[self.train_df["Target_Word"] == normalized_word]
        
        if word_train.empty:
            return []

        examples = []
        for sense_id, group in word_train.groupby("Sense_ID"):
            candidates = group
            if exclude_sentence:
                norm_ex = unicodedata.normalize("NFC", exclude_sentence.strip())
                candidates = candidates[candidates["Sentence"] != norm_ex]

            if not candidates.empty:
                sampled = candidates.head(max_examples_per_sense)
                for _, row in sampled.iterrows():
                    examples.append({
                        "sentence": row["Sentence"],
                        "target_word": row["Target_Word"],
                        "sense_id": row["Sense_ID"],
                        "meaning": row["English_Meaning"],
                    })

        return examples

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import urllib.request
import urllib.error

import pandas as pd
import numpy as np
from tqdm import tqdm
from datasets import Dataset, DatasetDict, load_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# MTG Jamendo dataset URLs
MTG_JAMENDO_URLS = {
    "full": "https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/master/data",
    "sample": "https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/master/data",
}

# Tag categorization
INSTRUMENT_KEYWORDS = {
    "acoustic", "electric", "synth", "digital", "string", "wind", "brass",
    "percussion", "drum", "guitar", "bass", "piano", "keyboard", "violin",
    "cello", "flute", "saxophone", "trumpet", "horn", "organ", "harp",
    "accordion", "banjo", "mandolin", "ukulele", "sitar", "tabla", "didgeridoo",
    "harmonica", "clarinet", "oboe", "xylophone", "marimba", "timpani",
    "timpani", "viola", "conga", "bongo", "tambourine", "triangle",
    "bell", "chime", "cymbal", "gong", "hi-hat", "kick", "snare", "tom",
    "vocal", "voice", "sung", "singing", "choir", "chant", "rap", "spoken",
    "theremin", "synthesizer", "electronic", "eurorack", "modular",
}

MOOD_KEYWORDS = {
    "happy", "sad", "melancholic", "cheerful", "joyful", "upbeat",
    "happy", "energetic", "calm", "relaxing", "chill", "peaceful",
    "serene", "tranquil", "dreamy", "ethereal", "mystical", "mysterious",
    "dark", "ominous", "eerie", "spooky", "scary", "intense",
    "passionate", "romantic", "love", "emotional", "sentimental",
    "nostalgic", "bittersweet", "melancholy", "hopeful", "inspiring",
    "uplifting", "motivating", "energizing", "funky", "groovy",
    "playful", "fun", "quirky", "whimsical", "silly", "anxious",
    "tense", "suspenseful", "meditative", "spiritual", "sacred",
    "festive", "celebratory", "triumphant", "powerful", "heavy",
}

GENRE_KEYWORDS = {
    "rock", "pop", "jazz", "classical", "electronic", "house", "techno",
    "trance", "dnb", "drum and bass", "dubstep", "garage", "grime",
    "hip hop", "hip-hop", "rap", "r&b", "soul", "funk", "disco",
    "reggae", "dub", "dancehall", "country", "folk", "indie",
    "alternative", "punk", "metal", "hard rock", "progressive",
    "psychedelic", "blues", "gospel", "ambient", "experimental",
    "avant", "noise", "ost", "soundtrack", "world", "latin",
    "african", "asian", "indian", "middle eastern", "arabic",
    "bollywood", "flamenco", "bossa nova", "samba", "salsa",
    "trap", "lo-fi", "synthwave", "vaporwave", "chillwave",
    "idm", "glitch", "breakcore", "jungle", "liquid", "dnb",
    "hardstyle", "industrial", "darkwave", "gothic", "post",
    "shoegaze", "grunge", "emo", "scene", "mathrock", "prog",
}

TEMPO_KEYWORDS = {
    "fast", "slow", "moderate", "medium", "quick", "rapid",
    "steady", "consistent", "variable", "upbeat", "uptempo",
    "downtempo", "mid-tempo", "midtempo", "bpm", "tempo",
    "accelerating", "decelerating", "paced", "groove", "beat",
}


class TagCategorizer:
    """Categorizes tags into instrument, mood, genre, and tempo."""

    def __init__(
        self,
        instrument_keywords: Set[str] = None,
        mood_keywords: Set[str] = None,
        genre_keywords: Set[str] = None,
        tempo_keywords: Set[str] = None,
    ):
        self.instrument_keywords = set(instrument_keywords or INSTRUMENT_KEYWORDS)
        self.mood_keywords = set(mood_keywords or MOOD_KEYWORDS)
        self.genre_keywords = set(genre_keywords or GENRE_KEYWORDS)
        self.tempo_keywords = set(tempo_keywords or TEMPO_KEYWORDS)

    def categorize_tags(
        self, tags: List[str]
    ) -> Dict[str, List[str]]:
        """
        Categorize a list of tags into instrument, mood, genre, and tempo.

        Args:
            tags: List of tag strings

        Returns:
            Dictionary with keys 'instrument', 'mood', 'genre', 'tempo'
            mapping to lists of relevant tags
        """
        categories = {
            "instrument": [],
            "mood": [],
            "genre": [],
            "tempo": [],
        }

        for tag in tags:
            tag_lower = tag.lower().strip()

            # Check each category
            if any(keyword in tag_lower for keyword in self.instrument_keywords):
                categories["instrument"].append(tag)
            if any(keyword in tag_lower for keyword in self.mood_keywords):
                categories["mood"].append(tag)
            if any(keyword in tag_lower for keyword in self.genre_keywords):
                categories["genre"].append(tag)
            if any(keyword in tag_lower for keyword in self.tempo_keywords):
                categories["tempo"].append(tag)

        # Remove duplicates while preserving order
        for key in categories:
            categories[key] = list(dict.fromkeys(categories[key]))

        return categories


class MTGJamendoDatasetCreator:
    """Creates a caption-generation-ready dataset from MTG Jamendo data."""

    def __init__(
        self,
        output_dir: Path = Path("data/mtg_jamendo"),
        train_split: float = 0.8,
        val_split: float = 0.1,
        random_state: int = 42,
        logger: Optional[logging.Logger] = None,
    ):
        self.output_dir = Path(output_dir)
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = 1.0 - train_split - val_split
        self.random_state = random_state
        self.tag_categorizer = TagCategorizer()
        self.logger = logger or logging.getLogger(__name__)

        np.random.seed(random_state)

    def download_jamendo_metadata(
        self, split: str = "full"
    ) -> pd.DataFrame:
        """
        Download MTG Jamendo dataset metadata.

        Args:
            split: Dataset split to download ('full' or 'sample')

        Returns:
            DataFrame with metadata
        """
        self.logger.info(f"Downloading MTG Jamendo dataset ({split} split)...")

        # Try to load from a local JSON file or create a sample dataset
        # The MTG Jamendo dataset structure - you may need to adjust based on
        # your actual data source
        metadata_list = []

        try:
            # Attempt to download from GitHub
            base_url = MTG_JAMENDO_URLS[split]
            
            # Common files in MTG Jamendo dataset
            for year in range(2016, 2019):  # Example years
                for split_type in ["train", "val", "test"]:
                    url = f"{base_url}/{year}/{split_type}_tags.json"
                    try:
                        self.logger.info(f"Trying to download: {url}")
                        with urllib.request.urlopen(url, timeout=10) as response:
                            data = json.loads(response.read())
                            metadata_list.extend(data)
                    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
                        self.logger.debug(f"Could not download {url}: {e}")
                        continue

            if not metadata_list:
                self.logger.warning(
                    "Could not download from GitHub. "
                    "Please provide a local dataset or configure data source."
                )
                return self._create_sample_dataset()

        except Exception as e:
            self.logger.error(f"Error downloading dataset: {e}")
            return self._create_sample_dataset()

        df = pd.DataFrame(metadata_list)
        self.logger.info(f"Downloaded {len(df)} samples from MTG Jamendo dataset")

        return df

    def _create_sample_dataset(self) -> pd.DataFrame:
        """Create a sample dataset for demonstration."""
        self.logger.warning("Creating sample dataset for demonstration.")

        sample_data = [
            {
                "track_id": f"track_{i:05d}",
                "tags": self._get_sample_tags(),
                "description": f"Sample music track {i}",
            }
            for i in range(100)
        ]

        return pd.DataFrame(sample_data)

    @staticmethod
    def _get_sample_tags() -> List[str]:
        """Get sample tags for demonstration."""
        instruments = ["acoustic guitar", "drums", "piano", "violin", "bass"]
        moods = ["happy", "energetic", "romantic", "peaceful", "melancholic"]
        genres = ["folk", "jazz", "pop", "classical", "indie rock"]
        tempos = ["slow tempo", "moderate tempo", "fast tempo", "uptempo"]

        return (
            np.random.choice(instruments, np.random.randint(1, 3), replace=False).tolist()
            + np.random.choice(moods, np.random.randint(1, 2), replace=False).tolist()
            + np.random.choice(genres, np.random.randint(1, 2), replace=False).tolist()
            + np.random.choice(tempos, np.random.randint(1, 2), replace=False).tolist()
        )

    def process_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw dataset and categorize tags.

        Args:
            df: Raw dataset DataFrame

        Returns:
            Processed DataFrame with categorized tags
        """
        self.logger.info("Processing and categorizing tags...")

        # Ensure we have the necessary columns
        if "tags" not in df.columns:
            self.logger.warning("'tags' column not found in dataset")
            df["tags"] = df.get("aspects", [])

        if "description" not in df.columns and "caption" not in df.columns:
            self.logger.warning("No description/caption column found")
            df["description"] = "No description available"
        else:
            df["description"] = df.get("caption", df.get("description", ""))

        processed_rows = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing tags"):
            tags = row.get("tags", [])

            # Handle different tag formats
            if isinstance(tags, str):
                if tags.startswith("["):
                    # JSON-like format
                    try:
                        tags = json.loads(tags.replace("'", '"'))
                    except (json.JSONDecodeError, ValueError):
                        tags = tags.strip("[]").split(", ")
                else:
                    tags = [t.strip() for t in tags.split(",")]

            elif not isinstance(tags, list):
                tags = []

            if not tags:
                continue

            # Categorize tags
            categorized = self.tag_categorizer.categorize_tags(tags)

            # Skip if missing any category
            if not all(
                categorized.get(cat) for cat in ["instrument", "mood", "genre", "tempo"]
            ):
                continue

            processed_rows.append(
                {
                    "description": row.get("description", ""),
                    "all_tags": ", ".join(tags),
                    "instrument_tags": ", ".join(categorized["instrument"]),
                    "mood_tags": ", ".join(categorized["mood"]),
                    "genre_tags": ", ".join(categorized["genre"]),
                    "tempo_tags": ", ".join(categorized["tempo"]),
                }
            )

        processed_df = pd.DataFrame(processed_rows)
        self.logger.info(
            f"Processed {len(processed_df)} samples with all tag categories"
        )

        return processed_df

    def create_splits(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create train/validation/test splits.

        Args:
            df: DataFrame to split

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        self.logger.info(
            f"Creating splits: train={self.train_split:.0%}, "
            f"val={self.val_split:.0%}, test={self.test_split:.0%}"
        )

        # Shuffle
        df = df.sample(frac=1.0, random_state=self.random_state).reset_index(drop=True)

        # Train split
        train_idx = int(len(df) * self.train_split)
        df_train = df[:train_idx]

        # Validation and test splits from remaining
        remaining = df[train_idx:]
        val_idx = int(len(remaining) * (self.val_split / (self.val_split + self.test_split)))
        df_val = remaining[:val_idx]
        df_test = remaining[val_idx:]

        return df_train, df_val, df_test

    def save_datasets(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        df_test: pd.DataFrame,
    ) -> None:
        """
        Save datasets to CSV files.

        Args:
            df_train: Training dataset
            df_val: Validation dataset
            df_test: Test dataset
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Rename description to caption for compatibility
        for df in [df_train, df_val, df_test]:
            if "description" in df.columns:
                df.rename(columns={"description": "caption"}, inplace=True)

        train_path = self.output_dir / "train.csv"
        val_path = self.output_dir / "validation.csv"
        test_path = self.output_dir / "test.csv"

        df_train.to_csv(train_path, index=False)
        df_val.to_csv(val_path, index=False)
        df_test.to_csv(test_path, index=False)

        self.logger.info(f"Training dataset: {len(df_train)} samples → {train_path}")
        self.logger.info(f"Validation dataset: {len(df_val)} samples → {val_path}")
        self.logger.info(f"Test dataset: {len(df_test)} samples → {test_path}")

        # Log statistics
        self.logger.info("\n=== Dataset Statistics ===")
        for split_name, df in [("Train", df_train), ("Validation", df_val), ("Test", df_test)]:
            self.logger.info(f"\n{split_name} Split ({len(df)} samples):")
            for tag_type in ["instrument", "mood", "genre", "tempo"]:
                col = f"{tag_type}_tags"
                avg_tags = df[col].apply(lambda x: len(x.split(", "))).mean()
                self.logger.info(f"  {tag_type:12} - avg tags: {avg_tags:.2f}")

    def create_dataset(self, split: str = "full") -> None:
        """
        Main pipeline to create the complete dataset.

        Args:
            split: Dataset split to download ('full' or 'sample')
        """
        self.logger.info("Starting MTG Jamendo dataset creation pipeline...")

        # Download data
        df_raw = self.download_jamendo_metadata(split=split)

        if df_raw.empty:
            self.logger.error("No data available. Please check data source configuration.")
            return

        # Process and categorize
        df_processed = self.process_dataset(df_raw)

        if df_processed.empty:
            self.logger.error("No samples with all tag categories found.")
            return

        # Create splits
        df_train, df_val, df_test = self.create_splits(df_processed)

        # Save
        self.save_datasets(df_train, df_val, df_test)

        self.logger.info("Dataset creation complete!")

    def push_to_hub(
        self,
        repo_name: str,
        private: bool = True,
    ) -> None:
        """
        Push the created dataset to HuggingFace Hub.

        Args:
            repo_name: Name of the HuggingFace Hub repository
            private: Whether the repository should be private
        """

        self.logger.info(f"Pushing dataset to HuggingFace Hub: {repo_name}")

        # Load datasets
        dataset = load_dataset("csv", data_files={
            "train": str(self.output_dir / "train.csv"),
            "validation": str(self.output_dir / "validation.csv"),
            "test": str(self.output_dir / "test.csv"),
        })

        dataset.push_to_hub(repo_name, private=private)
        self.logger.info("Dataset pushed to HuggingFace Hub successfully.")

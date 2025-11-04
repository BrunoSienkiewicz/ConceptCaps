from __future__ import annotations

import os

from typing import Any, Dict, List, Tuple
from pathlib import Path

from datasets import DatasetDict, load_dataset
from omegaconf import DictConfig


def _extract_tags(song_tags, concept_tags):
    res = []
    for c_tag in concept_tags:
        for s_tag in song_tags:
            if c_tag in s_tag:
                res.append(s_tag)
    return list(set(res))

def create_datasets(log, data_cfg: DictConfig, data_dir: Path) -> None:
    ds = load_dataset(data_cfg.dataset_name)
    df = ds['train'].to_pandas()

    df['aspect_list_transformed'] = df['aspect_list'].apply(lambda x: x.strip("[]").replace("'", ""))
    df['aspect_list_transformed'] = df['aspect_list_transformed'].apply(lambda x: x.split(', '))
    
    # TODO: These tags can be moved to a config file later
    tempo_tags = set([
        "fast tempo", "medium tempo", "slow tempo", "moderate tempo", "uptempo",
        "medium fast tempo", "slower tempo", "medium to uptempo", "mid-tempo",
        "quick tempo", "accelerated tempo", "steady tempo", "rapid tempo",
        "slow music", "very fast tempo", "slow to medium tempo", "medium-to-high pitch singing",
        "steady drumming rhythm", "dance rhythm", "various tempos", "tempo changes",
        "fast paced", "slow song", "mid tempo", "steady beat", "pulsating beats",
        "groovy rhythm", "4 on the floor kick pattern", "normal tempo", "fast beat"
    ])

    genre_tags = set([
        "rock", "pop", "jazz", "classical", "folk", "blues", "hip hop", "reggae",
        "metal", "country", "r&b", "edm", "trance", "techno", "dance music",
        "electronic dance music", "gospel", "ambient", "soul", "funk",
        "alternative rock", "ballad", "hip-hop", "techno pop", "world music",
        "disco", "trap", "punk rock", "latin pop", "house", "bluegrass",
        "indie rock", "new age", "grunge", "industrial", "dubstep",
        "carnatic music", "bossa nova", "baroque music", "surf rock",
        "ska", "lo-fi", "symphonic", "orchestral", "fusion music", "raga",
        "bollywood music", "afrobeat", "folk song", "christian rock", "soundtrack"
    ])

    mood_tags = set([
        "emotional", "passionate", "happy", "melancholic", "relaxing", "calming",
        "upbeat", "exciting", "mellow", "sentimental", "soothing", "joyful",
        "intense", "peaceful", "dreamy", "romantic mood", "ominous", "suspenseful",
        "haunting", "energetic", "chill", "cheerful", "nostalgic", "fun",
        "cool", "ethereal", "sad", "spooky", "hopeful", "playful",
        "mystical", "dark", "solemn", "festive", "inspirational", "sentimental",
        "powerful", "serene", "mysterious", "emphatic", "tranquil", "passionate singing",
        "ominous music", "romantic", "meditative", "joyous", "heartfelt", "uplifting",
        "enthusiastic", "melancholy", "emotional voice", "soothing melody", "heavenly", 
        "fearful", "vibrant", "soulful", "excited", "energetic drums", "charming"
    ])

    instrument_tags = set([
        "piano", "drums", "guitar", "bass guitar", "electric guitar", "acoustic guitar",
        "flute", "violin", "cello", "trumpet", "saxophone", "tambourine",
        "synth", "harmonica", "organ", "harp", "clarinet", "string section",
        "percussion", "banjo", "trombone", "didgeridoo", "mandolin", "tabla",
        "ukulele", "accordion", "xylophone", "viola", "timpani", "congas",
        "bongo", "triangle", "oboe", "bagpipes", "steel drums", "marimba",
        "dj mixer", "drum machine", "brass section", "horn", "sitar",
        "strings", "keyboard", "double bass", "synth bass", "guitar solo",
        "electric piano", "acoustic piano", "woodwind", "cymbals", "bells",
        "vibraphone", "hand claps", "snare", "hi-hat", "kick drum", 
        "conga", "tabla percussion", "theremin", "church organ", "trumpets",
        "bass drum", "djembe", "steel guitar", "harpsichord", "choir"
    ])

    concepts = {
        "tempo": tempo_tags,
        "genre": genre_tags,
        "mood": mood_tags,
        "instrument": instrument_tags
    }

    for concept, tags in concepts.items():
        df[concept + '_tags'] = df['aspect_list_transformed'].apply(
            lambda x: _extract_tags(x, tags)
        )

    df = df[["caption", "aspect_list_transformed", "tempo_tags", "genre_tags", "mood_tags", "instrument_tags"]]
    df = df[(df['tempo_tags'].map(len) > 0) & 
                        (df['genre_tags'].map(len) > 0) & 
                        (df['mood_tags'].map(len) > 0) & 
                        (df['instrument_tags'].map(len) > 0)] 
    df["aspect_list"] = df["aspect_list_transformed"].apply(lambda x: ', '.join(x))
    df["tempo_tags"] = df["tempo_tags"].apply(lambda x: ', '.join(x))
    df["genre_tags"] = df["genre_tags"].apply(lambda x: ', '.join(x))
    df["mood_tags"] = df["mood_tags"].apply(lambda x: ', '.join(x))
    df["instrument_tags"] = df["instrument_tags"].apply(lambda x: ', '.join(x))
    df = df[["caption", "aspect_list", "tempo_tags", "genre_tags", "mood_tags", "instrument_tags"]]

    os.makedirs(data_cfg.get("train_file").rsplit('/', 1)[0], exist_ok=True)

    df_train = df.sample(frac=0.8, random_state=42)
    df_train.to_csv(data_cfg.get("train_file"), index=False)
    log.info(f"Training dataset created with {len(df_train)} samples.")
    df_temp = df.drop(df_train.index)
    df_validation = df_temp.sample(frac=0.5, random_state=42)
    df_validation.to_csv(data_cfg.get("validation_file"), index=False)
    log.info(f"Validation dataset created with {len(df_validation)} samples.")
    df_test = df_temp.drop(df_validation.index)
    df_test.to_csv(data_cfg.get("test_file"), index=False)
    log.info(f"Test dataset created with {len(df_test)} samples.")

    log.info(f"Saved datasets to {data_dir}")

def _format_prompt(prompt_cfg: DictConfig, aspects: Any, reference_caption: str) -> str:
    user_prompt = prompt_cfg["user_prompt_template"].format(tags=aspects)
    return prompt_cfg["template"].format(
        system_prompt=prompt_cfg["system_prompt"],
        user_prompt=user_prompt,
        assistant_response=reference_caption,
    ).strip()


def prepare_datasets(data_cfg, raw_dataset: DatasetDict) -> DatasetDict:
    prompt_cfg = data_cfg.prompt
    text_column = data_cfg.text_column
    remove_columns = data_cfg.remove_columns
    if remove_columns is None:
        remove_columns = raw_dataset["train"].column_names

    def _transform_row(row: Dict[str, Any]) -> Dict[str, str]:
        formatted = _format_prompt(
            prompt_cfg,
            row[data_cfg.aspect_column],
            row[data_cfg.caption_column],
        )
        return {text_column: formatted}

    processed_dataset = raw_dataset.map(
        _transform_row,
        remove_columns=remove_columns,
    )

    return processed_dataset

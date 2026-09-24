"""Training and export script for MindCare Bi-LSTM Text Classification Model.

Constructs a balanced training dataset from existing project CSVs:
- dataset_chat/ (Addiction, Anxiety, Depression, Eating Disorder, Suicide)
- daily_chat/ (Neutral, Greeting)
- worng_context/ (OOD)

Trains a Bidirectional LSTM neural network using TensorFlow/Keras and exports:
- models/model_final.h5
- models/tokenizer.pickle
- models/model_metadata.json
"""

import json
import os
import pickle
import random
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

# Adjust path to import from config and preprocessing
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config.config import (
    CLASS_LABELS,
    CONFIDENCE_THRESHOLD,
    DAILY_CHAT_DIR,
    DATASET_CHAT_DIR,
    EMBEDDING_DIM,
    LSTM_UNITS,
    MAX_SEQUENCE_LENGTH,
    MAX_VOCAB_SIZE,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    MODELS_DIR,
    TOKENIZER_PATH,
    WRONG_CONTEXT_DIR,
)
from preprocessing.text_preprocessor import TextPreprocessor

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def load_balanced_data(samples_per_class: int = 1200) -> Tuple[List[str], List[int]]:
    """Sample text examples for each of the 8 classes from the project datasets."""
    preprocessor = TextPreprocessor()
    data: List[Tuple[str, int]] = []

    # Map class labels to integer indices
    label_to_idx = {label: i for i, label in enumerate(CLASS_LABELS)}

    print("Loading samples from project datasets...")

    # 1. Addiction
    addiction_file = DATASET_CHAT_DIR / "FullAddiction.csv"
    if addiction_file.exists():
        df = pd.read_csv(addiction_file, nrows=samples_per_class * 2, encoding_errors="ignore")
        col = "post" if "post" in df.columns else df.columns[0]
        texts = [preprocessor.clean_text(str(t)) for t in df[col].dropna() if len(str(t).strip()) > 20]
        selected = texts[:samples_per_class]
        for t in selected:
            data.append((t, label_to_idx["Addiction"]))
        print(f"  [+] Addiction: loaded {len(selected)} samples")

    # 2. Anxiety
    anxiety_file = DATASET_CHAT_DIR / "FullAnxiety.csv"
    if anxiety_file.exists():
        df = pd.read_csv(anxiety_file, nrows=samples_per_class * 2, encoding_errors="ignore")
        col = "post" if "post" in df.columns else df.columns[0]
        texts = [preprocessor.clean_text(str(t)) for t in df[col].dropna() if len(str(t).strip()) > 20]
        selected = texts[:samples_per_class]
        for t in selected:
            data.append((t, label_to_idx["Anxiety"]))
        print(f"  [+] Anxiety: loaded {len(selected)} samples")

    # 3. Depression
    depression_file = DATASET_CHAT_DIR / "FullDepression.csv"
    if depression_file.exists():
        df = pd.read_csv(depression_file, nrows=samples_per_class * 2, encoding_errors="ignore")
        col = "post" if "post" in df.columns else df.columns[0]
        texts = [preprocessor.clean_text(str(t)) for t in df[col].dropna() if len(str(t).strip()) > 20]
        selected = texts[:samples_per_class]
        for t in selected:
            data.append((t, label_to_idx["Depression"]))
        print(f"  [+] Depression: loaded {len(selected)} samples")

    # 4. Eating Disorder
    ed_file = DATASET_CHAT_DIR / "FullEatingDisorder.csv"
    if ed_file.exists():
        df = pd.read_csv(ed_file, nrows=samples_per_class * 2, encoding_errors="ignore")
        col = "post" if "post" in df.columns else df.columns[0]
        texts = [preprocessor.clean_text(str(t)) for t in df[col].dropna() if len(str(t).strip()) > 20]
        selected = texts[:samples_per_class]
        for t in selected:
            data.append((t, label_to_idx["Eating Disorder"]))
        print(f"  [+] Eating Disorder: loaded {len(selected)} samples")

    # 5. Suicide
    suicide_file = DATASET_CHAT_DIR / "FullSuicide.csv"
    if suicide_file.exists():
        df = pd.read_csv(suicide_file, nrows=samples_per_class * 2, encoding_errors="ignore")
        col = "post" if "post" in df.columns else df.columns[0]
        texts = [preprocessor.clean_text(str(t)) for t in df[col].dropna() if len(str(t).strip()) > 20]
        selected = texts[:samples_per_class]
        for t in selected:
            data.append((t, label_to_idx["Suicide"]))
        print(f"  [+] Suicide: loaded {len(selected)} samples")

    # 6. OOD (Out of Distribution) from worng_context
    ood_texts = []
    ag_news_file = WRONG_CONTEXT_DIR / "train.csv"
    if ag_news_file.exists():
        df_ag = pd.read_csv(ag_news_file, nrows=samples_per_class * 2, encoding_errors="ignore")
        col = "Description" if "Description" in df_ag.columns else df_ag.columns[-1]
        for t in df_ag[col].dropna():
            cleaned = preprocessor.clean_text(str(t))
            if len(cleaned) > 20:
                ood_texts.append(cleaned)

    selected_ood = ood_texts[:samples_per_class]
    for t in selected_ood:
        data.append((t, label_to_idx["OOD"]))
    print(f"  [+] OOD: loaded {len(selected_ood)} samples")

    # 7. Neutral & Greeting from daily_chat/train.csv
    neutral_texts = []
    greeting_texts = []
    daily_file = DAILY_CHAT_DIR / "train.csv"
    if daily_file.exists():
        df_daily = pd.read_csv(daily_file, nrows=2000, encoding_errors="ignore")
        col = "dialogue" if "dialogue" in df_daily.columns else df_daily.columns[1]

        greeting_starters = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", "how are you", "nice to meet you", "greetings"]
        for dialogue_str in df_daily[col].dropna():
            turns = str(dialogue_str).split("\n")
            for turn in turns:
                # Strip #PersonX#: prefixes
                clean_turn = turn
                if ":" in clean_turn:
                    clean_turn = clean_turn.split(":", 1)[1]
                clean_turn = preprocessor.clean_text(clean_turn)
                if not clean_turn or len(clean_turn) < 6:
                    continue

                is_greet = any(clean_turn.startswith(g) for g in greeting_starters)
                if is_greet and len(greeting_texts) < samples_per_class:
                    greeting_texts.append(clean_turn)
                elif not is_greet and len(neutral_texts) < samples_per_class:
                    neutral_texts.append(clean_turn)

    # Augment greetings with common natural variations if needed
    base_greetings = [
        "hello", "hi there", "hey", "good morning", "good evening", "howdy",
        "hello assistant", "hi I want to talk", "hey there", "greetings",
        "good day", "hello how are you doing", "hi can we talk"
    ]
    while len(greeting_texts) < samples_per_class:
        for bg in base_greetings:
            if len(greeting_texts) < samples_per_class:
                greeting_texts.append(preprocessor.clean_text(bg))

    for t in neutral_texts[:samples_per_class]:
        data.append((t, label_to_idx["Neutral"]))
    print(f"  [+] Neutral: loaded {len(neutral_texts[:samples_per_class])} samples")

    for t in greeting_texts[:samples_per_class]:
        data.append((t, label_to_idx["Greeting"]))
    print(f"  [+] Greeting: loaded {len(greeting_texts[:samples_per_class])} samples")

    # Shuffle the dataset
    random.shuffle(data)
    X = [item[0] for item in data]
    y = [item[1] for item in data]
    print(f"Total dataset compiled: {len(X)} examples across {len(CLASS_LABELS)} classes.")
    return X, y


def build_and_train_model():
    """Build, train, and export the Bi-LSTM model and tokenizer."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X_texts, y_labels = load_balanced_data(samples_per_class=1000)

    # 1. Fit Tokenizer
    print(f"Fitting Keras Tokenizer (max_words={MAX_VOCAB_SIZE})...")
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=MAX_VOCAB_SIZE,
        oov_token="<OOV>",
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
        lower=True,
    )
    tokenizer.fit_on_texts(X_texts)

    sequences = tokenizer.texts_to_sequences(X_texts)
    X_padded = tf.keras.preprocessing.sequence.pad_sequences(
        sequences, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post"
    )
    y_array = np.array(y_labels, dtype=np.int32)

    # 2. Build Bi-LSTM Architecture
    num_classes = len(CLASS_LABELS)
    print("Building Bi-LSTM Deep Learning Architecture...")
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(
            input_dim=MAX_VOCAB_SIZE,
            output_dim=EMBEDDING_DIM,
            input_length=MAX_SEQUENCE_LENGTH,
            name="embedding_layer",
        ),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(LSTM_UNITS, return_sequences=False),
            name="bidirectional_lstm",
        ),
        tf.keras.layers.Dropout(0.3, name="dropout_1"),
        tf.keras.layers.Dense(64, activation="relu", name="dense_intermediate"),
        tf.keras.layers.Dropout(0.2, name="dropout_2"),
        tf.keras.layers.Dense(num_classes, activation="softmax", name="output_classification"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # 3. Train Model
    print("Training Bi-LSTM text classification model...")
    epochs = 4
    batch_size = 64

    # Validation split 15%
    history = model.fit(
        X_padded,
        y_array,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.15,
        verbose=1,
    )

    # 4. Save Artifacts
    print(f"Saving model to {MODEL_PATH}...")
    model.save(str(MODEL_PATH))

    print(f"Saving tokenizer to {TOKENIZER_PATH}...")
    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

    # 5. Save Metadata
    metadata = {
        "architecture": "Bi-LSTM (Bidirectional Long Short-Term Memory)",
        "task": "Multi-class Text Classification",
        "framework": "TensorFlow / Keras",
        "num_classes": num_classes,
        "class_labels": CLASS_LABELS,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "max_vocab_size": MAX_VOCAB_SIZE,
        "embedding_dim": EMBEDDING_DIM,
        "lstm_units": LSTM_UNITS,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "vocab_size_fitted": min(len(tokenizer.word_index) + 1, MAX_VOCAB_SIZE),
        "total_training_samples": len(X_texts),
        "final_accuracy": float(history.history["accuracy"][-1]),
        "final_val_accuracy": float(history.history["val_accuracy"][-1]),
    }
    with open(MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved metadata to {MODEL_METADATA_PATH}")
    print("Training and export complete!")


if __name__ == "__main__":
    build_and_train_model()

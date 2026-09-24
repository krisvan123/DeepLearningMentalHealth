"""Tokenizer module for MindCare.

Loads and wraps tokenization artifacts (e.g., Keras Tokenizer / custom vocabulary)
and handles sequence padding and truncation.
"""

import pickle
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np

from config.config import MAX_SEQUENCE_LENGTH, TOKENIZER_PATH
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def pad_sequence_vector(
    sequence: List[int],
    max_len: int = MAX_SEQUENCE_LENGTH,
    padding: str = "post",
    truncating: str = "post",
) -> np.ndarray:
    """Pad or truncate a 1D sequence of integers to fixed length max_len."""
    if len(sequence) > max_len:
        if truncating == "post":
            seq = sequence[:max_len]
        else:
            seq = sequence[-max_len:]
    else:
        seq = sequence

    padded = np.zeros(max_len, dtype=np.int32)
    if padding == "post":
        padded[: len(seq)] = seq
    else:
        padded[-len(seq) :] = seq

    return padded


class TokenizerWrapper:
    """Wraps tokenizer artifact and handles text-to-tensor transformations."""

    def __init__(
        self,
        tokenizer_obj: Optional[Any] = None,
        max_length: int = MAX_SEQUENCE_LENGTH,
    ):
        self.tokenizer = tokenizer_obj
        self.max_length = max_length

    @classmethod
    def load_from_file(
        cls,
        tokenizer_path: Union[str, Path] = TOKENIZER_PATH,
        max_length: int = MAX_SEQUENCE_LENGTH,
    ) -> "TokenizerWrapper":
        """Load tokenizer instance from serialized pickle file."""
        path = Path(tokenizer_path)
        if not path.exists():
            raise FileNotFoundError(f"Tokenizer artifact not found at {path}")

        try:
            with open(path, "rb") as f:
                tokenizer_obj = pickle.load(f)
            logger.info("Successfully loaded tokenizer from %s", path)
            return cls(tokenizer_obj=tokenizer_obj, max_length=max_length)
        except Exception as exc:
            logger.error("Failed to load tokenizer from %s: %s", path, exc)
            raise

    def texts_to_sequences(self, texts: List[str]) -> List[List[int]]:
        """Convert list of texts to integer token sequences."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer artifact has not been initialized.")

        # Check if tokenizer has texts_to_sequences (Keras Tokenizer interface)
        if hasattr(self.tokenizer, "texts_to_sequences"):
            return self.tokenizer.texts_to_sequences(texts)

        # Check if tokenizer is a word-to-index dict
        if isinstance(self.tokenizer, dict):
            unk_idx = self.tokenizer.get("<OOV>", self.tokenizer.get("<unk>", 1))
            sequences = []
            for text in texts:
                tokens = text.split()
                seq = [self.tokenizer.get(tok, unk_idx) for tok in tokens]
                sequences.append(seq)
            return sequences

        raise NotImplementedError(
            f"Unsupported tokenizer object type: {type(self.tokenizer)}"
        )

    def prepare_input(self, text: str) -> np.ndarray:
        """Tokenize, pad, and format single text into model input shape (1, max_len)."""
        sequences = self.texts_to_sequences([text])
        seq = sequences[0] if sequences else []
        padded = pad_sequence_vector(seq, max_len=self.max_length)
        return np.expand_dims(padded, axis=0)

    @property
    def is_ready(self) -> bool:
        """Check if tokenizer is ready for conversion."""
        return self.tokenizer is not None

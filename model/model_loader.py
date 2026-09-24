"""Model loader module for MindCare.

Loads and caches Deep Learning models and tokenizers efficiently.
Supports TensorFlow/Keras and PyTorch models with graceful fallback when artifacts are missing.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Tuple, Union

from config.config import MODEL_PATH, TOKENIZER_PATH
from preprocessing.tokenizer import TokenizerWrapper
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def _load_keras_model(filepath: Path) -> Any:
    """Load a Keras/TensorFlow model from disk."""
    import tensorflow as tf

    # Suppress verbose TF logging during inference
    tf.get_logger().setLevel("ERROR")
    logger.info("Loading Keras model from %s", filepath)
    model = tf.keras.models.load_model(str(filepath), compile=False)
    return model


def _load_torch_model(filepath: Path) -> Any:
    """Load a PyTorch model from disk."""
    import torch

    logger.info("Loading PyTorch model from %s", filepath)
    model = torch.load(str(filepath), map_location=torch.device("cpu"))
    if hasattr(model, "eval"):
        model.eval()
    return model


def _load_model_internal(model_path: Path) -> Tuple[Optional[Any], Optional[str]]:
    """Internal implementation for loading model without caching wrapper."""
    if not model_path.exists():
        msg = f"Model artifact not found at '{model_path}'."
        logger.info(msg)
        return None, msg

    try:
        suffix = model_path.suffix.lower()
        if suffix in [".h5", ".keras"] or model_path.is_dir():
            model = _load_keras_model(model_path)
            return model, None
        elif suffix in [".pt", ".pth"]:
            model = _load_torch_model(model_path)
            return model, None
        else:
            # Try keras first, then torch
            try:
                model = _load_keras_model(model_path)
                return model, None
            except Exception:
                model = _load_torch_model(model_path)
                return model, None
    except Exception as exc:
        err_msg = f"Error loading model from {model_path}: {exc}"
        logger.error(err_msg)
        return None, err_msg


def _load_tokenizer_internal(tokenizer_path: Path) -> Tuple[Optional[TokenizerWrapper], Optional[str]]:
    """Internal implementation for loading tokenizer."""
    if not tokenizer_path.exists():
        msg = f"Tokenizer artifact not found at '{tokenizer_path}'."
        logger.info(msg)
        return None, msg

    try:
        wrapper = TokenizerWrapper.load_from_file(tokenizer_path)
        return wrapper, None
    except Exception as exc:
        err_msg = f"Error loading tokenizer from {tokenizer_path}: {exc}"
        logger.error(err_msg)
        return None, err_msg


# In-memory LRU caching for loaded models and tokenizers
@lru_cache(maxsize=4)
def load_cached_model(path_str: str) -> Tuple[Optional[Any], Optional[str]]:
    return _load_model_internal(Path(path_str))


@lru_cache(maxsize=4)
def load_cached_tokenizer(path_str: str) -> Tuple[Optional[TokenizerWrapper], Optional[str]]:
    return _load_tokenizer_internal(Path(path_str))


def load_model(model_path: Union[str, Path] = MODEL_PATH) -> Tuple[Optional[Any], Optional[str]]:
    """Public interface to load model using caching."""
    return load_cached_model(str(model_path))


def load_tokenizer(tokenizer_path: Union[str, Path] = TOKENIZER_PATH) -> Tuple[Optional[TokenizerWrapper], Optional[str]]:
    """Public interface to load tokenizer using caching."""
    return load_cached_tokenizer(str(tokenizer_path))

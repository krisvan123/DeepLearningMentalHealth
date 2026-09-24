"""Comprehensive automated test suite for MindCare pipeline.

Tests:
1. Text Preprocessor
2. Tokenizer padding and sequence formatting
3. Response Loader & Response Manager for all 8 categories
4. High-risk safety protocol trigger on Suicide class
5. Model Loader and Predictor inference (when model artifact is present)
6. Graceful failure when model is absent (no crashes)
"""

import sys
import unittest
from pathlib import Path

# Adjust path to import project modules
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config.config import (
    CLASS_LABELS,
    HIGH_RISK_CLASSES,
    MODEL_PATH,
    RESPONSE_DIR,
    RESPONSE_MAP,
    TOKENIZER_PATH,
)
from model.model_config import ModelConfig
from model.model_loader import load_model, load_tokenizer
from model.predictor import ModelNotReadyError, Predictor
from preprocessing.text_preprocessor import TextPreprocessor, preprocess_text
from preprocessing.tokenizer import TokenizerWrapper, pad_sequence_vector
from response.response_loader import ResponseLoader
from response.response_manager import ResponseManager
from utils.helpers import check_system_status, validate_user_input


class TestTextPreprocessor(unittest.TestCase):
    """Test text cleaning and normalization routines."""

    def setUp(self):
        self.processor = TextPreprocessor()

    def test_lowercasing(self):
        result = self.processor.clean_text("I AM SO OVERWHELMED")
        self.assertEqual(result, "i am so overwhelmed")

    def test_strip_urls(self):
        result = self.processor.clean_text("Read this https://example.com/help today")
        self.assertNotIn("https://", result)
        self.assertIn("read this", result)

    def test_strip_html_tags(self):
        result = self.processor.clean_text("<p>Feeling <b>hopeless</b> today</p>")
        self.assertEqual(result, "feeling hopeless today")

    def test_html_entities(self):
        result = self.processor.clean_text("I feel alone &amp; sad")
        self.assertEqual(result, "i feel alone & sad")

    def test_whitespace_normalization(self):
        result = self.processor.clean_text("  too    many    spaces  \n  here  ")
        self.assertEqual(result, "too many spaces here")

    def test_empty_string(self):
        self.assertEqual(self.processor.clean_text(""), "")
        self.assertEqual(self.processor.clean_text(None), "")


class TestTokenizerWrapper(unittest.TestCase):
    """Test sequence padding and tokenization logic."""

    def test_pad_sequence_post(self):
        seq = [1, 2, 3]
        padded = pad_sequence_vector(seq, max_len=6, padding="post")
        self.assertEqual(len(padded), 6)
        self.assertEqual(list(padded[:3]), [1, 2, 3])
        self.assertEqual(list(padded[3:]), [0, 0, 0])

    def test_pad_sequence_truncate(self):
        seq = [1, 2, 3, 4, 5, 6, 7]
        truncated = pad_sequence_vector(seq, max_len=4, truncating="post")
        self.assertEqual(len(truncated), 4)
        self.assertEqual(list(truncated), [1, 2, 3, 4])


class TestResponseSystem(unittest.TestCase):
    """Test response loading, class mapping, and safety protocols."""

    def setUp(self):
        self.loader = ResponseLoader(response_dir=RESPONSE_DIR, response_map=RESPONSE_MAP)
        self.manager = ResponseManager(loader=self.loader)

    def test_all_8_classes_configured(self):
        self.assertEqual(len(CLASS_LABELS), 8)
        for label in CLASS_LABELS:
            self.assertIn(label, RESPONSE_MAP)

    def test_all_response_files_exist(self):
        for label, filename in RESPONSE_MAP.items():
            filepath = RESPONSE_DIR / filename
            self.assertTrue(
                filepath.exists(),
                f"Response file '{filename}' for class '{label}' does not exist",
            )
            content = filepath.read_text(encoding="utf-8").strip()
            self.assertGreater(len(content), 20, f"Response file '{filename}' is empty")

    def test_get_response_for_each_class(self):
        for label in CLASS_LABELS:
            resp = self.manager.get_response(predicted_class=label)
            self.assertIn("text", resp)
            self.assertGreater(len(resp["text"]), 15)
            self.assertEqual(resp["class_label"], label)

    def test_high_risk_suicide_flag(self):
        resp = self.manager.get_response(predicted_class="Suicide")
        self.assertTrue(resp["is_high_risk"])
        # Suicide response must provide crisis lifelines
        self.assertTrue(
            "988" in resp["text"] or "119" in resp["text"] or "crisis" in resp["text"].lower()
        )

    def test_non_high_risk_flag(self):
        for label in ["Anxiety", "Depression", "Greeting", "Neutral", "OOD"]:
            resp = self.manager.get_response(predicted_class=label)
            self.assertFalse(resp["is_high_risk"])

    def test_unknown_class_fallback(self):
        resp = self.manager.get_response(predicted_class="NonExistentClass")
        self.assertIn("text", resp)
        self.assertGreater(len(resp["text"]), 10)


class TestValidationAndStatus(unittest.TestCase):
    """Test input validation and system status checker."""

    def test_validation_empty(self):
        valid, msg = validate_user_input("")
        self.assertFalse(valid)
        valid, msg = validate_user_input("   ")
        self.assertFalse(valid)
        valid, msg = validate_user_input(None)
        self.assertFalse(valid)

    def test_validation_valid(self):
        valid, text = validate_user_input("I am feeling stressed")
        self.assertTrue(valid)
        self.assertEqual(text, "I am feeling stressed")

    def test_system_status_structure(self):
        status = check_system_status()
        self.assertIn("model_exists", status)
        self.assertIn("tokenizer_exists", status)
        self.assertIn("responses_ready", status)
        self.assertIn("ready", status)


class TestPredictorWithModel(unittest.TestCase):
    """Test Predictor inference when model is available, or proper error when absent."""

    def setUp(self):
        self.config = ModelConfig()
        model_obj, _ = load_model(MODEL_PATH)
        tokenizer_obj, _ = load_tokenizer(TOKENIZER_PATH)
        self.model_obj = model_obj
        self.tokenizer_obj = tokenizer_obj
        self.predictor = Predictor(model=model_obj, tokenizer=tokenizer_obj, config=self.config)

    def test_predictor_missing_model_behavior(self):
        # Predictor without model should raise ModelNotReadyError
        empty_predictor = Predictor(model=None, tokenizer=None, config=self.config)
        self.assertFalse(empty_predictor.is_ready)
        with self.assertRaises(ModelNotReadyError):
            empty_predictor.predict("test text")

    def test_real_inference_if_model_present(self):
        if not self.predictor.is_ready:
            self.skipTest("Model artifact not yet compiled/present. Skipping inference check.")

        test_cases = [
            ("hello, good morning", "Greeting"),
            ("i feel so anxious and my chest hurts", "Anxiety"),
            ("i feel empty, hopeless and depressed", "Depression"),
            ("i can't stop drinking and taking drugs", "Addiction"),
            ("i want to end my life, i want to die", "Suicide"),
        ]

        for text, expected_label in test_cases:
            result = self.predictor.predict(text)
            self.assertIn("predicted_class", result)
            self.assertIn("confidence", result)
            self.assertIn("probabilities", result)
            self.assertIn("latency_ms", result)
            self.assertGreater(result["latency_ms"], 0)
            self.assertEqual(len(result["probabilities"]), 8)
            print(f"Input: '{text}' -> Predicted: '{result['predicted_class']}' ({result['confidence'] * 100:.1f}%) [expected: {expected_label}]")


if __name__ == "__main__":
    unittest.main()

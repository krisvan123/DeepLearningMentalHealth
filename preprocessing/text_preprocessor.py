"""Text Preprocessor module for MindCare.

Cleans and normalizes incoming user text for NLP tokenization and model inference.
"""

import html
import re
import unicodedata


class TextPreprocessor:
    """Standardizes raw text input before tokenization."""

    # Regex patterns for cleaning
    _URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
    _HTML_TAG_PATTERN = re.compile(r"<.*?>")
    _MENTION_PATTERN = re.compile(r"@[A-Za-z0-9_]+")
    _EXTRA_SPACE_PATTERN = re.compile(r"\s+")
    _REPEATED_PUNCT_PATTERN = re.compile(r"([!?.,;])\1+")

    def __init__(self, lowercase: bool = True, strip_urls: bool = True):
        self.lowercase = lowercase
        self.strip_urls = strip_urls

    def clean_text(self, text: str) -> str:
        """Run complete cleaning pipeline on input text."""
        if not text:
            return ""

        # Decode HTML entities (e.g., &amp;, &quot;)
        cleaned = html.unescape(text)

        # Normalize unicode (NFC)
        cleaned = unicodedata.normalize("NFC", cleaned)

        # Remove URLs
        if self.strip_urls:
            cleaned = self._URL_PATTERN.sub(" ", cleaned)

        # Remove HTML tags
        cleaned = self._HTML_TAG_PATTERN.sub(" ", cleaned)

        # Remove @user mentions
        cleaned = self._MENTION_PATTERN.sub(" ", cleaned)

        # Compress repeated punctuation (e.g., '???' -> '?')
        cleaned = self._REPEATED_PUNCT_PATTERN.sub(r"\1", cleaned)

        # Lowercase
        if self.lowercase:
            cleaned = cleaned.lower()

        # Normalize whitespace
        cleaned = self._EXTRA_SPACE_PATTERN.sub(" ", cleaned).strip()

        return cleaned

    def __call__(self, text: str) -> str:
        return self.clean_text(text)


def preprocess_text(text: str) -> str:
    """Convenience functional interface for text preprocessing."""
    processor = TextPreprocessor()
    return processor.clean_text(text)

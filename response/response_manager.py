"""Response manager for MindCare chatbot.

Coordinates selection of controlled responses based on predicted class,
confidence, and risk level.
"""

import random
from typing import Any, Dict, List, Optional

from config.config import HIGH_RISK_CLASSES
from response.response_loader import ResponseLoader
from utils.logging_utils import get_logger

logger = get_logger(__name__)

SAFE_FALLBACK_RESPONSE = (
    "Thank you for sharing your thoughts. I am listening and here to support your reflection. "
    "Please feel free to share whatever else is on your mind."
)


class ResponseManager:
    """Selects and formats safe, controlled responses for predicted classes."""

    def __init__(self, loader: Optional[ResponseLoader] = None):
        self.loader = loader or ResponseLoader()

    def get_response(
        self,
        predicted_class: str,
        confidence: Optional[float] = None,
        last_response_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Obtain a safe response mapped to the predicted class.

        Args:
            predicted_class: Label predicted by the classification pipeline.
            confidence: Optional confidence score (0.0 to 1.0).
            last_response_text: Optional last response text to avoid repetition.

        Returns:
            Dictionary containing:
                - text: The response message string
                - class_label: Category label
                - is_high_risk: Boolean indicating if crisis protocol applies
                - confidence: Score or None
        """
        is_high_risk = predicted_class in HIGH_RISK_CLASSES
        candidates: List[str] = self.loader.get_responses_for_class(predicted_class)

        if not candidates:
            logger.warning(
                "No responses configured for class '%s'. Using safe fallback.",
                predicted_class,
            )
            response_text = SAFE_FALLBACK_RESPONSE
        elif len(candidates) == 1 or last_response_text is None:
            response_text = candidates[0]
        else:
            # Filter out immediate repetition if alternatives exist
            alternatives = [c for c in candidates if c != last_response_text]
            if alternatives:
                response_text = random.choice(alternatives)
            else:
                response_text = random.choice(candidates)

        return {
            "text": response_text,
            "class_label": predicted_class,
            "is_high_risk": is_high_risk,
            "confidence": confidence,
        }

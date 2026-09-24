"""Response loader for MindCare chatbot.

Loads, parses, and caches controlled response templates from the respon/ directory.
"""

from pathlib import Path
from typing import Dict, List

from config.config import RESPONSE_DIR, RESPONSE_MAP
from utils.helpers import safe_read_text
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class ResponseLoader:
    """Manages loading and caching of response templates from disk."""

    def __init__(self, response_dir: Path = RESPONSE_DIR, response_map: Dict[str, str] = RESPONSE_MAP):
        self.response_dir = Path(response_dir)
        self.response_map = response_map
        self._cache: Dict[str, List[str]] = {}
        self._load_all()

    def _load_all(self) -> None:
        """Load all response templates into memory cache."""
        self._cache.clear()
        if not self.response_dir.exists():
            logger.warning("Response directory does not exist: %s", self.response_dir)
            return

        for category, filename in self.response_map.items():
            filepath = self.response_dir / filename
            if not filepath.exists():
                logger.warning("Missing response file for category '%s': %s", category, filepath)
                continue

            try:
                content = safe_read_text(filepath)
                # Parse responses separated by delimiter '---'
                parts = [p.strip() for p in content.split("---") if p.strip()]
                if parts:
                    self._cache[category] = parts
                else:
                    self._cache[category] = [content.strip()] if content.strip() else []
                logger.debug("Loaded %d responses for '%s'", len(self._cache[category]), category)
            except Exception as exc:
                logger.error("Error loading response file '%s': %s", filepath, exc)

    def reload(self) -> None:
        """Force reload response templates from disk."""
        self._load_all()

    def get_responses_for_class(self, category: str) -> List[str]:
        """Return list of response variations for a given category."""
        return self._cache.get(category, [])

    def is_category_available(self, category: str) -> bool:
        """Check if at least one response exists for category."""
        return bool(self._cache.get(category))

import re
from typing import List, Optional, Tuple, Dict


class InputValidator:
    MAX_CONTENT_LENGTH = 10000
    MAX_LABEL_LENGTH = 50
    MAX_LABELS = 20
    MAX_METADATA_KEYS = 50

    @staticmethod
    def sanitize_content(content: str) -> str:
        content = content.replace("\x00", "")
        content = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", content)
        return content.strip()

    @staticmethod
    def validate_store(content: str, labels: Optional[List[str]],
                       weight: float, metadata: Optional[Dict]) -> Tuple[bool, Optional[str]]:
        if not content or not content.strip():
            return False, "Content cannot be empty"
        if len(content) > InputValidator.MAX_CONTENT_LENGTH:
            return False, f"Content exceeds max length ({InputValidator.MAX_CONTENT_LENGTH})"
        if weight < 0.1 or weight > 10.0:
            return False, "Weight must be between 0.1 and 10.0"
        if labels is not None:
            if len(labels) > InputValidator.MAX_LABELS:
                return False, f"Too many labels (max {InputValidator.MAX_LABELS})"
            for lbl in labels:
                if not lbl or len(lbl) > InputValidator.MAX_LABEL_LENGTH:
                    return False, f"Invalid label: {lbl[:20]}"
                if re.match(r"^__.*__$", lbl):
                    return False, f"Reserved label pattern: {lbl}"
        if metadata is not None:
            if len(metadata) > InputValidator.MAX_METADATA_KEYS:
                return False, f"Too many metadata keys (max {InputValidator.MAX_METADATA_KEYS})"
        return True, None

    @staticmethod
    def validate_query(query: str, k: int) -> Tuple[bool, Optional[str]]:
        if not query and query != "":
            return False, "Query text is required"
        if len(query) > InputValidator.MAX_CONTENT_LENGTH:
            return False, f"Query exceeds max length ({InputValidator.MAX_CONTENT_LENGTH})"
        if k < 1 or k > 100:
            return False, "k must be between 1 and 100"
        return True, None

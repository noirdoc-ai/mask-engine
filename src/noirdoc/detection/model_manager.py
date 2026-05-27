from __future__ import annotations

import spacy.cli
import spacy.util
import structlog

logger = structlog.get_logger()

MODEL_MAP: dict[str, str] = {
    "de": "de_core_news_lg",
    "en": "en_core_web_lg",
}


def ensure_spacy_models(languages: list[str]) -> None:
    """Check for required spaCy models and download any that are missing."""
    for lang in languages:
        model_name = MODEL_MAP.get(lang)
        if model_name is None:
            continue
        if spacy.util.is_package(model_name):
            logger.info("spacy.model_present", model=model_name)
            continue
        logger.info("spacy.model_downloading", model=model_name)
        # spaCy exposes ``download`` at runtime but does not list it in
        # ``spacy.cli.__all__``, so mypy reports attr-defined. Accessing it as an
        # attribute (rather than a direct import) is also what the test suite
        # patches, so we keep the attribute access and narrowly silence the check.
        spacy.cli.download(model_name)  # type: ignore[attr-defined]
        logger.info("spacy.model_downloaded", model=model_name)

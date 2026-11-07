from __future__ import annotations

import asyncio
import re

import structlog

from noirdoc.detection.base import BaseDetector, DetectedEntity, DetectionError

log = structlog.get_logger(__name__)

# Strong indicators: if ANY of these appear in a multi-word PERSON entity, reject it.
_PERSON_STRONG_REJECT: set[str] = {
    # verbs / participles commonly absorbed by spaCy NER
    "wohnhaft",
    "verheiratet",
    "geschieden",
    "ledig",
    "verwitwet",
    "überwiesen",
    "geboren",
    "verstorben",
    "beschäftigt",
    "gemeldet",
    # common nouns that get falsely tagged
    "euro",
    "straße",
    "strasse",
    "nummer",
    "datum",
}

# Weak indicators: reject only if entity text ENDS with one of these
# (handles boundary absorption like "Hoffmann, und").
_PERSON_TRAILING_REJECT: set[str] = {
    "und",
    "oder",
    "mit",
    "für",
    "bei",
    "nach",
    "der",
    "die",
    "das",
    "ein",
    "eine",
    "er",
    "sie",
    "es",
}

# Pattern for username-like strings (lowercase + digits, no spaces)
_USERNAME_PATTERN = re.compile(r"^[a-z0-9._@\-]+$")

# InvertedNameRecognizer matches "Lastname, Firstname" / "Lastname, Dr. Firstname".
# Exempt that shape from the general comma rejection.
_INVERTED_NAME_PATTERN = re.compile(
    r"^[A-ZÄÖÜ][a-zäöüß]+(?:-[A-ZÄÖÜ][a-zäöüß]+)?,\s+"
    r"(?:(?:Dr|Prof)\.\s+)?[A-ZÄÖÜ][a-zäöüß]+(?:-[A-ZÄÖÜ][a-zäöüß]+)?$"
)


def _validate_person(entity: DetectedEntity) -> bool:
    """Return False if a PERSON entity is likely a false positive."""
    if entity.entity_type != "PERSON":
        return True

    text = entity.text

    if "\n" in text:
        return False

    # Reject entities containing commas (spaCy boundary absorption like "Hoffmann, und"),
    # except the "Lastname, Firstname" form produced by InvertedNameRecognizer.
    if "," in text and not _INVERTED_NAME_PATTERN.match(text):
        return False

    # Reject single-character entities
    if len(text) <= 1:
        return False

    # Reject username-like patterns (e.g. "cthwhdbfio0854", "al-kurdy8")
    if _USERNAME_PATTERN.match(text):
        return False

    # Multi-word validation
    words = [w.rstrip(",") for w in text.lower().split()]
    if len(words) > 1:
        if any(w in _PERSON_STRONG_REJECT for w in words):
            return False
        if words[-1] in _PERSON_TRAILING_REJECT:
            return False

    return True


# Per-entity-type score thresholds (higher = stricter, fewer FPs).
# Types not listed here use the global default.
_TYPE_THRESHOLDS: dict[str, float] = {
    "URL": 0.6,
    "DATE": 0.7,
}


class EnsembleDetector:
    """Kombiniert mehrere Detektoren und löst Überlappungen auf."""

    def __init__(
        self,
        detectors: list[BaseDetector],
        score_threshold: float = 0.5,
        *,
        fail_closed: bool = False,
    ) -> None:
        self.detectors = detectors
        self.score_threshold = score_threshold
        self.fail_closed = fail_closed

    async def detect(self, text: str, language: str = "de") -> list[DetectedEntity]:
        if not text:
            return []

        if not self.detectors:
            raise DetectionError("No PII detectors configured")

        results = await asyncio.gather(
            *(self._run_one(d, text, language) for d in self.detectors),
        )

        all_entities: list[DetectedEntity] = []
        failed: list[str] = []
        for detector, result in zip(self.detectors, results, strict=True):
            if result is None:
                failed.append(getattr(detector, "name", detector.__class__.__name__))
            else:
                all_entities.extend(result)

        if failed and (self.fail_closed or len(failed) == len(self.detectors)):
            raise DetectionError(
                f"PII detector(s) failed ({', '.join(failed)}); refusing to return unmasked text",
            )

        filtered = [
            e
            for e in all_entities
            if e.score >= _TYPE_THRESHOLDS.get(e.entity_type, self.score_threshold)
        ]
        # Validate BEFORE merging so a rejected span (e.g. "Max Müller wohnhaft")
        # cannot win the overlap resolution against a valid shorter span ("Max Müller")
        # and then get dropped, leaving the name unmasked.
        validated = [e for e in filtered if _validate_person(e)]
        merged = self._merge_entities(validated, text)
        return sorted(merged, key=lambda e: e.start)

    @staticmethod
    async def _run_one(
        detector: BaseDetector,
        text: str,
        language: str,
    ) -> list[DetectedEntity] | None:
        """Run one detector. On failure, log and return ``None``.

        Callers distinguish ``None`` (detector crashed) from ``[]`` (detector
        ran cleanly and found nothing) so ``detect()`` can raise
        :class:`DetectionError` when every detector fails (or when
        ``fail_closed=True``).
        """
        try:
            return await detector.detect(text, language)
        except Exception as exc:
            log.warning(
                "detection.detector_failed",
                detector=getattr(detector, "name", detector.__class__.__name__),
                language=language,
                error=str(exc),
            )
            return None

    def _merge_entities(
        self,
        entities: list[DetectedEntity],
        text: str | None = None,
    ) -> list[DetectedEntity]:
        """Resolve all overlapping spans — regardless of entity_type — into a
        single non-overlapping entity per transitively-overlapping cluster.

        1. Sort by start ascending, then span length descending.
        2. Group transitively overlapping spans into clusters.
        3. Within each cluster, pick the winning entity via ``_pick_winner``
           (higher score → longer span → presidio) for ``entity_type``/``score``/
           ``source``, and expand the span to the union ``[min(start), max(end))``
           so no detected character is left unmasked.
        """
        if not entities:
            return []

        sorted_ents = sorted(
            entities,
            key=lambda e: (e.start, -(e.end - e.start)),
        )

        accepted: list[DetectedEntity] = []
        cluster: list[DetectedEntity] = [sorted_ents[0]]
        cluster_end = sorted_ents[0].end

        for candidate in sorted_ents[1:]:
            if candidate.start < cluster_end:
                cluster.append(candidate)
                if candidate.end > cluster_end:
                    cluster_end = candidate.end
            else:
                accepted.append(self._resolve_cluster(cluster, text))
                cluster = [candidate]
                cluster_end = candidate.end

        accepted.append(self._resolve_cluster(cluster, text))
        return accepted

    @classmethod
    def _resolve_cluster(
        cls,
        cluster: list[DetectedEntity],
        text: str | None = None,
    ) -> DetectedEntity:
        winner = cluster[0]
        for candidate in cluster[1:]:
            winner = cls._pick_winner(winner, candidate)

        start = min(e.start for e in cluster)
        end = max(e.end for e in cluster)
        if winner.start == start and winner.end == end:
            return winner

        for ent in cluster:
            if ent.start == start and ent.end == end:
                span_text = ent.text
                break
        else:
            if (
                text is not None
                and len(text) >= end
                and all(text[e.start : e.end] == e.text for e in cluster)
            ):
                span_text = text[start:end]
            else:
                span_text = cls._union_text(cluster, start, end)

        return DetectedEntity(
            entity_type=winner.entity_type,
            text=span_text,
            start=start,
            end=end,
            score=winner.score,
            source=winner.source,
        )

    @staticmethod
    def _union_text(cluster: list[DetectedEntity], start: int, end: int) -> str:
        length = end - start
        chars: list[str | None] = [None] * length
        for ent in cluster:
            offset = ent.start - start
            for i, ch in enumerate(ent.text):
                pos = offset + i
                if 0 <= pos < length and chars[pos] is None:
                    chars[pos] = ch
        return "".join(c if c is not None else " " for c in chars)

    @staticmethod
    def _pick_winner(a: DetectedEntity, b: DetectedEntity) -> DetectedEntity:
        if a.score != b.score:
            return a if a.score > b.score else b
        len_a = a.end - a.start
        len_b = b.end - b.start
        if len_a != len_b:
            return a if len_a > len_b else b
        # Tie-break: presidio wins
        if a.source == "presidio":
            return a
        if b.source == "presidio":
            return b
        return a

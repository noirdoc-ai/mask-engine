from __future__ import annotations

import asyncio
import re

import structlog

from noirdoc.detection.base import BaseDetector, DetectedEntity

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

# "Lastname, Firstname" / "Lastname, Dr. Firstname" / "Doe-Roe, Firstname".
# Mirrors the InvertedNameRecognizer patterns so their hits survive validation.
_INVERTED_NAME_PATTERN = re.compile(
    r"^[A-ZÄÖÜ][a-zäöüß]+(?:-[A-ZÄÖÜ][a-zäöüß]+)?,\s+"
    r"(?:(?:Dr|Prof)\.\s+)?[A-ZÄÖÜ][a-zäöüß]+$",
)


def _validate_person(entity: DetectedEntity) -> bool:
    """Return False if a PERSON entity is likely a false positive."""
    if entity.entity_type != "PERSON":
        return True

    text = entity.text

    # Reject entities containing newlines or commas (spaCy boundary absorption),
    # except the "Lastname, Firstname" form that InvertedNameRecognizer emits.
    if "\n" in text:
        return False
    if "," in text and not _INVERTED_NAME_PATTERN.match(text):
        return False

    # Reject single-character entities
    if len(text) <= 1:
        return False

    # Reject username-like patterns (e.g. "cthwhdbfio0854", "al-kurdy8")
    if _USERNAME_PATTERN.match(text):
        return False

    # Multi-word validation
    words = text.lower().split()
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
    ) -> None:
        self.detectors = detectors
        self.score_threshold = score_threshold

    async def detect(self, text: str, language: str = "de") -> list[DetectedEntity]:
        if not text:
            return []

        results = await asyncio.gather(
            *(self._run_one(d, text, language) for d in self.detectors),
        )

        all_entities: list[DetectedEntity] = []
        for result in results:
            all_entities.extend(result)

        filtered = [
            e
            for e in all_entities
            if e.score >= _TYPE_THRESHOLDS.get(e.entity_type, self.score_threshold)
        ]
        # Validate before merging: a rejected PERSON span must not absorb the
        # valid, shorter spans it overlaps (e.g. "Max Müller wohnhaft" vs "Max Müller").
        validated = [e for e in filtered if _validate_person(e)]
        merged = self._merge_entities(validated)
        return sorted(merged, key=lambda e: e.start)

    @staticmethod
    async def _run_one(
        detector: BaseDetector,
        text: str,
        language: str,
    ) -> list[DetectedEntity]:
        """Run one detector. On failure, log and degrade to empty results.

        A silent ``return_exceptions=True`` would bury detector failures
        and cause silent leakage (e.g. PERSON detection going dark on a
        spaCy load error). We log explicitly so operators can spot the
        degraded state.
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
            return []

    def _merge_entities(self, entities: list[DetectedEntity]) -> list[DetectedEntity]:
        """
        Overlap Resolution:
        1. Group entities into clusters of (transitively) overlapping spans.
        2. Each cluster yields exactly one entity: the winner picks the type
           (higher score → longer span → presidio), the span is the union of
           the cluster so no detected character is left unmasked.
        3. Non-overlapping entities pass through unchanged.

        Overlaps must never survive this step regardless of type: the
        pseudonymization engine replaces spans by offset, and two overlapping
        spans corrupt each other (#34: an IBAN also tagged as LOCATION came
        out as ``<<LOCATION_1>>AN_1>>``).
        """
        sorted_ents = sorted(
            entities,
            key=lambda e: (e.start, -(e.end - e.start)),
        )

        clusters: list[list[DetectedEntity]] = []
        cluster_end = -1
        for candidate in sorted_ents:
            if clusters and candidate.start < cluster_end:
                clusters[-1].append(candidate)
                cluster_end = max(cluster_end, candidate.end)
            else:
                clusters.append([candidate])
                cluster_end = candidate.end

        return [self._resolve_cluster(cluster) for cluster in clusters]

    @classmethod
    def _resolve_cluster(cls, cluster: list[DetectedEntity]) -> DetectedEntity:
        winner = cluster[0]
        for other in cluster[1:]:
            winner = cls._pick_winner(winner, other)

        start = min(e.start for e in cluster)
        end = max(e.end for e in cluster)
        if winner.start == start and winner.end == end:
            return winner

        return winner.model_copy(
            update={"start": start, "end": end, "text": _union_text(cluster, start, end)},
        )

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


def _union_text(cluster: list[DetectedEntity], start: int, end: int) -> str:
    """Rebuild the text of a merged span from the overlapping members' texts."""
    chars: list[str | None] = [None] * (end - start)
    for e in cluster:
        if len(e.text) != e.end - e.start:
            continue
        for offset, char in enumerate(e.text):
            chars[e.start - start + offset] = char
    return "".join(c if c is not None else " " for c in chars)

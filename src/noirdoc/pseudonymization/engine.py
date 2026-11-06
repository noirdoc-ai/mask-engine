from __future__ import annotations

from noirdoc.detection.base import DetectedEntity
from noirdoc.pseudonymization.mapper import PseudonymMapper


class PseudonymizationEngine:
    """Ersetzt erkannte Entities im Text durch Pseudonyme."""

    def pseudonymize(
        self,
        text: str,
        entities: list[DetectedEntity],
        mapper: PseudonymMapper,
    ) -> str:
        """
        Entities von hinten nach vorne ersetzen, damit Offsets stimmen.

        Overlapping spans are merged first: replacing two overlapping spans by
        their original offsets would splice one pseudonym into the other and
        leave fragments of the original text behind (#34).
        """
        result = text
        for entity in reversed(_without_overlaps(text, entities)):
            pseudonym = mapper.get_or_create(entity.text, entity.entity_type)
            result = result[: entity.start] + pseudonym + result[entity.end :]
        return result


def _without_overlaps(text: str, entities: list[DetectedEntity]) -> list[DetectedEntity]:
    """Sort by start and merge overlapping spans into the earliest, longest one."""
    ordered = sorted(entities, key=lambda e: (e.start, -(e.end - e.start)))
    kept: list[DetectedEntity] = []
    for entity in ordered:
        if kept and entity.start < kept[-1].end:
            last = kept[-1]
            if entity.end > last.end:
                kept[-1] = last.model_copy(
                    update={"end": entity.end, "text": text[last.start : entity.end]},
                )
            continue
        kept.append(entity)
    return kept

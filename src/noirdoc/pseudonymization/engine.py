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
        Overlapping spans are merged to their union first so back-to-front
        replacement never overwrites an already-inserted placeholder.
        """
        non_overlapping = _without_overlaps(text, entities)
        sorted_entities = sorted(non_overlapping, key=lambda e: e.start, reverse=True)
        result = text
        for entity in sorted_entities:
            pseudonym = mapper.get_or_create(entity.text, entity.entity_type)
            result = result[: entity.start] + pseudonym + result[entity.end :]
        return result


def _without_overlaps(text: str, entities: list[DetectedEntity]) -> list[DetectedEntity]:
    """Merge any overlapping spans into their union span."""
    if len(entities) <= 1:
        return entities

    ordered = sorted(entities, key=lambda e: (e.start, -(e.end - e.start), -e.score))
    merged: list[DetectedEntity] = []
    cur = ordered[0]
    for nxt in ordered[1:]:
        if nxt.start < cur.end:
            new_end = max(cur.end, nxt.end)
            if new_end > cur.end:
                cur = DetectedEntity(
                    entity_type=cur.entity_type,
                    text=text[cur.start : new_end],
                    start=cur.start,
                    end=new_end,
                    score=max(cur.score, nxt.score),
                    source=cur.source,
                )
        else:
            merged.append(cur)
            cur = nxt
    merged.append(cur)
    return merged

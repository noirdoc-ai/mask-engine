from __future__ import annotations

from noirdoc.detection.base import DetectedEntity
from noirdoc.pseudonymization.engine import PseudonymizationEngine
from noirdoc.pseudonymization.mapper import PseudonymMapper


def _ent(entity_type: str, text: str, start: int, end: int) -> DetectedEntity:
    return DetectedEntity(
        entity_type=entity_type,
        text=text,
        start=start,
        end=end,
        score=0.9,
        source="test",
    )


def test_multiple_entities():
    text = "Max Müller wohnt in Berlin und ist unter max.mueller@test.de erreichbar."
    entities = [
        _ent("PERSON", "Max Müller", 0, 10),
        _ent("LOCATION", "Berlin", 20, 26),
        _ent("EMAIL", "max.mueller@test.de", 41, 60),
    ]
    mapper = PseudonymMapper()
    engine = PseudonymizationEngine()
    result = engine.pseudonymize(text, entities, mapper)
    assert "<<PERSON_1>>" in result
    assert "<<LOCATION_1>>" in result
    assert "<<EMAIL_1>>" in result
    assert "Max Müller" not in result
    assert "Berlin" not in result
    assert "max.mueller@test.de" not in result


def test_consistent_pseudonyms_same_name():
    text = "Max Müller und Lisa Schmidt kennen Max Müller."
    entities = [
        _ent("PERSON", "Max Müller", 0, 10),
        _ent("PERSON", "Lisa Schmidt", 15, 27),
        _ent("PERSON", "Max Müller", 35, 45),
    ]
    mapper = PseudonymMapper()
    engine = PseudonymizationEngine()
    result = engine.pseudonymize(text, entities, mapper)
    assert result == "<<PERSON_1>> und <<PERSON_2>> kennen <<PERSON_1>>."


def test_empty_text():
    mapper = PseudonymMapper()
    engine = PseudonymizationEngine()
    assert engine.pseudonymize("", [], mapper) == ""


def test_no_entities():
    mapper = PseudonymMapper()
    engine = PseudonymizationEngine()
    assert engine.pseudonymize("Hallo Welt", [], mapper) == "Hallo Welt"


def test_single_entity():
    text = "Hallo Max!"
    entities = [_ent("PERSON", "Max", 6, 9)]
    mapper = PseudonymMapper()
    engine = PseudonymizationEngine()
    result = engine.pseudonymize(text, entities, mapper)
    assert result == "Hallo <<PERSON_1>>!"


def test_offsets_preserved_with_replacement():
    """Replacing back-to-front ensures offsets stay correct."""
    text = "AB CD EF"
    entities = [
        _ent("PERSON", "AB", 0, 2),
        _ent("PERSON", "EF", 6, 8),
    ]
    mapper = PseudonymMapper()
    engine = PseudonymizationEngine()
    result = engine.pseudonymize(text, entities, mapper)
    assert "<<PERSON_1>>" in result
    assert "<<PERSON_2>>" in result
    assert "CD" in result


def test_overlapping_entities_same_start_do_not_corrupt():
    """#34: IBAN and a nested LOCATION at the same start produced '<<LOCATION_1>>AN_1>>'."""
    text = "IBAN: DE89 3704 0044 0532 0130 00"
    entities = [
        _ent("IBAN", text[6:], 6, len(text)),
        _ent("LOCATION", "DE89", 6, 10),
    ]
    mapper = PseudonymMapper()
    result = PseudonymizationEngine().pseudonymize(text, entities, mapper)
    assert result == "IBAN: <<IBAN_1>>"
    assert mapper.reverse_lookup("<<IBAN_1>>") == text[6:]


def test_overlapping_entity_starting_later_does_not_leak():
    """An inner span starting later must not shift the outer replacement and leak text."""
    text = "Termin bei Praxis Dr. Li Wagner."
    entities = [
        _ent("ORGANIZATION", "Praxis Dr. Li Wagner", 11, 31),
        _ent("PERSON", "Dr. Li", 18, 24),
    ]
    result = PseudonymizationEngine().pseudonymize(text, entities, PseudonymMapper())
    assert result == "Termin bei <<ORGANIZATION_1>>."
    assert "Wagner" not in result


def test_partially_overlapping_entities_are_merged():
    """Partially overlapping spans are replaced as one span covering both."""
    text = "Max Müller GmbH Berlin"
    entities = [
        _ent("PERSON", "Max Müller", 0, 10),
        _ent("ORGANIZATION", "Müller GmbH", 4, 15),
    ]
    mapper = PseudonymMapper()
    result = PseudonymizationEngine().pseudonymize(text, entities, mapper)
    assert result == "<<PERSON_1>> Berlin"
    assert mapper.reverse_lookup("<<PERSON_1>>") == "Max Müller GmbH"

import pytest
from logpipe.learning.similarity import find_alias
from logpipe.registry import TemplateEntry


def _entry(template_id: int, *tokens: str) -> TemplateEntry:
    """Create TemplateEntry"""

    return TemplateEntry(template_id = template_id, tokens = tokens, keys = ())


def test_opposite_meanings_never_merge():
    existing_door = [_entry(5, "Door", "sensor", "contact=open")]
    candidate_door = ("Door", "sensor", "contact=closed")

    assert find_alias(candidate = existing_door, existing_templates = candidate_door) is None

    existing_lock = [_entry(5, "Door", "sensor", "contact=open")]
    candidate_lock = ("Door", "sensor", "contact=closed")

    assert find_alias(candidate = candidate_lock, existing = existing_lock) is None


def test_values_only_diff_reuses_original_id():
    existing = [_entry(8, "Thermostat", "temp=<VAR:MEASURE>", "mode=auto")]
    candidate = ("Thermostat", "temp=<VAR:MEASURE>", "mode=auto", "fw=<VAR:MEASURE>")

    assert find_alias(candidate = candidate, existing = existing) == 8


def test_unrelated_templates_return_none():
    existing = [_entry(8, "Thermostat", "temp=<VAR:MEASURE>")]
    candidate = ("MQTT", "publish", "topic=home/events")

    assert find_alias(candidate = candidate, existing = existing) is None
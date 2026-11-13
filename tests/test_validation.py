import pytest
from pydantic import ValidationError
from models import Analysis
from pipeline import readable_transcript, file_id
from models import Segment

def test_category_rejected():
    with pytest.raises(ValidationError):
        Analysis(
            summary="test", goal="test", result="test",
            category="Несуществующая категория",
            evidence=[], next_action="test",confidence="низкая",speaker_roles={}
        )

def test_transcript_roles():
    seg = [Segment(start=0,end=2,speaker="A",text="Здравствуйте")]
    assert "Менеджер (A)" in readable_transcript(seg, {"A":"Менеджер"})

def test_id_deterministic():
    assert file_id(b"abc") == file_id(b"abc")
    assert file_id(b"abc") != file_id(b"abd")

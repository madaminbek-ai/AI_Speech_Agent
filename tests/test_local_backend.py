import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests

import local_backend as local
import pipeline
from models import Analysis, Segment


def analysis(**overrides):
    fields = dict(summary="Обсудили услугу. Клиент уточнил цену. Договоренностей нет.",
                  goal="Уточнить цену", result="Получена информация", category="Просто поинтересовался",
                  evidence=["Сколько стоит?"], next_action="Не согласовано", confidence="высокая",
                  speaker_roles={"A": "Клиент"})
    return Analysis(**(fields | overrides))


def segments():
    return [Segment(start=0, end=1, speaker="A", text="Сколько стоит?")]


def test_word_alignment_splits_at_actual_speaker_change():
    utterance = SimpleNamespace(words=[
        SimpleNamespace(start=0, end=0.4, word="Здравствуйте."),
        SimpleNamespace(start=0.5, end=0.9, word=" Удобно?"),
        SimpleNamespace(start=1.1, end=1.5, word=" Да.")])
    result = local.align_words([utterance], [(0, 1, "A"), (1, 2, "B")])
    assert [(s.speaker, s.text) for s in result] == [("A", "Здравствуйте. Удобно?"), ("B", "Да.")]
    assert result[0].end == 0.9
    assert result[1].start == 1.1


def test_speaker_overlap_sums_turns_and_handles_ambiguity():
    assert local.speaker_for_interval(0, 2, [(0, .7, "A"), (.8, 1.5, "A"), (0, 1, "B")]) == "A"
    assert local.speaker_for_interval(5, 6, [(0, 1, "A")]) == "UNKNOWN"
    assert local.speaker_for_interval(0, 1, [(0, 1, "A"), (0, 1, "B")]) == "UNKNOWN"


@pytest.mark.parametrize("url", ["https://api.example.com", "http://192.168.1.1:11434", "http://localhost@evil.test", "http://localhost:11434/api"])
def test_local_mode_rejects_nonlocal_endpoints(monkeypatch, url):
    monkeypatch.setenv("OLLAMA_BASE_URL", url)
    with pytest.raises(ValueError, match="localhost"):
        local.ollama_url()


def test_ollama_uses_schema_and_unloads_model(monkeypatch):
    request = Mock(return_value={"message": {"content": analysis().model_dump_json()}, "done_reason": "stop"})
    monkeypatch.setattr(local, "ollama_request", request)
    result = pipeline.analyze(None, segments(), backend="local")
    payload = request.call_args.args[1]
    assert payload["format"] == Analysis.model_json_schema()
    assert payload["stream"] is False
    assert payload["keep_alive"] == 0
    assert result.evidence == ["Сколько стоит?"]


def test_local_mode_never_falls_back_to_paid_api(monkeypatch):
    request = Mock(side_effect=RuntimeError("Ollama unavailable"))
    monkeypatch.setattr(local, "ollama_request", request)
    paid_client = Mock()
    with pytest.raises(RuntimeError, match="unavailable"):
        pipeline.analyze(paid_client, segments(), backend="local")
    assert not paid_client.mock_calls


def test_bad_json_is_rejected(monkeypatch):
    monkeypatch.setattr(local, "ollama_request", Mock(return_value={"message": {"content": "{}"}}))
    with pytest.raises(ValueError, match="формата"):
        pipeline.analyze(None, segments(), backend="local")


def test_truncated_generation_is_rejected(monkeypatch):
    monkeypatch.setattr(local, "ollama_request", Mock(return_value={"done_reason": "length"}))
    with pytest.raises(ValueError, match="оборвался"):
        pipeline.analyze(None, segments(), backend="local")


def test_long_transcript_is_not_sent_or_silently_cut(monkeypatch):
    monkeypatch.setenv("LOCAL_MAX_TRANSCRIPT_CHARS", "5")
    request = Mock()
    monkeypatch.setattr(local, "ollama_request", request)
    with pytest.raises(ValueError, match="слишком длинный"):
        pipeline.analyze(None, segments(), backend="local")
    request.assert_not_called()


def test_invalid_evidence_is_removed_and_unknown_role_is_preserved():
    result = pipeline.validate_analysis(analysis(evidence=[" ", "Выдуманная цитата"],
        speaker_roles={"UNKNOWN": "Менеджер", "invented": "Клиент"}),
        [Segment(start=0, end=1, speaker="UNKNOWN", text="Здравствуйте")])
    assert result.evidence == []
    assert result.confidence == "низкая"
    assert result.speaker_roles == {"UNKNOWN": "Неизвестно"}
    assert result.warnings


def test_evidence_cannot_join_different_speakers():
    result = pipeline.validate_analysis(analysis(evidence=["Да Нет"]), [
        Segment(start=0, end=1, speaker="A", text="Да"),
        Segment(start=1, end=2, speaker="B", text="Нет"),
    ])
    assert result.evidence == []
    assert result.confidence == "низкая"


@pytest.mark.parametrize("body", [[], None, "unexpected", {"error": "private server details"}])
def test_unexpected_ollama_response_has_clear_error(monkeypatch, body):
    response = Mock(status_code=200)
    response.json.return_value = body
    session = Mock()
    session.get.return_value = response
    context = Mock()
    context.__enter__ = Mock(return_value=session)
    context.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(local.requests, "Session", Mock(return_value=context))
    with pytest.raises(RuntimeError, match="Ollama") as error:
        local.ollama_request("/api/tags")
    assert "private server details" not in str(error.value)


def test_connection_failure_gives_actionable_error(monkeypatch):
    session = Mock()
    session.get.side_effect = requests.ConnectionError()
    context = Mock()
    context.__enter__ = Mock(return_value=session)
    context.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(local.requests, "Session", Mock(return_value=context))
    with pytest.raises(RuntimeError, match="ollama serve"):
        local.ollama_request("/api/tags")
    assert session.trust_env is False


def test_local_transcription_passes_cpu_and_word_timestamps(monkeypatch, tmp_path):
    import sys
    import numpy as np
    audio = np.zeros(16000, dtype=np.float32)
    word = SimpleNamespace(start=0, end=1, word="Привет")
    model = Mock()
    model.transcribe.return_value = (iter([SimpleNamespace(words=[word])]), SimpleNamespace())
    factory = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=factory))
    monkeypatch.setitem(sys.modules, "faster_whisper.audio", SimpleNamespace(decode_audio=lambda *a, **kw: audio))
    monkeypatch.setattr(local, "diarize_audio", lambda a: [(0, 1, "A")])
    result, duration = local.transcribe_local(tmp_path / "call.mp3")
    assert duration == 1
    assert result[0].speaker == "A"
    assert factory.call_args.kwargs["device"] == "cpu"
    assert factory.call_args.kwargs["compute_type"] == "int8"
    assert model.transcribe.call_args.kwargs["word_timestamps"] is True


def test_openai_backend_remains_explicit_and_available():
    paid = Mock()
    paid.beta.chat.completions.parse.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(parsed=analysis()))])
    result = pipeline.analyze(paid, segments(), backend="openai")
    assert result.category == "Просто поинтересовался"
    paid.beta.chat.completions.parse.assert_called_once()


def test_missing_hf_token_reports_setup_instead_of_transcribing(monkeypatch):
    monkeypatch.setattr(local.importlib.util, "find_spec", lambda name: object())
    monkeypatch.delenv("HF_TOKEN", raising=False)
    request = Mock()
    monkeypatch.setattr(local, "ollama_request", request)
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        local.check_local_setup()
    request.assert_not_called()


def test_diarizer_uses_waveform_and_exclusive_annotation(monkeypatch):
    import sys
    waveform = Mock()
    tensor = Mock()
    tensor.unsqueeze.return_value = waveform
    torch = SimpleNamespace(from_numpy=Mock(return_value=tensor), set_num_threads=Mock())
    turn = SimpleNamespace(start=0, end=1)
    annotation = SimpleNamespace(itertracks=Mock(return_value=[(turn, None, "A")]))
    diarizer = Mock(return_value=SimpleNamespace(exclusive_speaker_diarization=annotation))
    factory = Mock(return_value=diarizer)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "pyannote.audio", SimpleNamespace(Pipeline=SimpleNamespace(from_pretrained=factory)))
    monkeypatch.setenv("HF_TOKEN", "test-token")
    turns = local.diarize_audio([0.0])
    assert turns == [(0.0, 1.0, "A")]
    assert diarizer.call_args.args[0] == {"waveform": waveform, "sample_rate": 16000}
    assert factory.call_args.kwargs["token"] == "test-token"
    assert local.os.environ["PYANNOTE_METRICS_ENABLED"] == "0"

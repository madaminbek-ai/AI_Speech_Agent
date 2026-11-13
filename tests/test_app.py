from io import BytesIO
from pathlib import Path
from unittest.mock import Mock

from streamlit.testing.v1 import AppTest

import local_backend
import pipeline
import sheets
from models import Analysis, Segment


class Upload(BytesIO):
    name = "call.wav"


def test_free_mode_persists_deduplicates_and_retries_export(monkeypatch):
    import streamlit
    monkeypatch.setenv("AI_BACKEND", "local")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(streamlit, "file_uploader", lambda *a, **kw: [Upload(b"audio"), Upload(b"audio")])
    monkeypatch.setattr(local_backend, "check_local_setup", Mock())
    transcribe = Mock(return_value=([Segment(start=0, end=1, speaker="A", text="Здравствуйте")], 1))
    analyze = Mock(return_value=Analysis(summary="Разговор. Приветствие. Итог неизвестен.", goal="Неизвестно",
        result="Неизвестно", category="Просто поинтересовался", evidence=["Здравствуйте"],
        next_action="Не согласовано", confidence="низкая", speaker_roles={"A": "Неизвестно"}))
    monkeypatch.setattr(pipeline, "transcribe", transcribe)
    monkeypatch.setattr(pipeline, "analyze", analyze)
    worksheet = Mock()
    connect = Mock(side_effect=[RuntimeError("network down"), worksheet])
    append = Mock(return_value=True)
    monkeypatch.setattr(sheets, "google_worksheet", connect)
    monkeypatch.setattr(sheets, "append_unique", append)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
    assert not app.exception
    app.checkbox[0].check().run()
    next(button for button in app.button if button.label == "🚀 Анализировать").click().run()
    assert not app.exception
    assert len(app.text_area) == 1
    transcribe.assert_called_once()
    assert transcribe.call_args.args[0] is None
    assert transcribe.call_args.kwargs["backend"] == "local"
    assert any("экспорт не завершен" in item.value for item in app.error)
    app.checkbox[0].uncheck().run()
    assert len(app.text_area) == 1
    next(button for button in app.button if button.label == "Отправить в Google Sheets").click().run()
    assert not app.exception
    append.assert_called_once()
    next(button for button in app.button if button.label == "🚀 Анализировать").click().run()
    transcribe.assert_called_once()
    analyze.assert_called_once()
    assert len(app.text_area) == 1

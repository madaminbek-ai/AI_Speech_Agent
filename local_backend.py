from __future__ import annotations

import gc
import importlib.util
import json
import os
from pathlib import Path
from urllib.parse import urlparse

import requests
from pydantic import ValidationError

from models import Analysis, Segment

DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"


def ollama_url() -> str:
    url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    parsed = urlparse(url)

    if (parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
            or parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment):
        raise ValueError("Локальный режим: OLLAMA_BASE_URL должен указывать на localhost (http).")
    return url


def ollama_request(endpoint: str, payload: dict | None = None, timeout: float = 10) -> dict:
    with requests.Session() as session:
        session.trust_env = False
        try:
            if payload is None:
                response = session.get(ollama_url() + endpoint, timeout=(5, timeout), allow_redirects=False)
            else:
                response = session.post(ollama_url() + endpoint, json=payload,
                                        timeout=(5, timeout), allow_redirects=False)
        except requests.Timeout:
            raise RuntimeError("Ollama не ответила вовремя. Уменьшите модель или увеличьте OLLAMA_TIMEOUT.") from None
        except requests.ConnectionError:
            raise RuntimeError("Ollama не запущена. Установите Ollama и выполните: ollama serve") from None
        if response.status_code == 404:
            model = os.getenv("OLLAMA_MODEL", "qwen3:4b")
            raise RuntimeError(f"Модель Ollama не найдена. Выполните: ollama pull {model}")
        if response.status_code != 200:
            raise RuntimeError(f"Ошибка локальной Ollama (HTTP {response.status_code}). Проверьте журнал Ollama.")
        try:
            result = response.json()
        except ValueError:
            raise RuntimeError("Ollama вернула некорректный JSON.") from None
        if not isinstance(result, dict):
            raise RuntimeError("Ollama вернула JSON неожиданного формата.")
        if result.get("error"):
            raise RuntimeError("Ollama сообщила об ошибке. Проверьте журнал локального сервера.")
        return result


def check_local_setup() -> None:
    missing = []
    for module in ("faster_whisper", "torch", "pyannote.audio"):
        try:
            present = importlib.util.find_spec(module) is not None
        except (ModuleNotFoundError, ValueError):
            present = False
        if not present:
            missing.append(module)
    if missing:
        raise RuntimeError("Не установлены локальные зависимости: " + ", ".join(missing)
                           + ". Выполните: python -m pip install -r requirements-local.txt")
    if not os.getenv("HF_TOKEN", "").strip():
        raise RuntimeError("Добавьте HF_TOKEN в .env и примите условия модели "
                           "https://huggingface.co/pyannote/speaker-diarization-community-1. "
                           "Это бесплатный токен Hugging Face, не OpenAI.")
    model = os.getenv("OLLAMA_MODEL", "qwen3:4b")
    tags = ollama_request("/api/tags")
    names = {item.get("name") for item in tags.get("models", [])}
    if model not in names and model + ":latest" not in names:
        raise RuntimeError(f"Сначала загрузите модель: ollama pull {model}")


def speaker_for_interval(start: float, end: float, turns: list[tuple[float, float, str]]) -> str:

    overlap: dict[str, float] = {}
    for turn_start, turn_end, speaker in turns:
        amount = max(0.0, min(end, turn_end) - max(start, turn_start))
        overlap[speaker] = overlap.get(speaker, 0.0) + amount
    if not overlap or max(overlap.values()) <= 0:
        return "UNKNOWN"
    best = max(overlap, key=overlap.get)

    if sum(abs(value - overlap[best]) < 1e-8 for value in overlap.values()) > 1:
        return "UNKNOWN"
    return best


def align_words(recognized, turns: list[tuple[float, float, str]]) -> list[Segment]:
    result: list[Segment] = []
    for utterance in recognized:
        words = list(utterance.words or [])
        if not words:
            words = [utterance]
        for word in words:
            text = getattr(word, "word", None)
            if text is None:
                text = word.text
            if not text.strip():
                continue
            start, end = float(word.start), float(word.end)
            speaker = speaker_for_interval(start, end, turns)
            if result and result[-1].speaker == speaker and 0 <= start - result[-1].end <= 1.5:
                previous = result[-1]
                separator = "" if text.startswith(" ") or text[:1] in ",.!?:;" else " "
                previous.text += separator + text
                previous.end = max(previous.end, end)
            else:
                result.append(Segment(start=start, end=end, speaker=speaker, text=text.lstrip()))
    return result


def diarize_audio(audio) -> list[tuple[float, float, str]]:
    os.environ["PYANNOTE_METRICS_ENABLED"] = "0"
    import torch
    from pyannote.audio import Pipeline


    os.environ["PYANNOTE_METRICS_ENABLED"] = "0"
    torch.set_num_threads(int(os.getenv("LOCAL_CPU_THREADS", "4")))
    pipeline = None
    try:
        pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, token=os.getenv("HF_TOKEN"))
        if pipeline is None:
            raise RuntimeError("pipeline unavailable")

        output = pipeline({"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": 16000})
        annotation = output.exclusive_speaker_diarization
        return [(float(turn.start), float(turn.end), str(speaker))
                for turn, _, speaker in annotation.itertracks(yield_label=True)]
    except Exception as exc:

        cause = exc
        seen = set()
        while cause is not None and id(cause) not in seen:
            seen.add(id(cause))
            status = getattr(getattr(cause, "response", None), "status_code", None)
            if status in (401, 403):
                raise RuntimeError(
                    f"Hugging Face отказал в доступе к pyannote (HTTP {status}). "
                    "Войдите в аккаунт, которому принадлежит HF_TOKEN, и примите условия "
                    "https://huggingface.co/pyannote/speaker-diarization-community-1. "
                    "Проверьте, что токен действующий и разрешает чтение этой gated-модели. "
                    "После изменения .env перезапустите Streamlit."
                ) from None
            cause = cause.__cause__ or cause.__context__

        raise RuntimeError("Не удалось запустить pyannote Community-1. Проверьте HF_TOKEN, "
                           "принятие условий модели, доступ к интернету при первой загрузке "
                           "и свободную память. Детали установки — в README.md.") from None
    finally:
        del pipeline
        gc.collect()


def transcribe_local(path: Path) -> tuple[list[Segment], float]:
    try:
        from faster_whisper import WhisperModel
        from faster_whisper.audio import decode_audio
    except ImportError:
        raise RuntimeError("Установите зависимости: python -m pip install -r requirements-local.txt") from None
    try:
        audio = decode_audio(str(path), sampling_rate=16000)
    except Exception:
        raise ValueError("Файл не удалось декодировать как аудио. Проверьте формат и содержимое.") from None
    duration = len(audio) / 16000
    if duration <= 0:
        raise ValueError("Аудио пустое.")
    if duration > float(os.getenv("LOCAL_MAX_AUDIO_SECONDS", "1800")):
        raise ValueError("Запись длиннее LOCAL_MAX_AUDIO_SECONDS. Разделите ее перед обработкой.")
    turns = diarize_audio(audio)
    model = None
    try:
        model = WhisperModel(os.getenv("WHISPER_MODEL", "small"), device="cpu", compute_type="int8",
                             cpu_threads=int(os.getenv("LOCAL_CPU_THREADS", "4")),
                             download_root=str(Path(__file__).parent / ".models" / "whisper"))
        language = os.getenv("WHISPER_LANGUAGE", "").strip() or None
        recognized, _ = model.transcribe(audio, language=language, beam_size=5,
                                         word_timestamps=True, vad_filter=True)
        segments = align_words(recognized, turns)
        if not segments:
            raise ValueError("Речь не распознана. Проверьте запись и WHISPER_LANGUAGE.")
        return segments, duration
    finally:
        del model
        del audio
        gc.collect()


def analyze_local(segments: list[Segment], system_prompt: str, raw: str) -> Analysis:

    if len(raw) > int(os.getenv("LOCAL_MAX_TRANSCRIPT_CHARS", "16000")):
        raise ValueError("Транскрипт слишком длинный для локального анализа. "
                         "Разделите запись или настройте LOCAL_MAX_TRANSCRIPT_CHARS и OLLAMA_NUM_CTX.")
    schema = Analysis.model_json_schema()
    response = ollama_request("/api/chat", {
        "model": os.getenv("OLLAMA_MODEL", "qwen3:4b"),
        "messages": [
            {"role": "system", "content": system_prompt + "\nВерни только JSON по схеме:\n"
             + json.dumps(schema, ensure_ascii=False)},
            {"role": "user", "content": "Проанализируй транскрипт:\n" + raw},
        ],
        "format": schema,
        "stream": False,
        "think": False,
        "keep_alive": 0,
        "options": {"temperature": 0, "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "16384")),
                    "num_predict": 2048},
    }, timeout=float(os.getenv("OLLAMA_TIMEOUT", "900")))
    if response.get("done_reason") == "length":
        raise ValueError("Ответ Ollama оборвался по лимиту генерации. Попробуйте более короткую запись.")
    try:
        result = Analysis.model_validate_json(response["message"]["content"])
    except (KeyError, TypeError, ValidationError):
        raise ValueError("Ollama не вернула отчет нужного формата. Попробуйте другую локальную модель.") from None
    if any(s.speaker == "UNKNOWN" for s in segments):
        result.warnings.append("Часть слов не удалось надежно сопоставить со спикером.")
        result.confidence = "низкая"
    return result

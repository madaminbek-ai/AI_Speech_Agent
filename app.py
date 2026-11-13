import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from pipeline import transcribe, analyze, file_id, readable_transcript, validate_file
from sheets import google_worksheet, make_row, append_unique, HEADERS

load_dotenv(Path(__file__).with_name(".env"))
st.set_page_config(page_title="AI_Speech_Agent", page_icon="🎙️", layout="wide")
st.title("🎙️ AI_Speech_Agent")
st.caption("Анализ звонков • Разделение спикеров • 6 бизнес-категорий • Google Sheets")

providers = ["local", "openai"]
default_provider = os.getenv("AI_BACKEND", "local")
backend = st.selectbox("Режим обработки", providers,
                       index=providers.index(default_provider) if default_provider in providers else 0,
                       format_func=lambda value: "Бесплатно на компьютере" if value == "local" else "OpenAI API (платно)")
if backend == "local":
    st.info("Локальный режим: faster-whisper + pyannote Community-1 + Ollama. "
            "OpenAI API не используется. При первом запуске загружаются модели. "
            "При экспорте текст отчета и транскрипт отправляются в Google Sheets.")
    with st.expander("Подготовка бесплатного режима"):
        st.markdown("1. Установите зависимости: `python -m pip install -r requirements-local.txt`.\n"
                    "2. Установите [Ollama](https://ollama.com/download/linux), "
                    "запустите `ollama serve` и загрузите модель: `ollama pull qwen3:4b`.\n"
                    "3. Примите условия [pyannote Community-1]"
                    "(https://huggingface.co/pyannote/speaker-diarization-community-1).\n"
                    "4. Создайте [Hugging Face Read token](https://huggingface.co/settings/tokens) "
                    "и запишите его в `.env` как `HF_TOKEN=...`.\n"
                    "5. Перезапустите Streamlit. Подробности — в README.md.")
    if st.button("Проверить локальную настройку"):
        try:
            from local_backend import check_local_setup
            check_local_setup()
            st.success("Зависимости, токен и модель Ollama настроены. Доступ к pyannote проверяется при загрузке модели.")
        except Exception as exc:
            st.error(str(exc))
else:
    st.info("В этом режиме аудио и транскрипт отправляются в OpenAI API. Нужен оплаченный API-баланс.")
st.caption("Загружайте записи, обработка которых разрешена. Роли определяются по содержанию; "
           "при неоднозначности выводится «Неизвестно».")

uploads = st.file_uploader("Выберите аудиозаписи (до 25 MB каждая)",
                           type=["mp3", "wav", "m4a", "mp4", "mpeg", "mpga", "ogg", "flac", "webm"],
                           accept_multiple_files=True)
publish = st.checkbox("Автоматически записать результаты в Google Sheets", value=False)
st.session_state.setdefault("reports", {})
reports = st.session_state["reports"]

if st.button("🚀 Анализировать", type="primary", disabled=not uploads):
    client = None
    try:
        if backend == "local":
            from local_backend import check_local_setup
            check_local_setup()
        else:
            from openai import OpenAI
            key = os.getenv("OPENAI_API_KEY", "")
            if not key or key == "your_openai_api_key":
                raise ValueError("Добавьте действующий OPENAI_API_KEY в .env или выберите бесплатный режим.")
            client = OpenAI(api_key=key, timeout=180.0, max_retries=2)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    profile = json.dumps({"backend": backend,
                          "whisper": os.getenv("WHISPER_MODEL", "small"),
                          "language": os.getenv("WHISPER_LANGUAGE", ""),
                          "ollama": os.getenv("OLLAMA_MODEL", "qwen3:4b"),
                          "openai_analysis": os.getenv("OPENAI_ANALYSIS_MODEL", "gpt-4.1-mini"),
                          "openai_audio": os.getenv("OPENAI_DIARIZATION_MODEL", "gpt-4o-transcribe-diarize")}, sort_keys=True)
    batch_keys = set()
    completed = []
    try:
        for i, upload in enumerate(uploads, 1):
            data = upload.getvalue()
            call_id = file_id(data)
            report_key = file_id((profile + call_id).encode())
            if report_key in batch_keys:
                st.info(f"{upload.name}: одинаковая запись уже есть в этой пачке, повтор пропущен.")
                continue
            batch_keys.add(report_key)
            if report_key in reports:
                st.info(f"{upload.name}: используется готовый результат из текущей сессии.")
                completed.append(report_key)
                continue
            with st.status(f"[{i}/{len(uploads)}] {upload.name}", expanded=True) as status:
                temp_path = None
                try:
                    with tempfile.NamedTemporaryFile(suffix=Path(upload.name).suffix, delete=False) as temp:
                        temp.write(data)
                        temp_path = Path(temp.name)
                    validate_file(temp_path)
                    st.write("1/3 · Распознавание речи и спикеров. На CPU это может занять несколько минут.")
                    segments, duration = transcribe(client, temp_path, backend=backend)
                    st.write("2/3 · Анализ и проверка доказательств")
                    result = analyze(client, segments, backend=backend)
                    transcript = readable_transcript(segments, result.speaker_roles)
                    reports[report_key] = {"call_id": call_id, "filename": upload.name, "backend": backend,
                                           "analysis": result.model_dump(), "transcript": transcript,
                                           "row": make_row(call_id, upload.name, duration, result, transcript),
                                           "exported": False}
                    completed.append(report_key)
                    status.update(label=f"✅ {upload.name}", state="complete")
                except Exception as exc:
                    st.error(f"Ошибка обработки {upload.name}: {exc}")
                    status.update(label=f"❌ {upload.name}", state="error")
                finally:
                    if temp_path is not None:
                        temp_path.unlink(missing_ok=True)
    finally:
        if client is not None:
            client.close()
    if publish and completed:
        try:
            ws = google_worksheet()
            for report_key in completed:
                report = reports[report_key]
                append_unique(ws, report["row"])
                report["exported"] = True
            st.success("3/3 · Результаты записаны в Google Sheets (существующие Call ID пропущены).")
        except Exception:
            st.error("Google Sheets: экспорт не завершен. Анализ сохранен в текущей сессии; "
                     "повторите отправку кнопкой ниже. Проверьте доступ и соединение.")


if reports:
    st.subheader("Результаты текущей сессии")
    st.caption("Результаты доступны при переключении элементов интерфейса. "
               "Перед закрытием/обновлением вкладки скачайте отчет или отправьте его в Sheets.")
    for report_key, report in reports.items():
        result = report["analysis"]
        with st.expander(f"📞 {report['filename']}: {result['category']}", expanded=True):
            st.caption("Бесплатный локальный режим" if report["backend"] == "local" else "OpenAI API")
            for label, field in [("Саммари", "summary"), ("Цель", "goal"), ("Результат", "result"),
                                 ("Следующее действие", "next_action"), ("Уверенность", "confidence")]:
                st.write(f"**{label}:**", result[field])
            st.write("**Доказательства:**", result["evidence"])
            if result["warnings"]:
                st.warning("; ".join(result["warnings"]))
            st.text_area("Транскрипт с таймкодами", report["transcript"], height=220,
                         key=f"transcript_{report_key}", disabled=True)
            st.download_button("Скачать JSON", json.dumps(
                {"call_id": report["call_id"], "backend": report["backend"],
                 "analysis": result, "transcript": report["transcript"]}, ensure_ascii=False, indent=2),
                file_name=f"{report['call_id']}.json", mime="application/json", key=f"json_{report_key}")
            if report["exported"]:
                st.success("Google Sheets: экспорт выполнен или запись с таким Call ID уже существует.")
            elif st.button("Отправить в Google Sheets", key=f"export_{report_key}"):
                try:
                    inserted = append_unique(google_worksheet(), report["row"])
                    report["exported"] = True
                    st.success("Записано" if inserted else "Call ID уже есть в таблице — повторная строка не создана.")
                except Exception:
                    st.error("Не удалось отправить отчет. Проверьте доступ к Google Sheets и соединение.")
    df = pd.DataFrame([report["row"] for report in reports.values()], columns=HEADERS)
    st.dataframe(df.drop(columns=["Транскрипт"]), width="stretch")

    safe_df = df.map(lambda value: "'" + value if isinstance(value, str)
                     and value.lstrip().startswith(("=", "+", "-", "@")) else value)
    st.download_button("⬇️ Экспорт CSV (UTF-8 BOM)", safe_df.to_csv(index=False).encode("utf-8-sig"),
                       file_name="speech_analytics_report.csv", mime="text/csv")

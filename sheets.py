import os
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import gspread
from google.oauth2.service_account import Credentials

HEADERS = [
    "Дата обработки", "Call ID", "Файл", "Длительность (сек)",
    "Категория", "Саммари", "Цель", "Результат", "Следующее действие",
    "Уверенность", "Недостаточно данных", "Цитаты", "Предупреждения", "Транскрипт",
]

def google_worksheet():
    file_path = Path(os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "service_account.json"))
    if not file_path.is_absolute():
        file_path = Path(__file__).resolve().parent / file_path
    sheet_id = os.getenv("GOOGLE_SHEET_ID")
    if not sheet_id:
        raise RuntimeError("Не задан GOOGLE_SHEET_ID")
    credentials = Credentials.from_service_account_file(
        file_path, scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    client = gspread.authorize(credentials)
    book = client.open_by_key(sheet_id)
    try:
        worksheet = book.worksheet("Calls")
    except gspread.WorksheetNotFound:
        worksheet = book.add_worksheet(title="Calls", rows=1000, cols=20)
    headers = worksheet.row_values(1)
    if headers != HEADERS:
        if headers or worksheet.get_all_values():
            raise RuntimeError("Лист Calls имеет другую структуру. Используйте пустой лист или другую таблицу.")
        worksheet.update(range_name="A1:N1", values=[HEADERS], value_input_option="RAW")
        worksheet.freeze(rows=1)
    return worksheet

def make_row(call_id, filename, duration, result, transcript):
    return [
        datetime.now(ZoneInfo("Asia/Tashkent")).isoformat(timespec="seconds"),
        call_id, filename, duration if duration is not None else "",
        result.category, result.summary, result.goal, result.result,
        result.next_action, result.confidence,
        "Да" if result.insufficient_data else "Нет",
        " | ".join(result.evidence), " | ".join(result.warnings), transcript,
    ]

def append_unique(worksheet, row):

    existing = set(worksheet.col_values(2)[1:])
    if row[1] in existing:
        return False

    worksheet.append_row(row, value_input_option="RAW")
    return True

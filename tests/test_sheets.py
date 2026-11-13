from pathlib import Path
from unittest.mock import Mock

import pytest

import sheets


def worksheet(monkeypatch, headers, values=None):
    ws = Mock()
    ws.row_values.return_value = headers
    ws.get_all_values.return_value = values or []
    credentials = Mock()
    monkeypatch.setattr(sheets.Credentials, "from_service_account_file", credentials)
    client = Mock()
    client.open_by_key.return_value.worksheet.return_value = ws
    monkeypatch.setattr(sheets.gspread, "authorize", Mock(return_value=client))
    monkeypatch.setenv("GOOGLE_SHEET_ID", "test-sheet")
    monkeypatch.setenv("GOOGLE_SERVICE_ACCOUNT_FILE", "service_account.json")
    return ws, credentials


@pytest.mark.parametrize("headers, values", [(["Existing data"], []), ([], [[], ["Keep me"]])])
def test_existing_sheet_with_other_structure_is_not_modified(monkeypatch, headers, values):
    ws, _ = worksheet(monkeypatch, headers, values)
    with pytest.raises(RuntimeError, match="другую структуру"):
        sheets.google_worksheet()
    ws.update.assert_not_called()
    ws.freeze.assert_not_called()


def test_empty_sheet_is_initialized_and_credentials_are_project_relative(monkeypatch, tmp_path):
    ws, credentials = worksheet(monkeypatch, [])
    monkeypatch.chdir(tmp_path)
    assert sheets.google_worksheet() is ws
    assert credentials.call_args.args[0] == Path(sheets.__file__).resolve().parent / "service_account.json"
    ws.update.assert_called_once_with(range_name="A1:N1", values=[sheets.HEADERS], value_input_option="RAW")


def test_matching_headers_are_preserved(monkeypatch):
    ws, _ = worksheet(monkeypatch, sheets.HEADERS)
    sheets.google_worksheet()
    ws.update.assert_not_called()


def test_duplicate_call_is_not_appended():
    ws = Mock()
    ws.col_values.return_value = ["Call ID", "existing-id"]
    assert sheets.append_unique(ws, ["date", "existing-id"]) is False
    ws.append_row.assert_not_called()

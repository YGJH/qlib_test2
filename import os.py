import os
import sys
import pickle
import runpy
from pathlib import Path
import pandas as pd
import pytest

# test_main.py


# Helper to write a dummy get_yahoo_data.py next to main.py
def write_dummy_get_yahoo(script_dir):
    script = script_dir / "get_yahoo_data.py"
    script.write_text(
        "if __name__ == '__main__':\n"
        "    # dummy downloader\n"
        "    pass\n"
    )
    return script

@pytest.fixture(autouse=True)
def isolate_home(monkeypatch, tmp_path):
    # Redirect HOME so ~/.qlib points into tmp_path
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path

def run_main_and_catch(tmp_dir, capsys):
    # Execute main.py in its own namespace
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(tmp_dir / "main.py"), run_name="__main__")
    return exc

def test_missing_instruments(tmp_path, isolate_home, capsys):
    # Setup script directory
    script_dir = tmp_path
    write_dummy_get_yahoo(script_dir)
    # Create calendar file only
    cal_dir = tmp_path / ".qlib" / "qlib_data" / "us_stocks_yahoo" / "calendars"
    cal_dir.mkdir(parents=True)
    # non-empty calendar
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    pickle.dump(dates, (cal_dir / "day.pkl").open("wb"))
    # Copy main.py into tmp dir
    (script_dir / "main.py").write_text(Path(__file__).parent.joinpath("main.py").read_text())
    # Run and expect error about instruments
    exc = run_main_and_catch(script_dir, capsys)
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "✗ 找不到 instruments 文件" in out

def test_missing_calendar(tmp_path, isolate_home, capsys):
    script_dir = tmp_path
    write_dummy_get_yahoo(script_dir)
    # Create instruments file only
    instr_dir = tmp_path / ".qlib" / "qlib_data" / "us_stocks_yahoo" / "instruments"
    instr_dir.mkdir(parents=True)
    (instr_dir / "all.txt").write_text("AAPL\n")
    # Copy main.py
    (script_dir / "main.py").write_text(Path(__file__).parent.joinpath("main.py").read_text())
    # Run and expect error about calendar
    exc = run_main_and_catch(script_dir, capsys)
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "✗ 找不到日历文件" in out

def test_successful_run(tmp_path, isolate_home, capsys):
    script_dir = tmp_path
    write_dummy_get_yahoo(script_dir)
    # Create instruments
    base = tmp_path / ".qlib" / "qlib_data" / "us_stocks_yahoo"
    instr_dir = base / "instruments"
    instr_dir.mkdir(parents=True)
    (instr_dir / "all.txt").write_text("AAPL\nMSFT\n")
    # Create calendar
    cal_dir = base / "calendars"
    cal_dir.mkdir()
    dates = pd.date_range("2020-01-01", periods=3, freq="D")
    pickle.dump(dates, (cal_dir / "day.pkl").open("wb"))
    # Copy main.py
    (script_dir / "main.py").write_text(Path(__file__).parent.joinpath("main.py").read_text())
    # Run without expecting SystemExit
    runpy.run_path(str(script_dir / "main.py"), run_name="__main__")
    out = capsys.readouterr().out
    assert "程序執行完成！" in out
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data_ingestion import clean_text, process_pdfs  # noqa: E402


def test_clean_text_collapses_whitespace():
    assert clean_text("  Attention   is\n\nall\tyou need  ") == "Attention is all you need"


def test_clean_text_removes_numeric_citations():
    assert clean_text("as shown before [12] and later [3].") == "as shown before and later ."


def test_clean_text_keeps_other_brackets():
    assert clean_text("the set [a, b] holds") == "the set [a, b] holds"


def test_process_pdfs_writes_one_text_file_per_pdf(tmp_path):
    fitz = pytest.importorskip("fitz")
    raw, out = tmp_path / "raw", tmp_path / "out"
    raw.mkdir()
    for name, body in [("paper_a", "Transformers   use [1] self-attention."), ("paper_b", "Second paper.")]:
        doc = fitz.open()
        doc.new_page().insert_text((72, 72), body)
        doc.save(raw / f"{name}.pdf")
        doc.close()

    process_pdfs(str(raw), str(out))

    assert sorted(p.name for p in out.glob("*.txt")) == ["paper_a.txt", "paper_b.txt"]
    assert (out / "paper_a.txt").read_text() == "Transformers use self-attention."

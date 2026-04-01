"""
Test funzionali per le euristiche locali di filtraggio italiano.
"""

import json
from types import SimpleNamespace

from run_local_it import (
    basic_normalize,
    detect_noise,
    is_italian,
    italian_heuristic_score,
    make_filter,
)


class TestRealDatasetHeuristics:
    def test_training_docs_are_recognized_as_clean_italian(self, training_good_docs):
        assert training_good_docs, "Il dataset reale di training non deve essere vuoto"

        for doc in training_good_docs[:5]:
            cleaned = basic_normalize(doc["text"])
            score = italian_heuristic_score(cleaned)
            noise, reason = detect_noise(cleaned)
            ok, confidence, backend = is_italian(cleaned)

            assert cleaned
            assert noise is False, f"{doc['id']} è stato marcato come rumore: {reason}"
            assert score >= 0.55, f"Score troppo basso per {doc['id']}: {score}"
            assert ok is True, f"{doc['id']} non riconosciuto come italiano ({backend}, {confidence})"

    def test_filter_keeps_real_training_doc_without_writing_rejects(self, training_good_docs, temp_dir):
        keep_doc = training_good_docs[0]
        reject_path = temp_dir / "rejects" / "kept.jsonl"
        filter_fn = make_filter(str(reject_path))

        accepted = filter_fn(SimpleNamespace(id=keep_doc["id"], text=keep_doc["text"]))

        assert accepted is True
        if reject_path.exists():
            assert reject_path.read_text(encoding="utf-8").strip() == ""


class TestRejectPaths:
    def test_filter_rejects_non_italian_text_and_records_reason(self, temp_dir):
        reject_path = temp_dir / "rejects" / "non_it.jsonl"
        filter_fn = make_filter(str(reject_path))
        doc = SimpleNamespace(
            id="english_doc",
            text="This page explains how to configure your account settings and billing information.",
        )

        accepted = filter_fn(doc)

        assert accepted is False
        payload = json.loads(reject_path.read_text(encoding="utf-8").strip())
        assert payload["id"] == "english_doc"
        assert payload["reason"].startswith("non_it:")
        assert "confidence" in payload

    def test_filter_rejects_noise_and_records_reason(self, temp_dir):
        reject_path = temp_dir / "rejects" / "noise.jsonl"
        filter_fn = make_filter(str(reject_path))
        doc = SimpleNamespace(
            id="boilerplate_doc",
            text="Accetta i cookie, iscriviti alla newsletter e gestisci le impostazioni di privacy.",
        )

        accepted = filter_fn(doc)

        assert accepted is False
        payload = json.loads(reject_path.read_text(encoding="utf-8").strip())
        assert payload["id"] == "boilerplate_doc"
        assert payload["reason"].startswith("noise:")


class TestNormalization:
    def test_basic_normalize_removes_web_artifacts(self):
        text = "<p>Scrivici a test@example.com oppure visita https://example.com</p>"

        cleaned = basic_normalize(text)

        assert "<p>" not in cleaned
        assert "@" not in cleaned
        assert "https://" not in cleaned
        assert "Scrivici" in cleaned

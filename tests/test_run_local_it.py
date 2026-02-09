"""
Test per le funzioni di filtering italiano in run_local_it.py
"""

import sys
from pathlib import Path

import pytest

# Aggiungi src al path per importare i moduli
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from run_local_it import (
    basic_normalize,
    char_stats,
    token_stats,
    bigram_stats,
    anti_language_stats,
    italian_heuristic_score,
    detect_noise,
    is_italian,
    normalize_apostrophes,
    ITALIAN_STOPWORDS,
    COMMON_IT_BIGRAMS,
)


# =============================================================================
# Test basic_normalize
# =============================================================================

class TestBasicNormalize:
    """Test per la funzione basic_normalize."""

    def test_remove_html_tags(self):
        """Verifica che i tag HTML vengano rimossi."""
        text = "<p>Questo è un <strong>testo</strong> di prova</p>"
        result = basic_normalize(text)
        assert "<" not in result
        assert ">" not in result
        assert "Questo" in result
        assert "testo" in result

    def test_remove_urls(self):
        """Verifica che gli URL vengano rimossi."""
        text = "Visita https://example.com per maggiori info"
        result = basic_normalize(text)
        assert "https://" not in result
        assert "example.com" not in result
        assert "Visita" in result

    def test_remove_emails(self):
        """Verifica che le email vengano rimosse."""
        text = "Contattami a test@example.com per info"
        result = basic_normalize(text)
        assert "@" not in result
        assert "test@example.com" not in result

    def test_normalize_whitespace(self):
        """Verifica che gli spazi multipli vengano normalizzati."""
        text = "Testo   con    molti     spazi"
        result = basic_normalize(text)
        assert "   " not in result
        assert "Testo con molti spazi" == result

    def test_normalize_apostrophes(self):
        """Verifica che gli apostrofi vengano normalizzati."""
        text = "L’acqua dell’oceano"
        result = basic_normalize(text)
        assert "’" not in result


# =============================================================================
# Test char_stats
# =============================================================================

class TestCharStats:
    """Test per la funzione char_stats."""

    def test_basic_stats(self):
        """Verifica le statistiche base sui caratteri."""
        text = "Ciao mondo"
        stats = char_stats(text)
        
        assert stats["total"] == 10
        assert stats["letters"] == 9  # 'Ciao mondo' senza spazio
        assert stats["spaces"] == 1
        assert stats["digits"] == 0

    def test_letter_ratio(self):
        """Verifica il calcolo del letter_ratio."""
        text = "abcde12345"  # 5 lettere, 5 numeri
        stats = char_stats(text)
        
        assert stats["letter_ratio"] == 0.5
        assert stats["digit_ratio"] == 0.5

    def test_empty_text(self):
        """Verifica il comportamento con testo vuoto."""
        text = ""
        stats = char_stats(text)
        
        # total è max(len, 1) quindi 1
        assert stats["total"] == 1

    def test_punctuation(self):
        """Verifica il conteggio della punteggiatura."""
        text = "Ciao! Come stai?"
        stats = char_stats(text)
        
        assert stats["punct"] > 0


# =============================================================================
# Test token_stats
# =============================================================================

class TestTokenStats:
    """Test per la funzione token_stats."""

    def test_stopword_detection(self):
        """Verifica il rilevamento delle stopword italiane."""
        text = "il cane della mia amica è molto carino"
        stats = token_stats(text)
        
        # 'il', 'della', 'mia', 'è', 'molto' sono stopwords
        assert stats["n_tokens"] > 0
        assert stats["stop_ratio"] > 0

    def test_apostrophe_handling(self):
        """Verifica la gestione degli apostrofi."""
        text = "L'acqua dell'oceano all'orizzonte"
        stats = token_stats(text)
        
        assert stats["n_tokens"] > 0
        # l', dell', all' sono stopwords
        assert stats["stop_ratio"] > 0

    def test_empty_text(self):
        """Verifica il comportamento con testo vuoto."""
        text = ""
        stats = token_stats(text)
        
        assert stats["n_tokens"] == 0
        assert stats["stop_ratio"] == 0.0

    def test_unique_ratio(self):
        """Verifica il calcolo dell'unique_ratio."""
        # Testo con ripetizioni
        text = "ciao ciao ciao mondo mondo"
        stats = token_stats(text)
        
        # 2 token unici su 5 totali = 0.4
        assert stats["unique_ratio"] == pytest.approx(0.4, rel=0.01)


# =============================================================================
# Test bigram_stats
# =============================================================================

class TestBigramStats:
    """Test per la funzione bigram_stats."""

    def test_italian_bigrams_detected(self):
        """Verifica che i bigrammi italiani vengano rilevati."""
        text = "Il libro di cui ti ho parlato è molto interessante"
        stats = bigram_stats(text)
        
        assert stats["n_bigrams"] > 0
        # "di cui" è un bigramma comune
        assert stats["bigram_ratio"] > 0

    def test_short_text(self):
        """Verifica il comportamento con testi molto corti."""
        text = "ciao"
        stats = bigram_stats(text)
        
        assert stats["n_bigrams"] == 0
        assert stats["bigram_ratio"] == 0.0

    def test_no_common_bigrams(self):
        """Verifica con testo senza bigrammi comuni."""
        text = "albero verde giallo rosso blu"
        stats = bigram_stats(text)
        
        assert stats["n_bigrams"] > 0
        # Nessun bigramma nella whitelist
        assert stats["bigram_ratio"] == 0.0


# =============================================================================
# Test anti_language_stats
# =============================================================================

class TestAntiLanguageStats:
    """Test per la funzione anti_language_stats."""

    def test_spanish_detection(self):
        """Verifica il rilevamento di parole spagnole."""
        text = "los gatos también quieren comer porque tienen hambre"
        stats = anti_language_stats(text)
        
        assert stats["anti_es_ratio"] > 0

    def test_portuguese_detection(self):
        """Verifica il rilevamento di parole portoghesi."""
        text = "você não pode fazer isso também porque não é certo"
        stats = anti_language_stats(text)
        
        assert stats["anti_pt_ratio"] > 0

    def test_french_detection(self):
        """Verifica il rilevamento di parole francesi."""
        text = "les enfants sont dans le parc avec leurs parents"
        stats = anti_language_stats(text)
        
        assert stats["anti_fr_ratio"] > 0

    def test_italian_no_anti(self):
        """Verifica che il testo italiano non abbia anti-language alto."""
        text = "I bambini sono nel parco con i loro genitori italiani"
        stats = anti_language_stats(text)
        
        # L'italiano non dovrebbe avere ratio alti per altre lingue
        assert stats["anti_es_ratio"] < 0.1
        assert stats["anti_pt_ratio"] < 0.1
        assert stats["anti_fr_ratio"] < 0.1


# =============================================================================
# Test italian_heuristic_score
# =============================================================================

class TestItalianHeuristicScore:
    """Test per la funzione italian_heuristic_score."""

    def test_high_score_for_italian(self, italian_texts):
        """Verifica che i testi italiani abbiano score alto."""
        for doc in italian_texts:
            score = italian_heuristic_score(doc["text"])
            assert score >= 0.5, f"Score troppo basso per {doc['id']}: {score}"

    def test_low_score_for_english(self):
        """Verifica che l'inglese abbia score basso."""
        text = "The quick brown fox jumps over the lazy dog multiple times today"
        score = italian_heuristic_score(text)
        assert score < 0.5

    def test_low_score_for_spanish(self):
        """Verifica che lo spagnolo abbia score basso."""
        text = "El rápido zorro marrón salta sobre el perro perezoso varias veces"
        score = italian_heuristic_score(text)
        assert score < 0.6

    def test_score_bounds(self):
        """Verifica che lo score sia sempre tra 0 e 1."""
        texts = [
            "Testo normale italiano",
            "!@#$%^&*()",
            "a" * 1000,
            "",
        ]
        for text in texts:
            score = italian_heuristic_score(text)
            assert 0.0 <= score <= 1.0


# =============================================================================
# Test detect_noise
# =============================================================================

class TestDetectNoise:
    """Test per la funzione detect_noise."""

    def test_empty_text(self):
        """Verifica il rilevamento di testo vuoto."""
        is_noise, reason = detect_noise("")
        assert is_noise is True
        assert reason == "empty"

    def test_short_text(self):
        """Verifica il rilevamento di testo troppo corto."""
        is_noise, reason = detect_noise("ab")
        assert is_noise is True
        assert reason == "too_short"

    def test_long_repetition(self):
        """Verifica il rilevamento di ripetizioni lunghe."""
        is_noise, reason = detect_noise("aaaaaaaaaaaaaaaaaa bbbbb")
        assert is_noise is True
        assert reason == "long_repetition"

    def test_code_detection(self):
        """Verifica il rilevamento di codice."""
        code_samples = [
            "function test() { return 42; }",
            "class MyClass { constructor() {} }",
            "def my_function():",
            "import os; import sys",
        ]
        for code in code_samples:
            is_noise, reason = detect_noise(code)
            assert is_noise is True, f"Non rilevato come codice: {code}"

    def test_boilerplate_detection(self):
        """Verifica il rilevamento di boilerplate."""
        text = "Accetta i cookie e iscriviti alla newsletter per ricevere aggiornamenti"
        is_noise, reason = detect_noise(text)
        assert is_noise is True
        assert reason == "boilerplate"

    def test_valid_italian_text(self, italian_texts):
        """Verifica che il testo italiano valido non sia rumore."""
        for doc in italian_texts:
            is_noise, reason = detect_noise(doc["text"])
            assert is_noise is False, f"Falso positivo per {doc['id']}: {reason}"


# =============================================================================
# Test is_italian
# =============================================================================

class TestIsItalian:
    """Test per la funzione is_italian."""

    def test_italian_accepted(self, italian_texts):
        """Verifica che i testi italiani vengano accettati."""
        for doc in italian_texts:
            ok, confidence, backend = is_italian(doc["text"])
            assert ok is True, f"Testo italiano rifiutato: {doc['id']} (conf={confidence}, backend={backend})"

    def test_english_rejected(self):
        """Verifica che l'inglese venga rifiutato."""
        text = ("The quick brown fox jumps over the lazy dog. This is a sample text "
                "in English that should be filtered out by the Italian language detector.")
        ok, confidence, backend = is_italian(text)
        assert ok is False

    def test_spanish_rejected(self):
        """Verifica che lo spagnolo venga rifiutato."""
        text = ("Hola, cómo estás? Hoy hace un día muy bonito para pasear por el parque. "
                "Los pájaros cantan y las flores están floreciendo en primavera.")
        ok, confidence, backend = is_italian(text)
        assert ok is False

    def test_returns_confidence(self):
        """Verifica che venga restituita una confidence valida."""
        text = "Questo è un testo in italiano molto semplice"
        ok, confidence, backend = is_italian(text)
        
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
        assert isinstance(backend, str)


# =============================================================================
# Test normalize_apostrophes
# =============================================================================

class TestNormalizeApostrophes:
    """Test per la funzione normalize_apostrophes."""

    def test_curly_to_straight(self):
        """Verifica la conversione degli apostrofi curvi."""
        text = "L’acqua dell’oceano"
        result = normalize_apostrophes(text)
        assert "’" not in result
        assert "'" in result

    def test_no_change_needed(self):
        """Verifica che non cambi testo già normalizzato."""
        text = "L’acqua dell’oceano"
        result = normalize_apostrophes(text)
        assert result.replace("'", " ") == text.replace("’", " ")


# =============================================================================
# Test costanti
# =============================================================================

class TestConstants:
    """Test per le costanti definite nel modulo."""

    def test_italian_stopwords_not_empty(self):
        """Verifica che la lista di stopwords non sia vuota."""
        assert len(ITALIAN_STOPWORDS) > 0

    def test_common_bigrams_not_empty(self):
        """Verifica che la lista di bigrammi non sia vuota."""
        assert len(COMMON_IT_BIGRAMS) > 0

    def test_stopwords_lowercase(self):
        """Verifica che le stopwords siano in minuscolo."""
        for word in ITALIAN_STOPWORDS:
            assert word == word.lower(), f"Stopword non in minuscolo: {word}"

    def test_common_stopwords_present(self):
        """Verifica che le stopwords comuni siano presenti."""
        common = ["il", "la", "di", "che", "non", "è", "per", "con"]
        for word in common:
            assert word in ITALIAN_STOPWORDS, f"Stopword comune mancante: {word}"

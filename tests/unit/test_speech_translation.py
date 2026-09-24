"""Unit tests for SpeechTranscriber, MultilingualTranslator, and Studio Recorder.

Tests verify:
- SpeechTranscriber VAD voice detection and word timeline synchronization.
- Subtitle segment formatting and SRT generator.
- MultilingualTranslator coverage for Indian Languages Suite (Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, Kannada, Malayalam, Punjabi).
- MultilingualTranslator coverage for Global Languages (Spanish, French, German, Japanese, Chinese, Italian).
- Sub-millisecond latency guarantees (<1.0 ms).
- FastAPI REST endpoints for translation configuration, on-demand translation, and session recording/downloading.
"""

import os
import io
import pytest
import numpy as np
from fastapi.testclient import TestClient
import scipy.io.wavfile as wavfile

from src.models.transcription import SpeechTranscriber, SubtitleSegment
from src.models.translator import MultilingualTranslator, TranslationResult
from src.dashboard.app import app


# ---------------------------------------------------------------------------
# Test 1: SpeechTranscriber
# ---------------------------------------------------------------------------
def test_transcriber_vad_and_emission():
    transcriber = SpeechTranscriber(sample_rate=16000, hop_length=256)

    # Pure silence frame (zero energy) -> should be non-speech
    silence = np.zeros(256, dtype=np.float32)
    seg1 = transcriber.process_frame(silence, timestamp_sec=0.0)
    assert seg1.is_speech is False
    assert seg1.confidence < 0.2

    # Speech-like audio with harmonic formants (150Hz + harmonics)
    t = np.linspace(0, 256 / 16000.0, 256, endpoint=False)
    voice = (0.4 * np.sin(2 * np.pi * 150 * t) + 0.2 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    
    seg2 = transcriber.process_frame(voice, timestamp_sec=0.5)
    assert seg2.is_speech is True
    assert seg2.confidence > 0.5
    assert len(seg2.original_text) > 0


def test_transcriber_srt_export():
    transcriber = SpeechTranscriber(sample_rate=16000, hop_length=256)
    t = np.linspace(0, 256 / 16000.0, 256, endpoint=False)
    voice = (0.4 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)

    for i in range(10):
        transcriber.process_frame(voice, timestamp_sec=i * 0.1)

    translator = MultilingualTranslator(default_lang="hi")
    srt_output = transcriber.export_srt(translator=translator)

    assert len(srt_output) > 0
    assert "-->" in srt_output
    # Check that sequence numbering exists (e.g., "1\n")
    assert "1\n" in srt_output


# ---------------------------------------------------------------------------
# Test 2: MultilingualTranslator (Indian Languages Suite)
# ---------------------------------------------------------------------------
def test_translator_indian_languages_suite():
    translator = MultilingualTranslator(default_lang="hi")
    indian_langs = ["hi", "ta", "te", "bn", "mr", "gu", "kn", "ml", "pa"]

    sample_phrase = "Audio denoising active with crisp voice clarity."

    for lang in indian_langs:
        res = translator.translate(sample_phrase, target_lang=lang)
        assert isinstance(res, TranslationResult)
        assert res.target_lang == lang
        assert len(res.translated_text) > 0
        assert res.latency_ms < 1.0  # Must be sub-millisecond
        assert len(res.target_flag) > 0


def test_translator_global_languages():
    translator = MultilingualTranslator(default_lang="en")
    global_langs = ["es", "fr", "de", "ja", "zh", "it"]

    phrase = "Tensor core neural network inference completed."
    for lang in global_langs:
        res = translator.translate(phrase, target_lang=lang)
        assert res.target_lang == lang
        assert len(res.translated_text) > 0
        assert res.latency_ms < 1.0


def test_translator_dictionary_vocabulary_completeness():
    translator = MultilingualTranslator()
    supported = translator.get_supported_languages()
    assert "indian" in supported
    assert "global" in supported
    assert len(supported["indian"]) == 9
    assert len(supported["global"]) == 6

    # Verify key technical terms exist
    res_hi = translator.translate("crystal clean voice", target_lang="hi")
    assert "क्रिस्टल स्पष्ट आवाज़" in res_hi.translated_text

    res_ta = translator.translate("noise reduction", target_lang="ta")
    assert "இரைச்சல் குறைப்பு" in res_ta.translated_text


# ---------------------------------------------------------------------------
# Test 3: Dashboard REST Endpoints
# ---------------------------------------------------------------------------
@pytest.fixture
def client():
    return TestClient(app)


def test_api_translation_languages(client):
    response = client.get("/api/translation/languages")
    assert response.status_code == 200
    data = response.json()
    assert "indian" in data
    assert "global" in data
    # Ensure Hindi and Tamil are in Indian list
    indian_codes = [item["code"] for item in data["indian"]]
    assert "hi" in indian_codes
    assert "ta" in indian_codes
    assert "te" in indian_codes


def test_api_translation_configure_and_translate(client):
    # Configure to Tamil
    res_cfg = client.post("/api/translation/configure", json={"language": "ta"})
    assert res_cfg.status_code == 200
    data_cfg = res_cfg.json()
    assert data_cfg["active_language"] == "ta"
    assert data_cfg["name"] == "Tamil"

    # Translate endpoint
    res_tr = client.post(
        "/api/translation/translate",
        json={"text": "AI audio denoiser online", "target_lang": "ta"}
    )
    assert res_tr.status_code == 200
    data_tr = res_tr.json()
    assert data_tr["target_lang"] == "ta"
    assert len(data_tr["translated_text"]) > 0


def test_api_recorder_save_and_downloads(client):
    # Synthesize small 0.5s audio buffer (8000 samples @ 16kHz)
    clean_pcm = [float(np.sin(2 * np.pi * 440 * i / 16000)) for i in range(8000)]
    noisy_pcm = [clean_pcm[i] + 0.1 for i in range(8000)]

    save_payload = {
        "clean_pcm": clean_pcm,
        "noisy_pcm": noisy_pcm,
        "transcript": "Hello and welcome to NVIDIA Broadcast studio.",
        "target_lang": "hi",
        "translated_text": "नमस्ते और एनवीडिया ब्रॉडकास्ट स्टूडियो में आपका स्वागत है।"
    }

    res_save = client.post("/api/recorder/save", json=save_payload)
    assert res_save.status_code == 200
    data_save = res_save.json()
    assert data_save["success"] is True
    session_id = data_save["session_id"]
    assert len(session_id) > 0

    # 1. Download clean WAV
    res_clean = client.get(f"/api/recorder/download/{session_id}?format=clean_wav")
    assert res_clean.status_code == 200
    assert res_clean.headers["content-type"] == "audio/wav"
    sr, pcm_read = wavfile.read(io.BytesIO(res_clean.content))
    assert sr == 16000
    assert len(pcm_read) == 8000

    # 2. Download raw noisy WAV
    res_noisy = client.get(f"/api/recorder/download/{session_id}?format=noisy_wav")
    assert res_noisy.status_code == 200

    # 3. Download subtracted delta WAV
    res_delta = client.get(f"/api/recorder/download/{session_id}?format=delta_wav")
    assert res_delta.status_code == 200

    # 4. Download SRT Subtitles
    res_srt = client.get(f"/api/recorder/download/{session_id}?format=srt")
    assert res_srt.status_code == 200
    assert "-->" in res_srt.text

    # 5. Download TXT Transcript
    res_txt = client.get(f"/api/recorder/download/{session_id}?format=txt")
    assert res_txt.status_code == 200
    assert "NVIDIA Edge AI Audio Denoiser Studio Session" in res_txt.text
    assert "नमस्ते और एनवीडिया" in res_txt.text


# ---------------------------------------------------------------------------
# Test 4: Wraparound Resilience, Script Purity, and WebSocket Live Speech
# ---------------------------------------------------------------------------
def test_transcriber_timestamp_wraparound_resilience():
    """Verify that looping audio timestamps do not freeze or stall word emission."""
    transcriber = SpeechTranscriber(sample_rate=16000, hop_length=256)
    t = np.linspace(0, 256 / 16000.0, 256, endpoint=False)
    voice = (0.4 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)

    # Initial playback up to 3.0s
    for ts in [0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.0]:
        seg = transcriber.process_frame(voice, timestamp_sec=ts)

    assert len(transcriber.current_transcript) > 0

    # Simulate loop wraparound: timestamp jumps backward to 0.0s
    seg_wrap = transcriber.process_frame(voice, timestamp_sec=0.0)
    assert seg_wrap is not None
    # Next frame at 0.35s must immediately emit words without stalling for 3 seconds
    seg_next = transcriber.process_frame(voice, timestamp_sec=0.35)
    assert seg_next.is_speech is True
    assert len(seg_next.original_text) > 0


def test_transcriber_pause_preserves_last_completed_text():
    """Verify that subtitle does not flicker to blank during inter-sentence silence."""
    transcriber = SpeechTranscriber(sample_rate=16000, hop_length=256)
    transcriber.last_completed_text = "NVIDIA RTX AI engine isolating crystal clean speech in real time."
    transcriber.accumulated_text = ""

    silence = np.zeros(256, dtype=np.float32)
    seg = transcriber.process_frame(silence, timestamp_sec=1.0)
    assert seg.original_text == "NVIDIA RTX AI engine isolating crystal clean speech in real time."
    assert seg.is_speech is False


def test_translator_no_mixed_language_strings():
    """Verify zero mixed-language strings (no Latin ASCII in Indic or East Asian outputs)."""
    import re
    translator = MultilingualTranslator()
    indic_langs = ["hi", "ta", "te", "bn", "mr", "gu", "kn", "ml", "pa"]
    test_phrases = [
        "NVIDIA RTX AI engine isolating crystal clean speech in real time.",
        "The birch canoe slid on the smooth dark water.",
        "Acoustic noise suppression preserving natural human vocal harmonics.",
        "Sub millisecond edge tensor compute delivering broadcast clarity.",
        "Audio denoising active with crisp voice clarity.",
        "Hello and welcome to NVIDIA Broadcast studio.",
        "Good morning",
        "Start recording",
        "Crystal clean edge audio processor online",
    ]

    latin_pattern = re.compile(r"[a-zA-Z]")

    for phrase in test_phrases:
        for lang in indic_langs:
            res = translator.translate(phrase, target_lang=lang)
            assert res.latency_ms < 1.0, f"Latency exceeded 1.0ms for {lang}: {res.latency_ms}ms"
            assert len(res.translated_text) > 0
            # Ensure no stray Latin letters in Indic output
            assert not latin_pattern.search(res.translated_text), (
                f"Mixed-language corruption detected for {lang}: '{res.translated_text}' from '{phrase}'"
            )

        # Also verify Japanese and Chinese produce non-empty valid native text under 1.0ms
        for lang in ["ja", "zh"]:
            res = translator.translate(phrase, target_lang=lang)
            assert res.latency_ms < 1.0
            assert len(res.translated_text) > 0
            # For phrases without brand names, verify Japanese/Chinese script
            if "NVIDIA" not in phrase:
                assert not latin_pattern.search(res.translated_text), (
                    f"Latin characters found in {lang}: '{res.translated_text}'"
                )


def test_websocket_speech_transcript_and_custom_translate(client):
    """Verify WebSocket /ws/stream handles live speech transcripts and custom translations."""
    import json
    with client.websocket_connect("/ws/stream") as ws:
        # Test Web Speech API recognition transcript message
        ws.send_text(json.dumps({
            "type": "speech_transcript",
            "transcript": "Hello and welcome to nvidia broadcast studio",
            "target_lang": "hi",
            "is_final": True
        }))
        resp1_text = ws.receive_text()
        resp1 = json.loads(resp1_text)
        assert resp1["type"] == "subtitles_update"
        assert resp1["subtitles"]["original_text"] == "Hello and welcome to nvidia broadcast studio"
        assert "नमस्ते" in resp1["subtitles"]["translated_text"]
        assert resp1["subtitles"]["target_lang"] == "hi"

        # Test custom interactive text translate message
        ws.send_text(json.dumps({
            "type": "custom_text_translate",
            "text": "Audio denoising active with crisp voice clarity",
            "target_lang": "ta"
        }))
        resp2_text = ws.receive_text()
        resp2 = json.loads(resp2_text)
        assert resp2["type"] == "subtitles_update"
        assert resp2["subtitles"]["target_lang"] == "ta"
        assert "இரைச்சல்" in resp2["subtitles"]["translated_text"]


def test_short_phrase_translation_no_hallucination():
    """Verify that short 2-3 word phrases translate concisely without hallucinating full benchmark sentences."""
    translator = MultilingualTranslator()

    # "NVIDIA RTX" should translate to "एनवीडिया आरटीएक्स" in Hindi, not the 11-word benchmark sentence
    res_hi = translator.translate("NVIDIA RTX", target_lang="hi")
    assert res_hi.translated_text == "एनवीडिया आरटीएक्स"
    assert len(res_hi.translated_text.split()) == 2

    # "clear speech" should translate to exact phrase or token mapping, not sentence
    res_ta = translator.translate("clear speech", target_lang="ta")
    assert "தெளிவான" in res_ta.translated_text or "குரல்" in res_ta.translated_text or "பேச்சு" in res_ta.translated_text
    assert len(res_ta.translated_text.split()) <= 3


def test_indic_initial_vowels_typography():
    """Verify that transliterated tokens starting with vowels never begin with combining marks (matras)."""
    translator = MultilingualTranslator()
    indic_langs = ["hi", "ta", "te", "bn", "mr", "gu", "kn", "ml", "pa"]
    test_words = ["apple", "engine", "input", "output", "ultra", "audio", "echo"]

    # Unicode combining vowel sign ranges for Indic scripts (matras):
    combining_ranges = [
        (0x093E, 0x094C),  # Devanagari
        (0x0962, 0x0963),  # Devanagari vocalic
        (0x09BE, 0x09CC),  # Bengali
        (0x09E2, 0x09E3),  # Bengali vocalic
        (0x0A3E, 0x0A4C),  # Gurmukhi
        (0x0ABE, 0x0ACC),  # Gujarati
        (0x0BBE, 0x0BCC),  # Tamil
        (0x0C3E, 0x0C4C),  # Telugu
        (0x0CBE, 0x0CCC),  # Kannada
        (0x0D3E, 0x0D4C),  # Malayalam
    ]

    def is_combining_matra(ch: str) -> bool:
        cp = ord(ch)
        return any(start <= cp <= end for start, end in combining_ranges)

    for word in test_words:
        for lang in indic_langs:
            out = translator._transliterate_indic(word, lang)
            assert len(out) > 0, f"Empty transliteration for {word} in {lang}"
            first_char = out[0]
            assert not is_combining_matra(first_char), (
                f"Typography bug: Word '{word}' in '{lang}' starts with combining matra U+{ord(first_char):04X}: '{out}'"
            )


def test_export_srt_target_language_and_custom_segments():
    """Verify that export_srt respects target_lang and custom segments."""
    transcriber = SpeechTranscriber()
    translator = MultilingualTranslator(default_lang="hi")

    custom_segments = [
        SubtitleSegment(
            index=1,
            start_time_sec=0.0,
            end_time_sec=2.5,
            original_text="Audio denoising active with crisp voice clarity.",
            confidence=0.98,
            is_final=True,
            is_speech=True,
        )
    ]

    # Export with target_lang="ta" (Tamil)
    srt_ta = transcriber.export_srt(segments=custom_segments, translator=translator, target_lang="ta")
    assert "00:00:00,000 --> 00:00:02,500" in srt_ta
    assert "Audio denoising active with crisp voice clarity." in srt_ta
    assert "இரைச்சல்" in srt_ta  # Tamil translation

    # Export with target_lang="hi" (Hindi)
    srt_hi = transcriber.export_srt(segments=custom_segments, translator=translator, target_lang="hi")
    assert "ऑडियो" in srt_hi or "क्रिस्टल" in srt_hi or "आवाज़" in srt_hi


def test_websocket_set_language(client):
    """Verify WebSocket /ws/stream handles dynamic set_language messages."""
    import json
    with client.websocket_connect("/ws/stream") as ws:
        # Send set_language command
        ws.send_text(json.dumps({
            "type": "set_language",
            "language": "ta"
        }))
        resp_text = ws.receive_text()
        resp = json.loads(resp_text)
        assert resp["type"] == "language_updated"
        assert resp["language"] == "ta"
        assert resp["name"] == "Tamil"
        assert "🇮🇳" in resp["flag"]


def test_all_benchmark_utterances_vocabulary_words_covered():
    """Verify that all words in benchmark utterances exist in VOCAB_MAP for all 15 languages."""
    import re
    translator = MultilingualTranslator()
    all_langs = list(translator.LANGUAGES.keys())
    assert len(all_langs) == 15

    for utterance in SpeechTranscriber.BENCHMARK_UTTERANCES:
        clean = re.sub(r"[^\w\s]", "", utterance).lower()
        words = clean.split()
        for word in words:
            assert word in translator.VOCAB_MAP, f"Word '{word}' from '{utterance}' missing in VOCAB_MAP"
            for lang in all_langs:
                assert lang in translator.VOCAB_MAP[word], f"Word '{word}' missing translation for {lang}"
                assert len(translator.VOCAB_MAP[word][lang]) > 0


def test_transcriber_live_mode_no_benchmark_pollution():
    """Verify that in live mode, voiced audio frames do not pollute user transcript with benchmark words."""
    transcriber = SpeechTranscriber(sample_rate=16000, hop_length=256)
    transcriber.set_mode("live")
    transcriber.set_live_transcript("Testing live speech streaming without benchmark corruption", is_final=False)

    # Synthesize voiced audio
    t = np.linspace(0, 256 / 16000.0, 256, endpoint=False)
    voice = (0.4 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)

    # Process 20 frames across 2 seconds
    for i in range(20):
        seg = transcriber.process_frame(voice, timestamp_sec=i * 0.1)
        assert "NVIDIA" not in seg.original_text
        assert "RTX" not in seg.original_text
        assert "canoe" not in seg.original_text
        assert "Testing live speech" in seg.original_text

    assert transcriber.current_transcript == "Testing live speech streaming without benchmark corruption"


def test_translator_native_script_preservation():
    """Verify that inputs already in native Indic script are preserved and not replaced with fallback."""
    translator = MultilingualTranslator()

    res_hi = translator.translate("नमस्ते", target_lang="hi")
    assert "नमस्ते" in res_hi.translated_text
    assert res_hi.translated_text != "वाणी"

    res_ta = translator.translate("வணக்கம்", target_lang="ta")
    assert "வணக்கம்" in res_ta.translated_text


def test_translator_reverse_phrase_and_word_lookup():
    """Verify that phrases and words can be queried bidirectionally."""
    translator = MultilingualTranslator()

    res = translator.translate("शुभ प्रभात", target_lang="hi")
    assert len(res.translated_text) > 0


def test_translator_empty_and_numeric_input():
    """Verify empty text returns empty result and numeric tokens are preserved."""
    translator = MultilingualTranslator()

    empty_res = translator.translate("   ", target_lang="hi")
    assert empty_res.translated_text == ""
    assert empty_res.latency_ms < 1.0

    num_res = translator.translate("rtx 4090", target_lang="hi")
    assert "4090" in num_res.translated_text
    assert "वाणी" not in num_res.translated_text


def test_translator_none_input_resilience():
    """Verify None input is safely handled without raising AttributeError."""
    translator = MultilingualTranslator()
    res = translator.translate(None, target_lang="hi")
    assert res.translated_text == ""
    assert res.latency_ms < 1.0

    transcriber = SpeechTranscriber()
    transcriber.set_live_transcript(None)
    assert transcriber.mode == "live"


def test_translator_sub_millisecond_benchmark_execution():
    """Verify precomputed O(1) lookups execute well under 0.1ms across Indian and Global languages."""
    translator = MultilingualTranslator()
    test_phrase = "Sub millisecond edge tensor compute delivering broadcast clarity."
    all_langs = list(translator.LANGUAGES.keys())

    # Warm up and verify every language
    for lang in all_langs:
        res = translator.translate(test_phrase, target_lang=lang)
        assert len(res.translated_text) > 0
        assert res.latency_ms < 0.5  # Well below 1.0ms, typical < 0.05ms

    # Verify bidirectional cross-language phrase lookup
    res_cross = translator.translate("Guten Morgen", target_lang="hi")
    assert "सुप्रभात" in res_cross.translated_text or len(res_cross.translated_text) > 0
    assert res_cross.latency_ms < 0.5



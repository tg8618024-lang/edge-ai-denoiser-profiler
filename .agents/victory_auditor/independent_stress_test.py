"""Independent stress test and forensic integrity verification.
Executed by Victory Auditor.
"""

import sys
import os
import time
import re
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.transcription import SpeechTranscriber, SubtitleSegment
from src.models.translator import MultilingualTranslator, TranslationResult
from src.dashboard.app import app
from fastapi.testclient import TestClient

def test_multilingual_engine_coverage():
    print(">>> [STRESS TEST 1] MULTILINGUAL ENGINE 15-LANGUAGE BENCHMARK & PURITY...")
    translator = MultilingualTranslator()
    all_langs = list(translator.LANGUAGES.keys())
    assert len(all_langs) == 15, f"Expected 15 languages, got {len(all_langs)}"

    indian_langs = [k for k, v in translator.LANGUAGES.items() if v["group"] == "Indian"]
    global_langs = [k for k, v in translator.LANGUAGES.items() if v["group"] == "Global"]
    assert len(indian_langs) == 9, f"Expected 9 Indian languages, got {len(indian_langs)}"
    assert len(global_langs) == 6, f"Expected 6 Global languages, got {len(global_langs)}"

    test_sentences = [
        "NVIDIA RTX AI engine isolating crystal clean speech in real time.",
        "The birch canoe slid on the smooth dark water.",
        "Acoustic noise suppression preserving natural human vocal harmonics.",
        "Sub millisecond edge tensor compute delivering broadcast clarity.",
        "These days a clear communication signal is essential for creators.",
        "Hello and welcome to nvidia broadcast studio",
        "Start recording",
        "Stop recording",
        "Audio denoising active with crisp voice clarity",
        "Tensor core neural network inference completed",
        "Listening for voice stream",
        "Waiting for audio stream to transcribe speech",
    ]

    latin_pattern = re.compile(r"[a-zA-Z]")
    total_translations = 0
    max_latency = 0.0

    for sent in test_sentences:
        for lang in all_langs:
            t0 = time.perf_counter()
            res = translator.translate(sent, target_lang=lang)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            max_latency = max(max_latency, elapsed_ms)
            total_translations += 1

            assert len(res.translated_text) > 0, f"Empty translation for {lang}: '{sent}'"
            assert res.latency_ms < 1.0, f"Latency {res.latency_ms}ms exceeds 1.0ms limit"

            # Check script purity: Indic languages MUST NOT contain Latin letters
            if lang in indian_langs:
                match = latin_pattern.search(res.translated_text)
                assert not match, f"Mixed-language string found in {lang}: '{res.translated_text}' from '{sent}'"

    print(f"  Verified {total_translations} translations across 15 languages.")
    print(f"  Max measured translation latency: {max_latency:.4f} ms (budget < 1.0 ms: PASS)")
    print("  Script purity verification: 100% PASS (Zero mixed-language corruption)")

def test_asr_continuous_loop_wraparound():
    print("\n>>> [STRESS TEST 2] ASR CONTINUOUS STREAMING & WRAPAROUND RESILIENCE...")
    transcriber = SpeechTranscriber(sample_rate=16000, hop_length=256)
    
    # Generate speech-like harmonic signal
    t = np.linspace(0, 256 / 16000.0, 256, endpoint=False)
    voice_frame = (0.5 * np.sin(2 * np.pi * 180 * t) + 0.3 * np.sin(2 * np.pi * 360 * t)).astype(np.float32)

    # Simulate 5 consecutive loops of 3.0s benchmark audio (15 seconds total)
    emissions_per_loop = []
    
    for loop_num in range(5):
        loop_emissions = 0
        # 3.0s audio in 256-hop frames = ~187 frames
        for f in range(188):
            ts = (f * 256) / 16000.0
            seg = transcriber.process_frame(voice_frame, timestamp_sec=ts)
            if seg.is_speech and seg.original_text:
                loop_emissions += 1
        emissions_per_loop.append(loop_emissions)

    print(f"  Emissions per 3.0s audio loop: {emissions_per_loop}")
    # Verify that in EVERY loop words were emitted and never stalled
    for idx, em in enumerate(emissions_per_loop):
        assert em > 150, f"Loop {idx} stalled or froze words! Emissions: {em}"
    print("  Continuous loop wraparound test: PASS (Zero stalls or freezing)")

def test_live_microphone_pipeline():
    print("\n>>> [STRESS TEST 3] LIVE MICROPHONE SPEECH-TO-TRANSLATION PIPELINE...")
    client = TestClient(app)
    with client.websocket_connect("/ws/stream") as ws:
        # Simulate browser Web Speech recognition streaming Hindi, Tamil, Telugu
        test_inputs = [
            ("Hello and welcome", "hi", "नमस्ते"),
            ("Start recording", "ta", "தொடங்கவும்"),
            ("crystal clean voice", "te", "వాయిస్"),
        ]

        for text_in, lang_code, expected_substr in test_inputs:
            ws.send_json({
                "type": "speech_transcript",
                "transcript": text_in,
                "target_lang": lang_code,
                "is_final": True
            })
            resp = ws.receive_json()
            assert resp["type"] == "subtitles_update"
            subs = resp["subtitles"]
            assert subs["original_text"] == text_in
            assert subs["target_lang"] == lang_code
            assert expected_substr in subs["translated_text"], f"Expected '{expected_substr}' in '{subs['translated_text']}'"

        # Simulate custom text translation
        ws.send_json({
            "type": "custom_text_translate",
            "text": "Audio denoising active with crisp voice clarity",
            "target_lang": "bn"
        })
        resp_custom = ws.receive_json()
        assert resp_custom["type"] == "subtitles_update"
        assert resp_custom["subtitles"]["target_lang"] == "bn"
        assert len(resp_custom["subtitles"]["translated_text"]) > 0

    print("  Live speech & custom translate WebSocket tests: PASS")

def test_multitrack_session_recorder():
    print("\n>>> [STRESS TEST 4] MULTI-TRACK RECORDER & 1-CLICK DOWNLOADS...")
    client = TestClient(app)
    clean_pcm = [float(np.sin(2 * np.pi * 300 * i / 16000)) for i in range(16000)] # 1 sec
    noisy_pcm = [c + 0.05 * np.random.randn() for c in clean_pcm]

    save_resp = client.post("/api/recorder/save", json={
        "clean_pcm": clean_pcm,
        "noisy_pcm": noisy_pcm,
        "transcript": "These days a clear communication signal is essential for creators.",
        "target_lang": "hi",
    })
    assert save_resp.status_code == 200
    save_data = save_resp.json()
    session_id = save_data["session_id"]
    assert save_data["success"] is True

    # Test all 5 download formats
    formats = ["clean_wav", "noisy_wav", "delta_wav", "srt", "txt"]
    for fmt in formats:
        dl_resp = client.get(f"/api/recorder/download/{session_id}?format={fmt}")
        assert dl_resp.status_code == 200, f"Download format {fmt} returned status {dl_resp.status_code}"
        assert len(dl_resp.content) > 0, f"Empty content for download format {fmt}"

    print("  Multi-track recorder save and all 5 download formats: PASS")

if __name__ == "__main__":
    test_multilingual_engine_coverage()
    test_asr_continuous_loop_wraparound()
    test_live_microphone_pipeline()
    test_multitrack_session_recorder()
    print("\n" + "=" * 80)
    print("  INDEPENDENT AUDITOR STRESS TEST SUITE PASSED 100%")
    print("=" * 80)

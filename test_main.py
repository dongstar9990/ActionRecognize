import base64
import asyncio
import importlib
import os
import sys
import types
import unittest
from unittest import mock
from pathlib import Path
from tempfile import TemporaryDirectory

import api
from main import (
    UNREACHABLE_PHRASES,
    find_best_match,
    normalize_text,
    planned_clip_window,
    transcribe_audio,
)
from logistic_model import LogisticTextRecognizer, load_training_rows, train_logistic_model
from recognizer import RealtimeRecognizer


class SttFilterTests(unittest.TestCase):
    def test_normalize_text_removes_vietnamese_marks_and_punctuation(self):
        self.assertEqual(
            normalize_text(
                "Thuê bao quý khách vừa gọi tạm thời không liên lạc được, xin vui lòng gọi lại sau..."
            ),
            "thue bao quy khach vua goi tam thoi khong lien lac duoc xin vui long goi lai sau",
        )

    def test_find_best_match_accepts_stt_variations_across_adjacent_segments(self):
        segments = [
            {"start": 0.0, "end": 1.2, "text": "nhac cho"},
            {
                "start": 1.2,
                "end": 4.8,
                "text": "thue bao quy khach vua goi tam thoi khong lien lac duoc",
            },
            {"start": 4.8, "end": 7.0, "text": "xin vui long goi lai sau"},
            {"start": 7.0, "end": 9.0, "text": "cam on"},
        ]

        match = find_best_match(
            segments,
            target_phrases=UNREACHABLE_PHRASES,
            threshold=85,
        )

        self.assertIsNotNone(match)
        self.assertEqual(match["start"], 1.2)
        self.assertEqual(match["end"], 7.0)
        self.assertGreaterEqual(match["score"], 85)
        self.assertIn("goi lai sau", normalize_text(match["text"]))

    def test_find_best_match_rejects_unrelated_audio_text(self):
        segments = [
            {"start": 0.0, "end": 3.0, "text": "alo anh co nghe may khong"},
            {"start": 3.0, "end": 8.0, "text": "em goi tu cong ty tima de tu van khoan vay"},
        ]

        self.assertIsNone(find_best_match(segments, threshold=85))

    def test_planned_clip_window_adds_padding_without_negative_start(self):
        self.assertEqual(
            planned_clip_window(start=0.2, end=5.0, duration=5.3, padding=0.5),
            (0.0, 5.3),
        )
        self.assertEqual(
            planned_clip_window(start=1.2, end=5.0, duration=9.0, padding=0.5),
            (0.7, 5.5),
        )

    def test_transcribe_audio_forwards_realtime_speed_options(self):
        class FakeSegment:
            start = 0.0
            end = 1.0
            text = "alo"

        class FakeInfo:
            duration = 1.0

        class FakeModel:
            def __init__(self):
                self.kwargs = None

            def transcribe(self, audio_path, **kwargs):
                self.audio_path = audio_path
                self.kwargs = kwargs
                return [FakeSegment()], FakeInfo()

        model = FakeModel()

        transcript, segments, duration = transcribe_audio(
            model,
            Path("chunk.wav"),
            "vi",
            beam_size=1,
            vad_filter=False,
        )

        self.assertEqual(model.audio_path, "chunk.wav")
        self.assertEqual(model.kwargs["language"], "vi")
        self.assertEqual(model.kwargs["beam_size"], 1)
        self.assertFalse(model.kwargs["vad_filter"])
        self.assertEqual(transcript, "alo")
        self.assertEqual(segments, [{"start": 0.0, "end": 1.0, "text": "alo"}])
        self.assertEqual(duration, 1.0)


class RealtimeRecognizerTests(unittest.TestCase):
    def test_recognize_segments_returns_json_ready_match_payload(self):
        recognizer = RealtimeRecognizer(
            target_phrases=["de lai loi nhan sau tieng bip"],
            threshold=80,
            padding=0.25,
        )
        result = recognizer.recognize_segments(
            [
                {"start": 0.0, "end": 1.0, "text": "alo"},
                {
                    "start": 1.0,
                    "end": 3.0,
                    "text": "quy khach vui long de lai loi nhan sau tieng bip",
                },
            ],
            transcript="alo quy khach vui long de lai loi nhan sau tieng bip",
            duration=4.0,
            latency_ms=42,
        )

        self.assertTrue(result["matched"])
        self.assertEqual(result["matched_phrase"], "de lai loi nhan sau tieng bip")
        self.assertEqual(result["matched_text"], "quy khach vui long de lai loi nhan sau tieng bip")
        self.assertEqual(result["start_sec"], 0.75)
        self.assertEqual(result["end_sec"], 3.25)
        self.assertEqual(result["latency_ms"], 42)
        self.assertGreaterEqual(result["match_score"], 80)

    def test_recognize_segments_returns_best_score_when_not_matched(self):
        recognizer = RealtimeRecognizer(
            target_phrases=["de lai loi nhan sau tieng bip"],
            threshold=90,
        )
        result = recognizer.recognize_segments(
            [{"start": 0.0, "end": 2.0, "text": "alo em nghe may khong"}],
            transcript="alo em nghe may khong",
        )

        self.assertFalse(result["matched"])
        self.assertEqual(result["matched_text"], "")
        self.assertEqual(result["matched_phrase"], "")
        self.assertEqual(result["start_sec"], None)
        self.assertEqual(result["end_sec"], None)
        self.assertLess(result["best_score"], 90)

    def test_recognize_batches_returns_as_soon_as_window_matches(self):
        recognizer = RealtimeRecognizer(
            target_phrases=["de lai loi nhan sau tieng bip"],
            threshold=80,
        )
        result = recognizer.recognize_batches(
            [
                [{"start": 0.0, "end": 1.0, "text": "alo"}],
                [{"start": 1.0, "end": 2.0, "text": "vui long de lai loi nhan"}],
                [{"start": 2.0, "end": 3.0, "text": "sau tieng bip"}],
                [{"start": 3.0, "end": 4.0, "text": "cam on"}],
            ]
        )

        self.assertTrue(result["matched"])
        self.assertEqual(result["processed_batches"], 3)
        self.assertIn("sau tieng bip", result["matched_text"])


class ApiTests(unittest.TestCase):
    def test_api_module_imports_without_optional_fastapi_dependency(self):
        self.assertTrue(callable(api.run_server))
        self.assertTrue(callable(api.recognize_audio_url))

    def test_recognize_audio_url_uses_logistic_mode_with_injected_transcriber(self):
        class FakeClassifier:
            def predict_proba(self, texts):
                return [[0.1, 0.9]]

        def fake_transcribe(audio_path):
            return "quy khach vui long de lai loi nhan sau tieng bip", [
                {
                    "start": 0.0,
                    "end": 3.0,
                    "text": "quy khach vui long de lai loi nhan sau tieng bip",
                }
            ], 3.5

        with TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "sample.mp3"
            audio_path.write_bytes(b"audio")
            result = api.recognize_audio_url(
                str(audio_path),
                mode="logistic",
                logistic_recognizer=LogisticTextRecognizer(
                    classifier=FakeClassifier(),
                    model_path=Path(tmpdir) / "model.joblib",
                ),
                transcribe_func=fake_transcribe,
                threshold=0.8,
            )

        self.assertTrue(result["matched"])
        self.assertEqual(result["label"], "contains_information")
        self.assertEqual(result["probability"], 0.9)
        self.assertEqual(result["threshold"], 0.8)
        self.assertIn("de lai loi nhan", result["transcript"])

    def test_realtime_session_returns_partial_until_accumulated_chunks_match(self):
        transcribed_paths = []

        def fake_transcribe(audio_path):
            transcribed_paths.append(Path(audio_path))
            if len(transcribed_paths) == 1:
                return "alo", [{"start": 0.0, "end": 1.0, "text": "alo"}], 1.0
            return (
                "vui long de lai loi nhan sau tieng bip",
                [
                    {
                        "start": 0.0,
                        "end": 2.0,
                        "text": "vui long de lai loi nhan sau tieng bip",
                    }
                ],
                2.0,
            )

        with TemporaryDirectory() as tmpdir:
            session = api.RealtimeRecognitionSession(
                recognizer=RealtimeRecognizer(
                    target_phrases=["de lai loi nhan sau tieng bip"],
                    threshold=80,
                ),
                transcribe_func=fake_transcribe,
                cache_dir=Path(tmpdir),
                mode="fuzzy",
            )

            first = session.process_chunk(b"first audio", suffix=".wav")
            second = session.process_chunk(b"second audio", suffix=".wav")
            self.assertTrue(all(path.exists() for path in transcribed_paths))

        self.assertEqual(first["type"], "partial")
        self.assertFalse(first["matched"])
        self.assertEqual(first["chunk_index"], 1)
        self.assertEqual(first["processed_batches"], 1)
        self.assertEqual(second["type"], "match")
        self.assertTrue(second["matched"])
        self.assertEqual(second["chunk_index"], 2)
        self.assertEqual(second["processed_batches"], 2)
        self.assertEqual(second["start_sec"], 0.5)
        self.assertEqual(second["end_sec"], 3.0)
        self.assertIn("sau tieng bip", second["matched_text"])

    def test_realtime_session_supports_logistic_mode_on_accumulated_transcript(self):
        class FakeClassifier:
            def predict_proba(self, texts):
                return [[0.2, 0.8]]

        def fake_transcribe(audio_path):
            return "quy khach vui long de lai loi nhan sau tieng bip", [
                {
                    "start": 0.0,
                    "end": 2.0,
                    "text": "quy khach vui long de lai loi nhan sau tieng bip",
                }
            ], 2.0

        with TemporaryDirectory() as tmpdir:
            session = api.RealtimeRecognitionSession(
                recognizer=RealtimeRecognizer(["de lai loi nhan sau tieng bip"]),
                logistic_recognizer=LogisticTextRecognizer(
                    classifier=FakeClassifier(),
                    model_path=Path(tmpdir) / "model.joblib",
                ),
                transcribe_func=fake_transcribe,
                cache_dir=Path(tmpdir),
                mode="logistic",
                threshold=0.7,
            )

            result = session.process_chunk(b"audio", suffix=".wav")

        self.assertEqual(result["type"], "match")
        self.assertTrue(result["matched"])
        self.assertEqual(result["label"], "contains_information")
        self.assertEqual(result["threshold"], 0.7)
        self.assertEqual(result["chunk_index"], 1)
        self.assertEqual(result["processed_batches"], 1)

    def test_decode_websocket_audio_message_accepts_base64_audio_payload(self):
        encoded_audio = base64.b64encode(b"first audio").decode("ascii")

        chunk_bytes, suffix = api.decode_websocket_audio_message({
            "type": "audio",
            "data_base64": encoded_audio,
            "chunk_suffix": "wav",
        })

        self.assertEqual(chunk_bytes, b"first audio")
        self.assertEqual(suffix, ".wav")

    def test_decode_websocket_audio_message_rejects_invalid_base64(self):
        with self.assertRaisesRegex(ValueError, "invalid base64 audio payload"):
            api.decode_websocket_audio_message({
                "type": "audio",
                "data_base64": "not valid base64!",
                "chunk_suffix": ".wav",
            })

    def test_decode_websocket_audio_message_rejects_unsupported_text_payload(self):
        with self.assertRaisesRegex(
            ValueError,
            "expected binary audio chunk or JSON message type audio/end",
        ):
            api.decode_websocket_audio_message({"type": "ping"})

    def test_process_realtime_chunk_runs_session_work_in_thread(self):
        class RecordingSession:
            def process_chunk(self, chunk_bytes, suffix):
                raise AssertionError("process_chunk should be called through asyncio.to_thread")

        session = RecordingSession()
        calls = []

        async def fake_to_thread(func, *args):
            calls.append((func, args))
            return {"type": "partial"}

        with mock.patch("api.asyncio.to_thread", new=fake_to_thread):
            result = asyncio.run(api.process_realtime_chunk(
                session,
                b"audio",
                ".wav",
            ))

        self.assertEqual(result, {"type": "partial"})
        self.assertEqual(calls[0][0].__self__, session)
        self.assertEqual(calls[0][0].__name__, "process_chunk")
        self.assertEqual(calls[0][1], (b"audio", ".wav"))

    def test_get_cached_model_keeps_realtime_and_default_models_separate(self):
        loaded = []

        def fake_loader(model_size, device, compute_type):
            loaded.append((model_size, device, compute_type))
            return f"{model_size}-{device}-{compute_type}"

        cache = {}
        tiny = api.get_cached_model(cache, "tiny", "cpu", "int8", loader=fake_loader)
        small = api.get_cached_model(cache, "small", "cpu", "int8", loader=fake_loader)
        tiny_again = api.get_cached_model(cache, "tiny", "cpu", "int8", loader=fake_loader)

        self.assertEqual(tiny, "tiny-cpu-int8")
        self.assertEqual(small, "small-cpu-int8")
        self.assertIs(tiny_again, tiny)
        self.assertEqual(loaded, [("tiny", "cpu", "int8"), ("small", "cpu", "int8")])

    def test_websocket_realtime_endpoint_returns_partial_and_match(self):
        if os.environ.get("RUN_FASTAPI_WS_TESTS") != "1":
            self.skipTest("set RUN_FASTAPI_WS_TESTS=1 to run FastAPI websocket integration tests")
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("FastAPI test client is not installed")

        calls = []

        def fake_transcribe(audio_path):
            calls.append(Path(audio_path))
            if len(calls) == 1:
                return "alo", [{"start": 0.0, "end": 1.0, "text": "alo"}], 1.0
            return (
                "vui long de lai loi nhan sau tieng bip",
                [
                    {
                        "start": 0.0,
                        "end": 2.0,
                        "text": "vui long de lai loi nhan sau tieng bip",
                    }
                ],
                2.0,
            )

        with TemporaryDirectory() as tmpdir:
            app = api.build_api_app(
                recognizer=RealtimeRecognizer(
                    target_phrases=["de lai loi nhan sau tieng bip"],
                    threshold=80,
                ),
                transcribe_func=fake_transcribe,
                cache_dir=Path(tmpdir),
            )
            client = TestClient(app)
            with client.websocket_connect("/recognize/ws") as websocket:
                websocket.send_json({"mode": "fuzzy", "stop_on_match": False})
                websocket.send_bytes(b"first audio")
                first = websocket.receive_json()
                websocket.send_bytes(b"second audio")
                second = websocket.receive_json()
                websocket.send_json({"type": "end"})
                final = websocket.receive_json()

        self.assertEqual(first["type"], "partial")
        self.assertFalse(first["matched"])
        self.assertEqual(second["type"], "match")
        self.assertTrue(second["matched"])
        self.assertEqual(final["type"], "match")
        self.assertEqual(final["processed_batches"], 2)


class MicRealtimeClientTests(unittest.TestCase):
    def import_test_mic_without_running(self):
        sys.modules.pop("test_mic", None)
        fake_sounddevice = types.SimpleNamespace(rec=lambda *args, **kwargs: None)
        fake_websockets = types.SimpleNamespace(connect=lambda *args, **kwargs: None)
        with mock.patch.dict(
            sys.modules,
            {
                "sounddevice": fake_sounddevice,
                "websockets": fake_websockets,
            },
        ), mock.patch("asyncio.run"):
            return importlib.import_module("test_mic")

    def test_mic_realtime_config_uses_fuzzy_unreachable_detection(self):
        test_mic = self.import_test_mic_without_running()

        config = test_mic.build_realtime_config()

        self.assertEqual(test_mic.CHUNK_SEC, 1)
        self.assertEqual(config["mode"], "fuzzy")
        self.assertEqual(config["threshold"], 70)
        self.assertTrue(config["stop_on_match"])
        self.assertEqual(config["chunk_suffix"], ".wav")
        self.assertEqual(config["model_size"], "tiny")
        self.assertEqual(config["device"], "cpu")
        self.assertEqual(config["compute_type"], "int8")
        self.assertEqual(config["beam_size"], 1)
        self.assertFalse(config["vad_filter"])
        self.assertIn(
            "thuê bao quý khách vừa gọi tạm thời không liên lạc được",
            config["target_phrases"],
        )

    def test_mic_formats_partial_and_match_results_for_console(self):
        test_mic = self.import_test_mic_without_running()

        partial = test_mic.format_result({
            "type": "partial",
            "matched": False,
            "chunk_index": 3,
            "best_score": 64,
            "best_text": "thue bao quy khach",
        })
        match = test_mic.format_result({
            "type": "match",
            "matched": True,
            "chunk_index": 4,
            "match_score": 92,
            "matched_text": "thue bao quy khach vua goi tam thoi khong lien lac duoc",
        })

        self.assertEqual(partial, "[3] partial best=64 text=thue bao quy khach")
        self.assertEqual(
            match,
            "[4] MATCH score=92 text=thue bao quy khach vua goi tam thoi khong lien lac duoc",
        )

    def test_mic_audio_gate_rejects_silence_and_accepts_speech(self):
        test_mic = self.import_test_mic_without_running()

        silence = [0, 0, 0, 0]
        quiet_noise = [10, -10, 10, -10]
        speech = [1000, -1000, 1000, -1000]

        self.assertEqual(test_mic.audio_rms(silence), 0.0)
        self.assertFalse(test_mic.should_send_audio(quiet_noise, threshold=300))
        self.assertTrue(test_mic.should_send_audio(speech, threshold=300))

    def test_mic_websocket_options_tolerate_slow_local_inference(self):
        test_mic = self.import_test_mic_without_running()

        options = test_mic.build_websocket_connect_kwargs()

        self.assertEqual(options["ping_interval"], 30)
        self.assertEqual(options["ping_timeout"], 120)
        self.assertEqual(options["close_timeout"], 2)


class LogisticModelTests(unittest.TestCase):
    def test_load_training_rows_maps_statuses_and_skips_unusable_rows(self):
        with TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "processed.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "rec,status,best_score,best_text,error",
                        "a,match,100,co thong tin,",
                        "b,no_match,22,khong lien quan,",
                        "c,error,,,download failed",
                        "d,no_match,0,,",
                    ]
                ),
                encoding="utf-8",
            )

            texts, labels = load_training_rows(csv_path)

        self.assertEqual(texts, ["co thong tin", "khong lien quan"])
        self.assertEqual(labels, [1, 0])

    def test_train_logistic_model_saves_pipeline_and_predicts_probability(self):
        with TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.joblib"
            recognizer = train_logistic_model(
                [
                    "de lai loi nhan sau tieng bip",
                    "vui long de lai loi nhan sau tieng bip",
                    "xin chao anh chi tu van khoan vay",
                    "bam phim mot de nghe lai",
                ],
                [1, 1, 0, 0],
                model_path,
            )

            positive = recognizer.predict(
                "quy khach vui long de lai loi nhan sau tieng bip",
                threshold=0.5,
            )
            self.assertTrue(model_path.exists())

        self.assertTrue(positive["matched"])
        self.assertEqual(positive["label"], "contains_information")
        self.assertGreaterEqual(positive["probability"], 0.5)

    def test_logistic_recognizer_applies_custom_threshold(self):
        class FakeClassifier:
            def predict_proba(self, texts):
                return [[0.35, 0.65]]

        recognizer = LogisticTextRecognizer(
            classifier=FakeClassifier(),
            model_path=Path("fake.joblib"),
        )

        low_threshold = recognizer.predict("co thong tin", threshold=0.6)
        high_threshold = recognizer.predict("co thong tin", threshold=0.8)

        self.assertTrue(low_threshold["matched"])
        self.assertFalse(high_threshold["matched"])
        self.assertEqual(high_threshold["label"], "no_information")

    def test_logistic_recognizer_normalizes_transcript_before_prediction(self):
        class RecordingClassifier:
            def __init__(self):
                self.seen_texts = None

            def predict_proba(self, texts):
                self.seen_texts = texts
                return [[0.2, 0.8]]

        classifier = RecordingClassifier()
        recognizer = LogisticTextRecognizer(
            classifier=classifier,
            model_path=Path("fake.joblib"),
        )

        recognizer.predict("Quý khách vui lòng để lại lời nhắn sau tiếng BÍP!")

        self.assertEqual(
            classifier.seen_texts,
            ["quy khach vui long de lai loi nhan sau tieng bip"],
        )


if __name__ == "__main__":
    unittest.main()

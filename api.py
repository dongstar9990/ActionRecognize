import argparse
import asyncio
import base64
import binascii
import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from logistic_model import DEFAULT_MODEL_PATH, LogisticTextRecognizer
from main import (
    DEFAULT_OUTPUT_DIR,
    TARGET_PHRASES,
    download_audio,
    load_whisper_model,
    transcribe_audio,
)
from recognizer import RealtimeRecognizer


DEFAULT_MODEL_SIZE = "medium"
DEFAULT_DEVICE = "cpu"
DEFAULT_COMPUTE_TYPE = "int8"


def recognize_audio_url(
    audio_url,
    mode="logistic",
    recognizer=None,
    logistic_recognizer=None,
    transcribe_func=None,
    model=None,
    model_size=DEFAULT_MODEL_SIZE,
    device=DEFAULT_DEVICE,
    compute_type=DEFAULT_COMPUTE_TYPE,
    language="vi",
    cache_dir=None,
    threshold=None,
):
    recognizer = recognizer or RealtimeRecognizer(TARGET_PHRASES)
    cache_dir = Path(cache_dir or DEFAULT_OUTPUT_DIR / "api_cache")
    audio_path = resolve_audio_source(audio_url, cache_dir)

    started_at = time.monotonic()
    if transcribe_func is not None:
        transcript, segments, duration = transcribe_func(audio_path)
    else:
        active_model = model or load_whisper_model(model_size, device, compute_type)
        transcript, segments, duration = transcribe_audio(active_model, audio_path, language)

    latency_ms = round((time.monotonic() - started_at) * 1000)
    if mode == "logistic":
        logistic_recognizer = logistic_recognizer or LogisticTextRecognizer(
            model_path=DEFAULT_MODEL_PATH
        )
        return logistic_recognizer.predict(
            transcript,
            threshold=0.5 if threshold is None else threshold,
            latency_ms=latency_ms,
        )
    if mode != "fuzzy":
        raise ValueError("mode must be 'logistic' or 'fuzzy'")

    return recognizer.recognize_segments(
        segments,
        transcript=transcript,
        duration=duration,
        latency_ms=latency_ms,
    )


def resolve_audio_source(audio_url, cache_dir):
    parsed = urlparse(str(audio_url))
    if parsed.scheme in ("http", "https"):
        return download_audio(audio_url, cache_dir)
    audio_path = Path(audio_url)
    if audio_path.exists():
        return audio_path
    return download_audio(audio_url, cache_dir)


def error_payload(message):
    return {"type": "error", "error": message}


def decode_websocket_audio_message(payload):
    if payload.get("type") != "audio":
        raise ValueError("expected binary audio chunk or JSON message type audio/end")

    encoded_audio = payload.get("data_base64")
    if not isinstance(encoded_audio, str) or not encoded_audio:
        raise ValueError("data_base64 is required for audio messages")

    try:
        chunk_bytes = base64.b64decode(encoded_audio, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("invalid base64 audio payload") from exc

    if not chunk_bytes:
        raise ValueError("decoded audio payload is empty")

    suffix = RealtimeRecognitionSession._safe_suffix(
        payload.get("chunk_suffix") or payload.get("suffix") or ".wav"
    )
    return chunk_bytes, suffix


async def process_realtime_chunk(session, chunk_bytes, suffix):
    return await asyncio.to_thread(session.process_chunk, chunk_bytes, suffix)


def get_cached_model(model_cache, model_size, device, compute_type, loader=load_whisper_model):
    key = (model_size, device, compute_type)
    if key not in model_cache:
        model_cache[key] = loader(model_size, device, compute_type)
    return model_cache[key]


class RealtimeRecognitionSession:
    def __init__(
        self,
        recognizer,
        logistic_recognizer=None,
        transcribe_func=None,
        model=None,
        model_size=DEFAULT_MODEL_SIZE,
        device=DEFAULT_DEVICE,
        compute_type=DEFAULT_COMPUTE_TYPE,
        language="vi",
        beam_size=5,
        vad_filter=True,
        cache_dir=None,
        mode="fuzzy",
        threshold=None,
    ):
        if mode not in ("fuzzy", "logistic"):
            raise ValueError("mode must be 'logistic' or 'fuzzy'")

        self.recognizer = recognizer
        self.logistic_recognizer = logistic_recognizer
        self.transcribe_func = transcribe_func
        self.model = model
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self.mode = mode
        self.threshold = threshold
        self.started_at = time.monotonic()
        self.chunk_index = 0
        self.duration = 0.0
        self.segments = []
        self.transcript_parts = []

        base_dir = Path(cache_dir or DEFAULT_OUTPUT_DIR / "api_cache") / "realtime"
        self.chunk_dir = base_dir / uuid.uuid4().hex
        self.chunk_dir.mkdir(parents=True, exist_ok=True)

    def process_chunk(self, chunk_bytes, suffix=".bin"):
        self.chunk_index += 1
        chunk_path = self.chunk_dir / f"chunk_{self.chunk_index:06d}{self._safe_suffix(suffix)}"
        chunk_path.write_bytes(chunk_bytes)

        if self.transcribe_func is not None:
            transcript, segments, duration = self.transcribe_func(chunk_path)
        else:
            active_model = self.model or load_whisper_model(
                self.model_size,
                self.device,
                self.compute_type,
            )
            self.model = active_model
            transcript, segments, duration = transcribe_audio(
                active_model,
                chunk_path,
                self.language,
                beam_size=self.beam_size,
                vad_filter=self.vad_filter,
            )

        offset = self.duration
        shifted_segments = [
            {
                "start": float(segment["start"]) + offset,
                "end": float(segment["end"]) + offset,
                "text": segment["text"],
            }
            for segment in segments
        ]
        self.segments.extend(shifted_segments)
        if transcript.strip():
            self.transcript_parts.append(transcript.strip())

        chunk_duration = self._chunk_duration(duration, segments)
        self.duration += chunk_duration
        full_transcript = " ".join(self.transcript_parts).strip()
        latency_ms = round((time.monotonic() - self.started_at) * 1000)

        if self.mode == "logistic":
            logistic_recognizer = self.logistic_recognizer or LogisticTextRecognizer(
                model_path=DEFAULT_MODEL_PATH
            )
            self.logistic_recognizer = logistic_recognizer
            result = logistic_recognizer.predict(
                full_transcript,
                threshold=0.5 if self.threshold is None else self.threshold,
                latency_ms=latency_ms,
            )
            result["processed_batches"] = self.chunk_index
        else:
            result = self.recognizer.recognize_segments(
                self.segments,
                transcript=full_transcript,
                duration=self.duration,
                latency_ms=latency_ms,
                processed_batches=self.chunk_index,
            )

        result["type"] = "match" if result["matched"] else "partial"
        result["chunk_index"] = self.chunk_index
        return result

    def final_result(self):
        if self.mode == "logistic":
            logistic_recognizer = self.logistic_recognizer or LogisticTextRecognizer(
                model_path=DEFAULT_MODEL_PATH
            )
            result = logistic_recognizer.predict(
                " ".join(self.transcript_parts).strip(),
                threshold=0.5 if self.threshold is None else self.threshold,
                latency_ms=round((time.monotonic() - self.started_at) * 1000),
            )
            result["processed_batches"] = self.chunk_index
        else:
            result = self.recognizer.recognize_segments(
                self.segments,
                transcript=" ".join(self.transcript_parts).strip(),
                duration=self.duration,
                latency_ms=round((time.monotonic() - self.started_at) * 1000),
                processed_batches=self.chunk_index,
            )
        result["type"] = "match" if result["matched"] else "final"
        result["chunk_index"] = self.chunk_index
        return result

    @staticmethod
    def _safe_suffix(suffix):
        if not suffix or suffix == ".":
            return ".bin"
        suffix = str(suffix)
        if not suffix.startswith("."):
            suffix = f".{suffix}"
        return suffix

    @staticmethod
    def _chunk_duration(duration, segments):
        if duration is not None:
            return float(duration)
        if segments:
            return max(float(segment["end"]) for segment in segments)
        return 0.0


def build_api_app(
    recognizer=None,
    transcribe_func=None,
    model=None,
    model_size=DEFAULT_MODEL_SIZE,
    device=DEFAULT_DEVICE,
    compute_type=DEFAULT_COMPUTE_TYPE,
    language="vi",
    cache_dir=None,
):
    try:
        from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise RuntimeError(
            "Missing API dependencies. Install fastapi and uvicorn to serve the realtime API."
        ) from exc

    class RecognizeRequest(BaseModel):
        audio_url: str
        target_phrases: list[str] | None = None
        threshold: float | None = Field(default=None, ge=0)
        mode: str = "logistic"

    app = FastAPI(title="ActionRecognize Realtime API")
    app.state.recognizer = recognizer or RealtimeRecognizer(TARGET_PHRASES)
    app.state.model = model
    app.state.models = {}
    if model is not None:
        app.state.models[(model_size, device, compute_type)] = model
    app.state.transcribe_func = transcribe_func
    app.state.cache_dir = Path(cache_dir or DEFAULT_OUTPUT_DIR / "api_cache")

    @app.post("/recognize")
    def recognize(request: RecognizeRequest):
        active_recognizer = app.state.recognizer
        if request.target_phrases is not None or request.threshold is not None:
            active_recognizer = RealtimeRecognizer(
                target_phrases=request.target_phrases or active_recognizer.target_phrases,
                threshold=(
                    request.threshold
                    if request.threshold is not None
                    else active_recognizer.threshold
                ),
                padding=active_recognizer.padding,
                max_window_segments=active_recognizer.max_window_segments,
            )

        try:
            if app.state.model is None and app.state.transcribe_func is None:
                app.state.model = get_cached_model(
                    app.state.models,
                    model_size,
                    device,
                    compute_type,
                )
            return recognize_audio_url(
                request.audio_url,
                mode=request.mode,
                recognizer=active_recognizer,
                logistic_recognizer=LogisticTextRecognizer(),
                transcribe_func=app.state.transcribe_func,
                model=app.state.model,
                model_size=model_size,
                device=device,
                compute_type=compute_type,
                language=language,
                cache_dir=app.state.cache_dir,
                threshold=request.threshold,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.websocket("/recognize/ws")
    async def recognize_realtime(websocket: WebSocket):
        await websocket.accept()
        try:
            config = await websocket.receive_json()
        except Exception:
            await websocket.send_json(error_payload("initial JSON config is required"))
            await websocket.close(code=1003)
            return

        mode = config.get("mode", "fuzzy")
        stop_on_match = config.get("stop_on_match", True)
        threshold = config.get("threshold")
        realtime_model_size = config.get("model_size", model_size)
        realtime_device = config.get("device", device)
        realtime_compute_type = config.get("compute_type", compute_type)
        realtime_beam_size = int(config.get("beam_size", 5))
        realtime_vad_filter = bool(config.get("vad_filter", True))
        active_recognizer = app.state.recognizer
        if config.get("target_phrases") is not None or threshold is not None:
            active_recognizer = RealtimeRecognizer(
                target_phrases=config.get("target_phrases") or active_recognizer.target_phrases,
                threshold=threshold if threshold is not None else active_recognizer.threshold,
                padding=active_recognizer.padding,
                max_window_segments=active_recognizer.max_window_segments,
            )

        try:
            active_model = app.state.model
            if app.state.transcribe_func is None:
                active_model = get_cached_model(
                    app.state.models,
                    realtime_model_size,
                    realtime_device,
                    realtime_compute_type,
                )

            session = RealtimeRecognitionSession(
                recognizer=active_recognizer,
                logistic_recognizer=LogisticTextRecognizer(),
                transcribe_func=app.state.transcribe_func,
                model=active_model,
                model_size=realtime_model_size,
                device=realtime_device,
                compute_type=realtime_compute_type,
                language=config.get("language", language),
                beam_size=realtime_beam_size,
                vad_filter=realtime_vad_filter,
                cache_dir=app.state.cache_dir,
                mode=mode,
                threshold=threshold,
            )

            while True:
                message = await websocket.receive()
                if "bytes" in message and message["bytes"] is not None:
                    result = await process_realtime_chunk(
                        session,
                        message["bytes"],
                        config.get("chunk_suffix", ".wav"),
                    )
                    await websocket.send_json(result)
                    if result["matched"] and stop_on_match:
                        await websocket.close(code=1000)
                        return
                elif "text" in message and message["text"] is not None:
                    payload = json.loads(message["text"])
                    if payload.get("type") == "end":
                        await websocket.send_json(session.final_result())
                        await websocket.close(code=1000)
                        return
                    chunk_bytes, suffix = decode_websocket_audio_message(payload)
                    result = await process_realtime_chunk(session, chunk_bytes, suffix)
                    await websocket.send_json(result)
                    if result["matched"] and stop_on_match:
                        await websocket.close(code=1000)
                        return
        except WebSocketDisconnect:
            return
        except (json.JSONDecodeError, ValueError) as exc:
            await websocket.send_json(error_payload(str(exc)))
        except Exception as exc:
            await websocket.send_json(error_payload(str(exc)))
            await websocket.close(code=1011)

    return app


class RecognitionHandler(BaseHTTPRequestHandler):
    recognizer = RealtimeRecognizer(TARGET_PHRASES)
    logistic_recognizer = LogisticTextRecognizer()
    model = None
    model_size = DEFAULT_MODEL_SIZE
    device = DEFAULT_DEVICE
    compute_type = DEFAULT_COMPUTE_TYPE
    language = "vi"
    cache_dir = DEFAULT_OUTPUT_DIR / "api_cache"

    def do_POST(self):
        if urlparse(self.path).path != "/recognize":
            self._write_json(404, {"error": "not found"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(content_length) or b"{}")
            audio_url = payload["audio_url"]
            mode = payload.get("mode", "logistic")
            recognizer = self.recognizer
            if payload.get("target_phrases") is not None or payload.get("threshold") is not None:
                recognizer = RealtimeRecognizer(
                    target_phrases=payload.get("target_phrases") or recognizer.target_phrases,
                    threshold=payload.get("threshold", recognizer.threshold),
                    padding=recognizer.padding,
                    max_window_segments=recognizer.max_window_segments,
                )
            if self.model is None:
                self.__class__.model = load_whisper_model(
                    self.model_size,
                    self.device,
                    self.compute_type,
                )
            result = recognize_audio_url(
                audio_url,
                mode=mode,
                recognizer=recognizer,
                logistic_recognizer=self.logistic_recognizer,
                model=self.model,
                model_size=self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                language=self.language,
                cache_dir=self.cache_dir,
                threshold=payload.get("threshold"),
            )
        except KeyError:
            self._write_json(400, {"error": "audio_url is required"})
            return
        except Exception as exc:
            self._write_json(400, {"error": str(exc)})
            return

        self._write_json(200, result)

    def log_message(self, format, *args):
        return

    def _write_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_server(host="127.0.0.1", port=8000):
    server = ThreadingHTTPServer((host, port), RecognitionHandler)
    print(f"Serving realtime recognition API at http://{host}:{port}")
    server.serve_forever()


try:
    app = build_api_app()
except RuntimeError as exc:
    print(f"[api] cannot build app: {exc}")
    app = None


def main(argv=None):
    parser = argparse.ArgumentParser(description="Serve the realtime recognition JSON API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    run_server(args.host, args.port)


if __name__ == "__main__":
    main()

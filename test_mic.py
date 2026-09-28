import asyncio
import io
import json
import math
import wave


SR = 16000
CHUNK_SEC = 1
WS_URL = "ws://127.0.0.1:8000/recognize/ws"
MIN_SPEECH_RMS = 300
NOISE_MULTIPLIER = 3
PING_INTERVAL = 30
PING_TIMEOUT = 120
CLOSE_TIMEOUT = 2

UNREACHABLE_TARGET_PHRASES = [
    "thuê bao quý khách vừa gọi tạm thời không liên lạc được",
    "thuê bao quý khách vừa gọi tạm thời không liên lạc được xin vui lòng gọi lại sau",
    "thuê bao quý khách vừa gọi hiện không liên lạc được",
    "thuê bao quý khách vừa gọi hiện đang bận",
]


def build_realtime_config():
    return {
        "mode": "fuzzy",
        "threshold": 70,
        "stop_on_match": True,
        "chunk_suffix": ".wav",
        "language": "vi",
        "model_size": "tiny",
        "device": "cpu",
        "compute_type": "int8",
        "beam_size": 1,
        "vad_filter": False,
        "target_phrases": UNREACHABLE_TARGET_PHRASES,
    }


def build_websocket_connect_kwargs():
    return {
        "ping_interval": PING_INTERVAL,
        "ping_timeout": PING_TIMEOUT,
        "close_timeout": CLOSE_TIMEOUT,
    }


def to_wav_bytes(pcm_int16):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SR)
        wav_file.writeframes(pcm_int16.tobytes())
    return buf.getvalue()


def audio_rms(pcm_int16):
    if len(pcm_int16) == 0:
        return 0.0
    square_sum = sum(float(sample) * float(sample) for sample in pcm_int16)
    return math.sqrt(square_sum / len(pcm_int16))


def should_send_audio(pcm_int16, threshold):
    return audio_rms(pcm_int16) >= threshold


def calibrate_noise(sd):
    print("Giữ yên 1 giây để đo nhiễu nền...", flush=True)
    audio = sd.rec(
        SR,
        samplerate=SR,
        channels=1,
        dtype="int16",
        blocking=True,
    )
    noise_rms = audio_rms(audio[:, 0])
    threshold = max(MIN_SPEECH_RMS, noise_rms * NOISE_MULTIPLIER)
    print(f"noise_rms={noise_rms:.1f} speech_threshold={threshold:.1f}", flush=True)
    return threshold


def format_result(result):
    chunk_index = result.get("chunk_index", "?")
    if result.get("matched"):
        score = result.get("match_score", result.get("probability", "?"))
        text = result.get("matched_text") or result.get("transcript", "")
        return f"[{chunk_index}] MATCH score={score} text={text}"

    score = result.get("best_score", result.get("probability", "?"))
    text = result.get("best_text") or result.get("transcript", "")
    return f"[{chunk_index}] {result.get('type', 'partial')} best={score} text={text}"


async def main():
    import sounddevice as sd
    import websockets
    from websockets.exceptions import ConnectionClosed

    async with websockets.connect(WS_URL, **build_websocket_connect_kwargs()) as ws:
        await ws.send(json.dumps(build_realtime_config(), ensure_ascii=False))
        speech_threshold = calibrate_noise(sd)
        skipped_chunks = 0
        print("Đang nghe realtime... nói đi, Ctrl+C để dừng")

        try:
            while True:
                audio = await asyncio.to_thread(
                    lambda: sd.rec(
                        int(SR * CHUNK_SEC),
                        samplerate=SR,
                        channels=1,
                        dtype="int16",
                        blocking=True,
                    )
                )
                pcm = audio[:, 0]
                rms = audio_rms(pcm)
                if not should_send_audio(pcm, speech_threshold):
                    skipped_chunks += 1
                    if skipped_chunks == 1 or skipped_chunks % 5 == 0:
                        print(
                            f"skip silence rms={rms:.1f} threshold={speech_threshold:.1f}",
                            flush=True,
                        )
                    continue

                skipped_chunks = 0
                await ws.send(to_wav_bytes(pcm))
                try:
                    result = json.loads(await ws.recv())
                except ConnectionClosed as exc:
                    print(f"WebSocket closed: {exc}", flush=True)
                    return
                print(format_result(result), flush=True)
                if result.get("matched"):
                    return
        except asyncio.CancelledError:
            try:
                await ws.send(json.dumps({"type": "end"}))
                result = json.loads(await asyncio.wait_for(ws.recv(), timeout=2))
                print(format_result(result), flush=True)
            except Exception:
                pass
            raise


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Đã dừng.")

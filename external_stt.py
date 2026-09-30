"""
Tích hợp API STT ngoài (g-ailab.com) vào service nhận diện realtime hiện có.

Cách dùng nhanh (chỉ gọi STT, không qua fuzzy/logistic recognizer):
    from external_stt import transcribe_with_external_api
    text = transcribe_with_external_api("path/to/audio.wav")

Cách dùng tích hợp vào pipeline có sẵn (recognize_audio_url / RealtimeRecognitionSession):
    truyền transcribe_func=external_transcribe_func khi khởi tạo app hoặc session.
"""

import io
import os
from pathlib import Path

import requests

# Nên để trong biến môi trường, không hardcode trong code khi deploy thật.
EXTERNAL_STT_URL = "https://api.g-ailab.com/api/v1/stt"
EXTERNAL_STT_API_KEY = os.environ.get(
    "EXTERNAL_STT_API_KEY",
    "Gspeech_5GlHt0m40rqTwLlJxv19WkfMV6wZMj2ydQHCDoAQQsA",  # tạm thời để đây theo curl bạn gửi
)


class ExternalSTTError(RuntimeError):
    """Raised khi gọi API STT ngoài thất bại hoặc response không hợp lệ."""


def transcribe_with_external_api(
        audio_path,
        api_key: str | None = None,
        timeout: float = 60.0,
        extra_fields: dict | None = None,
) -> dict:
    """
    Gọi API STT ngoài, tương đương với:
        curl --location 'https://api.g-ailab.com/api/v1/stt' \
             --header 'Authorization: Bearer <api_key>' \
             --form 'file=@"<audio_path>"'

    Trả về dict JSON gốc từ API (đã raise lỗi nếu status != 200).
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"audio file not found: {audio_path}")

    headers = {
        "Authorization": f"Bearer {api_key or EXTERNAL_STT_API_KEY}",
    }

    with audio_path.open("rb") as f:
        files = {"file": (audio_path.name, f, _guess_mime(audio_path))}
        response = requests.post(
            EXTERNAL_STT_URL,
            headers=headers,
            files=files,
            data=extra_fields or {},
            timeout=timeout,
        )

    if response.status_code != 200:
        raise ExternalSTTError(
            f"external STT API returned {response.status_code}: {response.text}"
        )

    try:
        return response.json()
    except ValueError as exc:
        raise ExternalSTTError(f"invalid JSON response: {response.text}") from exc


def transcribe_bytes_with_external_api(
        audio_bytes: bytes,
        filename: str = "audio.wav",
        api_key: str | None = None,
        timeout: float = 60.0,
        extra_fields: dict | None = None,
) -> dict:
    """
    Giống transcribe_with_external_api nhưng nhận thẳng bytes audio trong RAM,
    KHÔNG cần ghi ra file tạm trên ổ đĩa trước. Dùng cho realtime/websocket để
    tránh I/O ổ đĩa cho từng chunk nhỏ.
    """
    if not audio_bytes:
        raise ValueError("audio_bytes is empty")

    headers = {
        "Authorization": f"Bearer {api_key or EXTERNAL_STT_API_KEY}",
    }

    files = {"file": (filename, io.BytesIO(audio_bytes), _guess_mime(Path(filename)))}
    response = requests.post(
        EXTERNAL_STT_URL,
        headers=headers,
        files=files,
        data=extra_fields or {},
        timeout=timeout,
    )

    if response.status_code != 200:
        raise ExternalSTTError(
            f"external STT API returned {response.status_code}: {response.text}"
        )

    try:
        return response.json()
    except ValueError as exc:
        raise ExternalSTTError(f"invalid JSON response: {response.text}") from exc


def _guess_mime(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".m4a": "audio/mp4",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
        ".webm": "audio/webm",
    }.get(suffix, "application/octet-stream")


def _extract_transcript(payload: dict) -> str:
    """
    Đọc transcript từ response thật của API g-ailab.com, ví dụ:
        {"text": "...", "segments": [...], "duration_sec": 0.36, ...}
    """
    if not isinstance(payload, dict):
        return str(payload)
    return payload.get("text", "") or ""


def _extract_segments(payload: dict) -> list[dict]:
    """
    Đọc segments từ response thật, mỗi segment có dạng:
        {"text": "...", "start_time": 0.0, "end_time": 17.14}
    Chuẩn hóa về format mà phần còn lại của pipeline (RealtimeRecognizer) đang dùng:
        [{"start": float, "end": float, "text": str}, ...]
    """
    segments = payload.get("segments") if isinstance(payload, dict) else None
    if not segments:
        return []

    normalized = []
    for seg in segments:
        try:
            normalized.append(
                {
                    "start": float(seg.get("start_time", 0.0)),
                    "end": float(seg.get("end_time", 0.0)),
                    "text": seg.get("text", ""),
                }
            )
        except (TypeError, ValueError):
            continue
    return normalized


def _extract_audio_duration(payload: dict) -> float:
    """
    LƯU Ý: field "duration_sec" trong response của API này là thời gian XỬ LÝ
    của API (vd 0.36s), KHÔNG PHẢI độ dài audio thật. Độ dài audio thật phải lấy
    từ end_time của segment cuối cùng (vd 17.14s).
    """
    segments = payload.get("segments") if isinstance(payload, dict) else None
    if segments:
        try:
            return max(float(seg.get("end_time", 0.0)) for seg in segments)
        except (TypeError, ValueError):
            pass
    return 0.0


def external_transcribe_func(audio_path):
    """
    Hàm tương thích với chữ ký `transcribe_func(audio_path) -> (transcript, segments, duration)`
    mà `recognize_audio_url` và `RealtimeRecognitionSession` trong api_server.py đang mong đợi.

    Cách dùng:
        from api_server import recognize_audio_url
        from external_stt import external_transcribe_func

        result = recognize_audio_url(
            "path/to/audio.wav",
            mode="logistic",
            transcribe_func=external_transcribe_func,
        )
    """
    payload = transcribe_with_external_api(audio_path)

    transcript = _extract_transcript(payload)
    segments = _extract_segments(payload)
    duration = _extract_audio_duration(payload)  # độ dài audio thật, lấy từ segments

    return transcript, segments, duration


def external_transcribe_bytes_func(chunk_bytes: bytes, suffix: str = ".wav"):
    """
    Hàm tương thích với chữ ký `transcribe_bytes_func(chunk_bytes, suffix) ->
    (transcript, segments, duration)` mà `RealtimeRecognitionSession` dùng khi
    KHÔNG muốn ghi chunk ra ổ đĩa. Gửi thẳng bytes trong RAM lên API ngoài.
    """
    filename = f"chunk{suffix or '.wav'}"
    payload = transcribe_bytes_with_external_api(chunk_bytes, filename=filename)

    transcript = _extract_transcript(payload)
    segments = _extract_segments(payload)
    duration = _extract_audio_duration(payload)

    return transcript, segments, duration


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print("Usage: python external_stt.py <audio_file>")
        raise SystemExit(1)

    result = transcribe_with_external_api(sys.argv[1])
    print(json.dumps(result, ensure_ascii=False, indent=2))
import argparse
import csv
import hashlib
import shutil
import subprocess
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from urllib.request import Request, urlopen


# Each entry is one variant of a notice; a call matches if ANY of them is found.
# When several variants score equally, the LONGEST one wins, so the clip keeps the
# full sentence ("... xin vui lòng gọi lại sau") whenever it is present.

# Current target: voicemail notice.
VOICEMAIL_PHRASES = [
    "để lại lời nhắn sau tiếng bíp",
    "quý khách vui lòng để lại lời nhắn sau tiếng bíp",
    "quý khách vui lòng để lại lời nhắn sau tiếng bíp xin vui lòng gọi lại sau",
]

# Old target: "subscriber busy / unreachable" notices (kept for reuse).
UNREACHABLE_PHRASES = [
    # --- đang bận ---
    "thuê bao quý khách vừa gọi hiện đang bận",
    "thuê bao quý khách vừa gọi hiện đang bận xin vui lòng gọi lại sau",
    "thuê bao quý khách vừa gọi hiện đang bận xin quý khách vui lòng gọi lại sau",
    # --- không liên lạc được ---
    "thuê bao quý khách vừa gọi tạm thời không liên lạc được",
    "thuê bao quý khách vừa gọi tạm thời không liên lạc được "
    "xin vui lòng gọi lại sau",
    "thuê bao quý khách vừa gọi tạm thời không liên lạc được "
    "xin quý khách vui lòng gọi lại sau",
    "thuê bao quý khách vừa gọi hiện không liên lạc được",
    "thuê bao quý khách vừa gọi hiện không liên lạc được "
    "xin quý khách vui lòng gọi lại sau",
    # --- tắt máy / ngoài vùng phủ sóng (wording not verified against your audio) ---
    "thuê bao quý khách vừa gọi hiện đã tắt máy",
    "thuê bao quý khách vừa gọi hiện nằm ngoài vùng phủ sóng",
]

# Which phrases to search for. To search for both kinds of notice use:
#   TARGET_PHRASES = VOICEMAIL_PHRASES + UNREACHABLE_PHRASES
TARGET_PHRASES = VOICEMAIL_PHRASES
DEFAULT_OUTPUT_DIR = Path("output")
USER_AGENT = "ActionRecognize-STT/1.0"

RESULT_FIELDS = [
    "rec", "duration", "billsec", "transcript", "matched_text",
    "matched_phrase", "match_score", "start_sec", "end_sec", "clip_path",
]
LOG_FIELDS = ["rec", "status", "best_score", "best_text", "error"]


# ---------- text matching ----------

def normalize_text(text):
    text = text.casefold()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.replace("đ", "d")
    text = "".join(ch if ch.isalnum() else " " for ch in text)
    return " ".join(text.split())


def fuzzy_score(candidate, target):
    candidate = normalize_text(candidate)
    target = normalize_text(target)
    if not candidate or not target:
        return 0
    if target in candidate:
        return 100

    base_score = SequenceMatcher(None, candidate, target).ratio()
    partial_score = _partial_ratio(candidate, target)
    # Token overlap ignores word order, so it is down-weighted: it must not be
    # able to pass the threshold on its own.
    token_score = _token_overlap_score(candidate, target) * 0.8
    return round(max(base_score, partial_score, token_score) * 100)


def _partial_ratio(candidate, target):
    candidate_words = candidate.split()
    target_words = target.split()
    if len(candidate_words) <= len(target_words):
        return SequenceMatcher(None, candidate, target).ratio()

    window_size = len(target_words)
    best = 0.0
    for start in range(0, len(candidate_words) - window_size + 1):
        window = " ".join(candidate_words[start : start + window_size])
        best = max(best, SequenceMatcher(None, window, target).ratio())
    return best


def _token_overlap_score(candidate, target):
    candidate_tokens = set(candidate.split())
    target_tokens = set(target.split())
    if not candidate_tokens or not target_tokens:
        return 0.0
    return len(candidate_tokens & target_tokens) / len(target_tokens)


def find_best_match(
    segments,
    target_phrases=TARGET_PHRASES,
    max_window_segments=6,
    threshold=None,
):
    """Return the best-scoring window over all target phrases (even if low),
    or None if there is no text at all. The caller compares match["score"]
    with the threshold; match["phrase"] says which phrase matched."""
    if isinstance(target_phrases, str):
        target_phrases = [target_phrases]
    best_match = None
    clean_segments = [
        {
            "start": float(s["start"]),
            "end": float(s["end"]),
            "text": str(s.get("text", "")).strip(),
        }
        for s in segments
        if str(s.get("text", "")).strip()
    ]

    for start_index in range(len(clean_segments)):
        combined_text = []
        for end_index in range(
                start_index, min(len(clean_segments), start_index + max_window_segments)
        ):
            combined_text.append(clean_segments[end_index]["text"])
            text = " ".join(combined_text)
            span = clean_segments[end_index]["end"] - clean_segments[start_index]["start"]
            phrase, score = max(
                ((ph, fuzzy_score(text, ph)) for ph in target_phrases),
                key=lambda item: (item[1], len(item[0].split())),
            )
            # Rank: higher score, then the longer phrase (complete sentence),
            # then the tighter window (no extra speech such as "Alo").
            rank = (score, len(phrase.split()), -span)
            if best_match is None or rank > best_match["rank"]:
                best_match = {
                    "score": score,
                    "start": clean_segments[start_index]["start"],
                    "end": clean_segments[end_index]["end"],
                    "text": text,
                    "phrase": phrase,
                    "rank": rank,
                }
    if best_match is not None and threshold is not None and best_match["score"] < threshold:
        return None
    return best_match


def planned_clip_window(start, end, duration=None, padding=0.5):
    clip_start = max(0.0, round(float(start) - padding, 3))
    clip_end = round(float(end) + padding, 3)
    if duration is not None:
        clip_end = min(round(float(duration), 3), clip_end)
    if clip_end <= clip_start:
        clip_end = round(float(end), 3)
    return clip_start, clip_end


# ---------- IO ----------

def parse_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_records(csv_path, max_billsec=None):
    with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        for row_number, row in enumerate(reader, start=2):
            rec = (row.get("rec") or "").strip()
            if not rec:
                continue
            if max_billsec is not None:
                billsec = parse_float(row.get("billsec"))
                if billsec is not None and billsec > max_billsec:
                    continue
            yield row_number, row


def stable_audio_name(url):
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    suffix = Path(url.split("?", 1)[0]).suffix or ".mp3"
    return f"{digest}{suffix}"


def download_audio(url, cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / stable_audio_name(url)
    if destination.exists() and destination.stat().st_size > 0:
        return destination

    request = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=60) as response:
            data = response.read()
    except OSError as exc:  # URLError, HTTPError, timeouts
        raise RuntimeError(f"download failed: {exc}") from exc

    if not data:
        raise RuntimeError("downloaded audio is empty")
    tmp = destination.with_suffix(destination.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(destination)  # never leave a half-written file in the cache
    return destination


def load_done(log_path):
    if not log_path.exists():
        return set()
    with log_path.open("r", newline="", encoding="utf-8-sig") as f:
        return {r["rec"] for r in csv.DictReader(f) if r.get("status") in ("match", "no_match")}


def open_csv_append(path, fieldnames):
    is_new = not path.exists() or path.stat().st_size == 0
    f = path.open("a", newline="", encoding="utf-8-sig")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if is_new:
        writer.writeheader()
    return f, writer


# ---------- audio ----------

def load_whisper_model(model_size, device, compute_type):
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: faster-whisper. Run: pip install faster-whisper"
        ) from exc
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def transcribe_audio(model, audio_path, language, beam_size=5, vad_filter=True):
    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        vad_filter=vad_filter,
        beam_size=beam_size,
        condition_on_previous_text=False,  # avoids repeated/hallucinated loops
    )
    segment_rows = [
        {"start": s.start, "end": s.end, "text": s.text} for s in segments
    ]
    transcript = " ".join(s["text"].strip() for s in segment_rows).strip()
    return transcript, segment_rows, getattr(info, "duration", None)


def crop_audio(source_path, clip_path, start, end):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found in PATH")

    clip_path.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.01, round(end - start, 3))
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start:.3f}", "-i", str(source_path),
        "-t", f"{duration:.3f}",
        "-c:a", "libmp3lame", "-q:a", "2",  # re-encode: exact cut points
        str(clip_path),
    ]
    subprocess.run(command, check=True)


# ---------- main flow ----------

def process_records(args):
    output_dir = args.output_dir
    cache_dir = output_dir / "cache_3"
    clips_dir = output_dir / "clips_3"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "matched_records_2.csv"
    log_path = output_dir / "processed_log_2.csv"

    done = load_done(log_path)  # resume: skip records already handled
    model = load_whisper_model(args.model_size, args.device, args.compute_type)

    processed = matched = 0
    results_file, results_writer = open_csv_append(results_path, RESULT_FIELDS)
    log_file, log_writer = open_csv_append(log_path, LOG_FIELDS)

    try:
        for row_number, row in read_records(args.input, args.max_billsec):
            rec = row["rec"].strip()
            if rec in done:
                continue
            if args.limit is not None and processed >= args.limit:
                break
            processed += 1

            try:
                audio_path = download_audio(rec, cache_dir)
                transcript, segments, detected_duration = transcribe_audio(
                    model, audio_path, args.language
                )
                match = find_best_match(
                    segments,
                    target_phrases=args.target_phrases,
                    max_window_segments=args.max_window_segments,
                )

                if match is None or match["score"] < args.threshold:
                    log_writer.writerow({
                        "rec": rec, "status": "no_match",
                        "best_score": match["score"] if match else 0,
                        "best_text": match["text"] if match else "", "error": "",
                    })
                    log_file.flush()
                    print(f"[{processed}] no match (best={match['score'] if match else 0}): row {row_number}", flush=True)
                    continue

                # Prefer the real audio length from whisper over the CDR duration
                duration = detected_duration or parse_float(row.get("duration"))
                start, end = planned_clip_window(
                    match["start"], match["end"], duration=duration, padding=args.padding
                )
                clip_path = clips_dir / f"{Path(audio_path).stem}_{start:.3f}_{end:.3f}.mp3"
                crop_audio(audio_path, clip_path, start, end)
                matched += 1

                results_writer.writerow({
                    "rec": rec,
                    "duration": row.get("duration", ""),
                    "billsec": row.get("billsec", ""),
                    "transcript": transcript,
                    "matched_text": match["text"],
                    "matched_phrase": match["phrase"],
                    "match_score": match["score"],
                    "start_sec": start,
                    "end_sec": end,
                    "clip_path": str(clip_path),
                })
                log_writer.writerow({
                    "rec": rec, "status": "match", "best_score": match["score"],
                    "best_text": match["text"], "error": "",
                })
                results_file.flush()
                log_file.flush()
                print(f"[{processed}] match score={match['score']} clip={clip_path}", flush=True)
            except Exception as exc:
                # Errors are NOT marked done, so the next run retries them
                log_writer.writerow({
                    "rec": rec, "status": "error", "best_score": "",
                    "best_text": "", "error": str(exc),
                })
                log_file.flush()
                print(f"[{processed}] error row {row_number}: {exc}", file=sys.stderr)
    finally:
        results_file.close()
        log_file.close()

    return results_path, processed, matched


def build_parser():
    parser = argparse.ArgumentParser(
        description="Transcribe mp3 records, find unreachable-subscriber notices, and crop matched clips."
    )
    parser.add_argument("--input", type=Path, default=Path("stt.csv"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-size", default="small", help="faster-whisper model size (small/medium/large-v3)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--language", default="vi")
    parser.add_argument("--threshold", type=int, default=70)
    parser.add_argument("--padding", type=float, default=0.5)
    parser.add_argument("--max-window-segments", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None, help="only process N new records (for testing)")
    parser.add_argument("--max-billsec", type=float, default=None,
                        help="skip calls answered longer than this (e.g. 0 = only unanswered calls)")
    parser.add_argument("--target-phrase", action="append", default=None,
                        help="override target phrases; repeat the flag for several phrases")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.target_phrases = args.target_phrase or TARGET_PHRASES
    results_path, processed, matched = process_records(args)
    print(f"Processed {processed} records, matched {matched}. Results: {results_path}")


if __name__ == "__main__":
    main()

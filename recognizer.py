import time

from main import find_best_match, planned_clip_window


class RealtimeRecognizer:
    """Turn timestamped STT segments into JSON-ready recognition results."""

    def __init__(
        self,
        target_phrases,
        threshold=70,
        padding=0.5,
        max_window_segments=6,
    ):
        self.target_phrases = target_phrases
        self.threshold = threshold
        self.padding = padding
        self.max_window_segments = max_window_segments

    def recognize_segments(
        self,
        segments,
        transcript="",
        duration=None,
        latency_ms=None,
        processed_batches=None,
    ):
        best = find_best_match(
            segments,
            target_phrases=self.target_phrases,
            max_window_segments=self.max_window_segments,
        )
        if best is None or best["score"] < self.threshold:
            return {
                "matched": False,
                "transcript": transcript,
                "matched_text": "",
                "matched_phrase": "",
                "match_score": 0,
                "best_score": best["score"] if best else 0,
                "best_text": best["text"] if best else "",
                "start_sec": None,
                "end_sec": None,
                "latency_ms": latency_ms,
                "processed_batches": processed_batches,
            }

        start, end = planned_clip_window(
            best["start"],
            best["end"],
            duration=duration,
            padding=self.padding,
        )
        return {
            "matched": True,
            "transcript": transcript,
            "matched_text": best["text"],
            "matched_phrase": best["phrase"],
            "match_score": best["score"],
            "best_score": best["score"],
            "best_text": best["text"],
            "start_sec": start,
            "end_sec": end,
            "latency_ms": latency_ms,
            "processed_batches": processed_batches,
        }

    def recognize_batches(self, segment_batches, duration=None):
        started_at = time.monotonic()
        segments = []
        transcript_parts = []

        for index, batch in enumerate(segment_batches, start=1):
            segments.extend(batch)
            transcript_parts.extend(
                str(segment.get("text", "")).strip()
                for segment in batch
                if str(segment.get("text", "")).strip()
            )
            result = self.recognize_segments(
                segments,
                transcript=" ".join(transcript_parts),
                duration=duration,
                latency_ms=round((time.monotonic() - started_at) * 1000),
                processed_batches=index,
            )
            if result["matched"]:
                return result

        return self.recognize_segments(
            segments,
            transcript=" ".join(transcript_parts),
            duration=duration,
            latency_ms=round((time.monotonic() - started_at) * 1000),
            processed_batches=len(segment_batches),
        )

import re
from fastapi import FastAPI, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="Omi Hackathon API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ──────────────────────────────────────────────────────────────

class Segment(BaseModel):
    """A single spoken segment from the transcript."""
    text: str
    speaker: str = "unknown"
    start: float = 0.0
    end: float = 0.0

class OmiWebhookPayload(BaseModel):
    """Payload received from the Omi webhook."""
    segments: list[Segment] = Field(default_factory=list)
    transcript: str = Field(default="")
    summary: str = Field(default="")


# ── Helpers ─────────────────────────────────────────────────────────────

def extract_segments(payload: OmiWebhookPayload):
    """
    Returns a list of (text, start, end) tuples from segments or transcript.
    """
    if payload.segments:
        return [(seg.text, seg.start, seg.end) for seg in payload.segments]
    # fallback: treat whole transcript as one segment
    text = payload.transcript.strip()
    if text:
        return [(text, 0.0, 0.0)]
    return []

def format_transcript_with_speakers(payload: OmiWebhookPayload) -> str:
    """
    Build a readable transcript with speaker labels and timestamps.
    Supports three input styles:
      1. Structured segments with speaker + timestamps
      2. Raw transcript text (auto-labels as S1)
      3. Inline speaker markers like "Speaker 1: ..." already in text
    """
    if payload.segments:
        lines: list[str] = []
        speaker_map: dict[str, str] = {}
        counter = 1
        for seg in payload.segments:
            key = seg.speaker or "unknown"
            if key not in speaker_map:
                speaker_map[key] = f"S{counter}"
                counter += 1
            label = speaker_map[key]
            ts = f"[{seg.start:.1f}s–{seg.end:.1f}s]"
            lines.append(f"{ts} {label}: {seg.text.strip()}")
        return "\n".join(lines)

    # Fallback: plain transcript string
    text = payload.transcript.strip()
    if not text:
        return ""

    # If the text already has speaker markers, normalise them
    if re.search(r"(?i)(speaker\s*\d|s\d\s*:)", text):
        return text

    # Otherwise treat the whole thing as a single speaker
    return f"[0.0s] S1: {text}"


async def analyze_transcript(formatted: str) -> dict:
    """Send the formatted transcript to Gemini for analysis."""
    prompt = (
        "You are an expert conversation analyst.\n\n"
        "Below is a transcript where each line is formatted as:\n"
        "  [start–end] SpeakerLabel: text\n\n"
        f"Transcript:\n{formatted}\n\n"
        "Instructions:\n"
        "1. Identify each unique speaker (S1, S2, …) and give a short role "
        "   description if possible (e.g. 'Interviewer', 'Customer').\n"
        "2. Produce a concise headline summarising the conversation.\n"
        "3. List 3 key points from the conversation, noting which speaker "
        "   said what and at which timestamp.\n\n"
        "Respond in this exact JSON format (no markdown fences):\n"
        "{\n"
        '  "speakers": {"S1": "Role/description", "S2": "Role/description"},\n'
        '  "headline": "...",\n'
        '  "key_points": [\n'
        '    {"timestamp": "0.0s", "speaker": "S1", "point": "..."},\n'
        '    {"timestamp": "...", "speaker": "...", "point": "..."},\n'
        '    {"timestamp": "...", "speaker": "...", "point": "..."}\n'
        "  ]\n"
        "}"
    )
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        for attempt in range(3):
            try:
                response = await asyncio.to_thread(model.generate_content, prompt)
                break
            except Exception as retry_err:
                if "429" in str(retry_err) and attempt < 2:
                    wait = (attempt + 1) * 5
                    logger.warning("Rate limited, retrying in %ds…", wait)
                    await asyncio.sleep(wait)
                else:
                    raise
        raw = response.text.strip()
        # Strip markdown code fences if Gemini wraps them
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        import json
        return json.loads(raw)
    except Exception as e:
        logger.error("Gemini call failed: %s", e)
        raise RuntimeError(f"Gemini AI call failed: {e}")


def mock_analysis(formatted: str) -> dict:
    """Return a realistic mock response for testing when Gemini quota is exhausted."""
    # Extract speakers from the formatted transcript
    speakers_found = set(re.findall(r"\b(S\d+)\b", formatted))
    speakers = {s: f"Speaker {s}" for s in sorted(speakers_found)} or {"S1": "Speaker"}
    lines = [l.strip() for l in formatted.split("\n") if l.strip()]

    key_points = []
    for line in lines[:3]:
        ts_match = re.search(r"\[([\d.]+)s", line)
        sp_match = re.search(r"(S\d+):", line)
        text = re.sub(r"^\[.*?\]\s*S\d+:\s*", "", line)
        key_points.append({
            "timestamp": ts_match.group(1) + "s" if ts_match else "0.0s",
            "speaker": sp_match.group(1) if sp_match else "S1",
            "point": text[:80],
        })

    return {
        "speakers": speakers,
        "headline": f"Conversation between {len(speakers)} speakers ({', '.join(sorted(speakers))})",
        "key_points": key_points,
        "_demo_mode": True,
    }


# ── Routes ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/webhook")
async def webhook(
    payload: OmiWebhookPayload,
    user_id: str = Query(..., min_length=1),
):
    formatted = format_transcript_with_speakers(payload)
    if not formatted:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No transcript content provided.",
        )

    # --- SOS/Police keyword detection ---
    segments = extract_segments(payload)
    sos_keyword = "share the location to select contact"
    police_keyword = "police"
    sos_count = 5
    sos_window = 10.0  # seconds
    sos_times = []
    police_times = []
    for text, start, end in segments:
        # Lowercase for matching
        t = text.lower()
        # Find all occurrences in this segment
        idx = 0
        while True:
            idx = t.find(sos_keyword, idx)
            if idx == -1:
                break
            sos_times.append(start)
            idx += len(sos_keyword)
        idx = 0
        while True:
            idx = t.find(police_keyword, idx)
            if idx == -1:
                break
            police_times.append(start)
            idx += len(police_keyword)

    sos_alert = False
    police_alert = False
    # Check for N sos keywords in T seconds
    sos_times.sort()
    for i in range(len(sos_times) - sos_count + 1):
        if sos_times[i + sos_count - 1] - sos_times[i] <= sos_window:
            sos_alert = True
            break
    # Check for N police keywords in T seconds (same logic, can adjust if needed)
    police_times.sort()
    for i in range(len(police_times) - sos_count + 1):
        if police_times[i + sos_count - 1] - police_times[i] <= sos_window:
            police_alert = True
            break

    response = {
        "user_id": user_id,
        "formatted_transcript": formatted,
        "sos_alert": sos_alert,
        "police_alert": police_alert,
    }
    return response

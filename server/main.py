import os
import re
import logging
from fastapi import FastAPI, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import google.generativeai as genai
import asyncio

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in .env — see .env.example")

genai.configure(api_key=GEMINI_API_KEY)

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

    logger.info("Processing transcript for user=%s (%d chars)", user_id, len(formatted))

    try:
        result = await analyze_transcript(formatted)
        return {
            "user_id": user_id,
            "formatted_transcript": formatted,
            **result,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )

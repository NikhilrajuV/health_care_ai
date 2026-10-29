"""
Document-grounded HealthCare AI chatbot response generation.
Expected passage format:
{"text": "...", "source": "file.pdf", "page": 1, "relevance_score": 0.72}
"""
import os
from dotenv import load_dotenv
from groq import Groq

from src.safety import (
    FALLBACK_ANSWER,
    has_emergency_language,
    prepare_context_or_fallback,
)

load_dotenv(override=True)

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()


def _make_client():
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing. Check your .env file.")
    return Groq(api_key=api_key)


def generate_response(
    question,
    history_text="",
    medical_context=None,
    image_context=None,
    image_filename=None,
):
    """
    Generate an answer using only evidence in the supplied medical passages.
    When the question is unsupported, return a conservative fallback.
    """
    question = (question or "").strip()
    passages = medical_context or []

    # Image follow-ups may use the supplied image description, but must not
    # present it as diagnosis or as evidence from a medical PDF.
    if not passages and not image_context:
        if has_emergency_language(question):
            return (
                "Your message may describe an emergency. Please seek urgent "
                "medical help now; in India, call 112 when appropriate. "
                "I could not verify this from the uploaded medical documents."
            )
        return FALLBACK_ANSWER

    usable_passages, safety_message = prepare_context_or_fallback(question, passages)
    if safety_message and not image_context:
        return safety_message

    medical_text = "\n\n".join(
        f"Source: {item.get('source', 'Unknown PDF')}, page {item.get('page', '?')}\n"
        f"Passage: {item.get('text', '')}"
        for item in usable_passages
    ) or "No usable medical document passages are available."

    image_text = (
        f"Image filename: {image_filename or 'Unknown'}\n"
        f"Image description (not a diagnosis): {image_context}"
        if image_context else "No image context is available."
    )

    system_message = """
You are HealthCare AI, an educational healthcare assistant.

STRICT DOCUMENT-GROUNDING:
- For questions about medical facts, answer ONLY from the provided usable PDF passages.
- Do not use outside knowledge to fill gaps.
- If the passages do not directly support an answer, say:
  "I could not find relevant information in the uploaded medical documents."
- Do not infer a fact merely because it sounds medically plausible.
- Cite each factual claim with the exact filename and page from the supplied passages.
- Never invent filenames, page numbers, quotes, tests, ranges, diagnoses, or treatments.
- If only part of a question is supported, answer only that part and state what is missing.
- Do not provide personalized diagnosis, treatment, or medication dosing.
- If possible emergency symptoms are described, recommend urgent professional help.
- The image description is not proof of a medical condition and must not be treated as a diagnosis.
- Use concise Markdown bullets/headings. Do not use HTML.
"""

    user_message = f"""
Conversation history (for context only; not a source of medical facts):
{history_text or "None"}

Current image description:
{image_text}

Medical passages that may be used as evidence:
{medical_text}

Current question:
{question}

Answer using only directly supporting passages. Cite each claim as
[filename, page N]. If there is not enough evidence, use the exact fallback
message rather than supplying general medical knowledge.
"""

    client = _make_client()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        temperature=0,
        max_completion_tokens=1000,
    )
    answer = response.choices[0].message.content
    if not answer or not answer.strip():
        raise RuntimeError("Groq returned an empty response.")
    return answer.strip()

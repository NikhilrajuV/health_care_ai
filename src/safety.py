"""
Safety and grounding helpers for the HealthCare AI prototype.
These checks reduce unsupported answers; they are not a clinical safety guarantee.
"""
import re

FALLBACK_ANSWER = (
    "I could not find relevant information in the uploaded medical documents."
)

# Disease/topic names that should be explicitly represented in retrieved passages.
TOPIC_ALIASES = {
    "diabetes": ("diabetes", "blood glucose", "blood sugar", "insulin"),
    "malaria": ("malaria", "plasmodium", "anopheles"),
    "hypertension": ("hypertension", "high blood pressure"),
    "asthma": ("asthma", "wheezing", "bronchial"),
    "heart attack": ("heart attack", "myocardial infarction", "cardiac arrest"),
}


def normalize_text(text: str) -> str:
    """Normalize text for conservative lexical checks."""
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def is_medication_dose_question(question: str) -> bool:
    """Identify requests for specific medicine doses or dosing schedules."""
    q = normalize_text(question)
    dose_terms = (
        "dose", "dosage", "how many mg", "how much should i take",
        "how many tablets", "how often should i take", "medication dose",
        "medicine dose", "take how many", "what dose",
    )
    medicine_terms = (
        "medicine", "medication", "tablet", "pill", "drug", "insulin",
        "metformin", "antibiotic", "mg", "milligram",
    )
    return any(term in q for term in dose_terms) and (
        any(term in q for term in medicine_terms) or "dose" in q or "dosage" in q
    )


def has_emergency_language(question: str) -> bool:
    """Simple keyword screen for possible emergencies; not a diagnostic tool."""
    q = normalize_text(question)
    terms = (
        "can't breathe", "cannot breathe", "difficulty breathing",
        "chest pain", "unconscious", "not waking", "seizure",
        "severe bleeding", "overdose", "suicidal",
    )
    return any(term in q for term in terms)


def topic_matches_passages(question: str, passages: list[dict]) -> bool:
    """
    Conservative topic check. If a known disease is named in the question,
    require at least one alias to occur in the retrieved passages.
    For other questions, require meaningful word overlap.
    """
    q = normalize_text(question)
    passage_text = normalize_text(" ".join(
        str(item.get("text", "")) for item in (passages or [])
    ))
    if not q or not passage_text:
        return False

    for topic, aliases in TOPIC_ALIASES.items():
        if topic in q or any(alias in q for alias in aliases):
            return any(alias in passage_text for alias in aliases)

    # Generic query: ignore common words and check whether substantive terms
    # appear in the retrieved text. This is intentionally conservative.
    stopwords = {
        "what", "which", "when", "where", "why", "how", "does", "do", "is",
        "are", "the", "a", "an", "of", "for", "to", "in", "on", "and", "or",
        "with", "my", "me", "i", "can", "could", "please", "explain", "tell",
        "about", "uploaded", "documents", "document", "pdf", "information",
        "common", "symptoms", "difference", "between", "normal", "range",
    }
    words = {
        w for w in re.findall(r"[a-z]{3,}", q)
        if w not in stopwords
    }
    if not words:
        return False
    matched = sum(1 for word in words if word in passage_text)
    return matched >= 1


def prepare_context_or_fallback(question: str, passages: list[dict]) -> tuple[list[dict], str | None]:
    """
    Return usable passages, or a safe fallback reason when there is no evidence.
    """
    if is_medication_dose_question(question):
        # Do not attempt personalized dosing, even if a PDF mentions medicines.
        return [], (
            "I can’t recommend a personal medication dose. The right dose depends "
            "on the medicine and individual clinical factors. Please follow your "
            "prescriber’s instructions or ask a qualified healthcare professional."
        )

    if not passages or not topic_matches_passages(question, passages):
        return [], FALLBACK_ANSWER

    return passages, None

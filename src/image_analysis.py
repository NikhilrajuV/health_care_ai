import os
from dotenv import load_dotenv
from groq import Groq
import base64

load_dotenv(override=True)

VISION_MODEL = os.getenv(
    "GROQ_VISION_MODEL",
    "qwen/qwen3.8-27b",
).strip()


def analyze_image(image_bytes: bytes, mime_type: str) -> str:
    """Generate a general description of an uploaded image."""

    api_key = os.getenv("GROQ_API_KEY", "").strip()

    if not api_key:
        raise ValueError("GROQ_API_KEY is missing in .env.")

    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/webp",
    }

    if mime_type not in allowed_types:
        raise ValueError(
            "Please upload a JPG, JPEG, PNG, or WEBP image."
        )

    if not image_bytes:
        raise ValueError("The uploaded image is empty.")

    if len(image_bytes) > 10 * 1024 * 1024:
        raise ValueError("Image must be smaller than 10 MB.")

    encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{mime_type};base64,{encoded_image}"

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are HealthCare AI's cautious image-understanding component. "
                    "Describe only observable features and readable text; when the "
                    "image is medical-related, state what is visibly present, what "
                    "cannot reliably be determined from the image, and which details "
                    "need clinician interpretation. You may mention broad possibilities "
                    "as possibilities only, never diagnose or confirm a disease from an "
                    "image. Do not invent measurements, symptoms, test results, or "
                    "medical history. Do not identify people or infer sensitive personal "
                    "attributes. If image quality or context is inadequate, say so. "
                    "For urgent visible warning signs or user-described emergency symptoms, "
                    "advise urgent professional care without delaying it for more analysis."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Describe the image and any visible medical-related features carefully. "
                            "Separate direct observations from possibilities, mention uncertainty, "
                            "and do not claim a confirmed diagnosis. If this is a medical image, "
                            "recommend interpretation by a qualified healthcare professional."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url,
                        },
                    },
                ],
            },
        ],
        temperature=0.2,
        max_completion_tokens=1024,
    )

    result = response.choices[0].message.content

    if not result:
        raise RuntimeError("The vision model returned an empty response.")

    return result.strip()

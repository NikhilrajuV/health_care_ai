
import os
import base64
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv(override=True)

api_key = os.getenv("GROQ_API_KEY", "").strip()
model = os.getenv("GROQ_VISION_MODEL", "").strip()

if not api_key:
    raise SystemExit("GROQ_API_KEY missing in .env")

if not model:
    raise SystemExit("GROQ_VISION_MODEL missing in .env")

# Put a test image named test.jpg in the project root.
image_path = Path("test.jpg")

if not image_path.exists():
    raise SystemExit(
        "Place a JPG image named test.jpg in the project root "
        "and run this script again."
    )

image_bytes = image_path.read_bytes()
encoded = base64.b64encode(image_bytes).decode("utf-8")
data_url = f"data:image/jpeg;base64,{encoded}"

client = Groq(api_key=api_key)

try:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe the visible objects in this image.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                ],
            }
        ],
        max_completion_tokens=300,
    )

    print("\nVISION TEST SUCCESS")
    print(response.choices[0].message.content)

except Exception as exc:
    print("\nVISION TEST FAILED")
    print(type(exc).__name__)
    print(str(exc))

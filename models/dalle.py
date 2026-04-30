from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def get_client(api_key: Optional[str] = None) -> OpenAI:
    resolved_api_key = api_key or OPENAI_API_KEY
    if not resolved_api_key:
        raise ValueError("OPENAI_API_KEY is missing from the environment.")
    return OpenAI(api_key=resolved_api_key)


def generate_image(
    prompt: str,
    client: Optional[OpenAI] = None,
    size: str = "1024x1024",
) -> str:
    if not prompt or not prompt.strip():
        raise ValueError("Prompt must not be empty.")

    active_client = client or get_client()

    try:
        response = active_client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=size,
            n=1,
        )
    except RateLimitError as exc:
        raise RuntimeError("OpenAI rate limit reached. Please retry later.") from exc
    except APITimeoutError as exc:
        raise RuntimeError("OpenAI request timed out while generating the image.") from exc
    except APIConnectionError as exc:
        raise RuntimeError("Unable to connect to the OpenAI API.") from exc
    except APIStatusError as exc:
        raise RuntimeError(
            f"OpenAI API returned an error status ({exc.status_code})."
        ) from exc
    except Exception as exc:
        raise RuntimeError("Unexpected error while generating an image with DALL-E 3.") from exc

    if not response.data or not response.data[0].url:
        raise RuntimeError("OpenAI did not return an image URL.")

    return response.data[0].url

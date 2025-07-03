# -*- coding: utf-8 -*-


from typing import Literal

from cogkit.api.models.request import RequestParams


class ImageGenerationParams(RequestParams):
    prompt: str
    model: str = "cogview-4"
    n: int = 1
    size: Literal[
        "1024x1024",
        "768x1024",
        "768x1344",
        "864x1152",
        "1344x768",
        "1152x864",
        "1440x720",
        "720x1440",
    ] = "1024x1024"
    num_inference_steps: int = 50
    guidance_scale: float = 3.5
    lora_path: str | None = None
    lora_scale: float = 1.0
    user: str | None = None
    # ! unsupported parameters
    # quality: Literal["standard", "hd"] = "standard"
    # response_format: Literal["url", "b64_json"] = "url"
    # style: Literal["vivid", "natural"] = "vivid"

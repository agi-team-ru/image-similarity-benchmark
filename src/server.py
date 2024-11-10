import io
import os
from typing import List
from fastapi import FastAPI
import uvicorn
import random
import logging
from dto import ImgSimilarityRequest, ImgSimilarityResponse
from utils import base64_decode
from server_params import (
    LocalServerMethod,
    LocalServerOptions,
    LOCAl_SERVER_ENDPOINT,
    LOCAl_SERVER_PORT,
)
from PIL import Image

from similarities import ClipSimilarity


logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper())

logger = logging.getLogger(__name__)


app = FastAPI()

model = ClipSimilarity(model_name_or_path="OFA-Sys/chinese-clip-vit-base-patch16")


@app.get("/")
async def version():
    return {"version": "1.0.0"}


@app.post(LOCAl_SERVER_ENDPOINT)
async def img_similarity(request: ImgSimilarityRequest) -> ImgSimilarityResponse:
    options = LocalServerOptions.model_validate(request.options)
    score: float
    if options.method == LocalServerMethod.ALWAYS_FALSE:
        score = 0.0
    elif options.method == LocalServerMethod.ALWAYS_TRUE:
        score = 1.0
    elif options.method == LocalServerMethod.ALWAYS_RANDOM:
        score = random.random()
    elif options.method == LocalServerMethod.LIB_SIMILARITIES:
        score = process_lib_similarities(decode_images(request))
    elif options.method == LocalServerMethod.THREAT_EXCHANGE:
        score = process_threat_exchange(decode_images(request))
    else:
        raise Exception(f"Unknown method {options.method}")

    return ImgSimilarityResponse(score=score)


def process_threat_exchange(images: List[bytes]):
    pil_images = [Image.open(io.BytesIO(img_bytes)) for img_bytes in images]
    res = model.similarity(pil_images[0], pil_images[1])
    return float(res)

def process_lib_similarities(images: List[bytes]):
    pil_images = [Image.open(io.BytesIO(img_bytes)) for img_bytes in images]
    res = model.similarity(pil_images[0], pil_images[1])
    return float(res)


def decode_images(request: ImgSimilarityRequest):
    return [base64_decode(img) for img in request.images]


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        port=LOCAl_SERVER_PORT,
        reload=False,
        log_level="debug",
    )

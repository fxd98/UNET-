# Copyright (C) 2022, François-Guillaume Fernandez.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import torch
from app.schemas import ClsCandidate
from app.vision import classification_model, classification_preprocessor, decode_image
from fastapi import APIRouter, File, UploadFile, status

router = APIRouter()


@router.post("/", response_model=ClsCandidate, status_code=status.HTTP_200_OK, summary="Perform image classification")
async def classify(file: UploadFile = File(...)):
    """Runs holocron vision model to analyze the input image"""
    img_tensor = classification_preprocessor(decode_image(file.file.read()))
    out = classification_model(img_tensor.unsqueeze(0)).squeeze(0)
    probs = torch.softmax(out, dim=0)

    return ClsCandidate(
        value=classification_model.default_cfg['classes'][probs.argmax().item()],
        confidence=probs.max().item(),
    )

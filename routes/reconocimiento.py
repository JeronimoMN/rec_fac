import cv2
from fastapi import APIRouter, File, UploadFile
import os
import uuid
from services.face_recognition import predict_identity

router = APIRouter()

IMAGEDIR = "images/"


@router.post("/upload/")
async def upload_file(file: UploadFile ):

    """
    Subir un archivo y procesar reconocimiento facial.
    """

    print(file.content_type)
    file.filename = f"{uuid.uuid4()}.jpg"
    contents = await file.read()

    file_path = os.path.join(IMAGEDIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(contents)

    recognition_result = predict_identity(file_path)

    return recognition_result
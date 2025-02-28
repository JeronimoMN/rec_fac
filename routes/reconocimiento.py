from fastapi import APIRouter, File, UploadFile, Form
import os
import uuid
from services.face_recognition import predict_identity
import base64

router = APIRouter()

IMAGEDIR = "images/"

@router.post("/upload/")
async def upload_file(register_image: str = File(...), type: str = Form(), access_time: str = Form(), status: str = Form()):


    """
    Subir un archivo y procesar reconocimiento facial.
    """
    image_data = base64.b64decode(register_image)

    # Crear un nombre de archivo único (puedes mejorarlo según tu lógica)
    file_name = f"user_default_{access_time.replace(':', '-')}.jpg"
    file_path = os.path.join(IMAGEDIR, file_name)

    # Guardar la imagen como archivo JPG
    with open(file_path, "wb") as f:
        f.write(image_data)

    recognition_result = predict_identity(file_path)
    print(status)
    print(f"Time: {access_time}")
    print(f"Type Access: {type}")

    return recognition_result
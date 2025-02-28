from base64 import b64decode

import httpx
from fastapi import APIRouter, File, UploadFile, Form
import uuid
import os
import base64

router = APIRouter()

IMAGEDIR = "images/"


@router.post('/rfid')
async def rfid(register_image: str = File(...), type: str = Form(), access_time: str = Form(), status: str = Form(), id_user: str = Form()):
    ##contents = await file.read()

    image_data = base64.b64decode(register_image)

    # Crear un nombre de archivo único (puedes mejorarlo según tu lógica)
    file_name = f"user_{id_user}_{access_time.replace(':', '-')}.jpg"
    file_path = os.path.join(IMAGEDIR, file_name)

    # Guardar la imagen como archivo JPG
    with open(file_path, "wb") as f:
        f.write(image_data)

    print(type)
    print(access_time)
    print(id_user)
    print(status)

    return True

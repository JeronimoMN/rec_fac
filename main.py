import uvicorn
import os
from fastapi import FastAPI, UploadFile
from routes import reconocimiento, model_training


app = FastAPI()
#app.run(host="0.0.0.0", port=8000)

# Registrar los routes
app.include_router(reconocimiento.router, prefix="/recognition", tags=["Recognition"])
app.include_router(model_training.router, prefix="/model_training", tags=["Recognition"])

IMAGEDIR = "images/"


@app.get("/")
async def root():
    """
    Endpoint raíz de la aplicación.
    """
    return {"message": "¡Bienvenido a la API de Reconocimiento Facial!"}

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):

    file_path = os.path.join(IMAGEDIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    return {"filename": file.filename}


if __name__ == "__main__":
    config = uvicorn.Config("main:app", port=8001, log_level="info")
    server = uvicorn.Server(config)
    server.run()
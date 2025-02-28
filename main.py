import uvicorn
import os
from fastapi import FastAPI, UploadFile, Form
from routes import reconocimiento, model_training, rfid
from services.face_recognition import predict_identity

app = FastAPI()
#app.run(host="0.0.0.0", port=8000)

# Registrar los routes
app.include_router(reconocimiento.router, prefix="/recognition", tags=["Recognition"])
app.include_router(model_training.router, prefix="/model_training", tags=["Training"])
app.include_router(rfid.router, prefix="/rfid", tags=["RFID"])

IMAGEDIR = "images/"

if __name__ == "__main__":
    config = uvicorn.Config("main:app", port=8001, log_level="info")
    server = uvicorn.Server(config)
    server.run()
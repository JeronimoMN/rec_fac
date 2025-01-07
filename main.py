from fastapi import FastAPI
from routes import reconocimiento, model_training
app = FastAPI()

# Registrar los routes
app.include_router(reconocimiento.router, prefix="/recognition", tags=["Recognition"])
app.include_router(model_training.router, prefix="/model_training", tags=["Recognition"])


@app.get("/")
async def root():
    """
    Endpoint raíz de la aplicación.
    """
    return {"message": "¡Bienvenido a la API de Reconocimiento Facial!"}

import cv2
import numpy as np
import pickle
from keras_facenet import FaceNet
from mtcnn.mtcnn import MTCNN
from sklearn.preprocessing import LabelEncoder
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import os
from random import randint
import uuid

IMAGEDIR= "images/"
app = FastAPI()

# Cargar el modelo SVM
with open('files/svm_model_160x160.pkl', 'rb') as f:
    model = pickle.load(f)

# Cargar los embeddings y las etiquetas
data = np.load('files/faces_embeddings_done_4classes.npz')
EMBEDDED_X = data['arr_0']
Y = data['arr_1']

# Inicializar el detector de rostros y el embebedor
detector = MTCNN()
embedder = FaceNet()

# Inicializar el codificador
encoder = LabelEncoder()
encoder.fit(Y)  # Ajustar el codificador a las etiquetas

# Definir un umbral
UMBRAL = 1

def es_coincidencia(embedding1, embedding2, umbral):
    # Calcular la distancia euclidiana entre los dos embeddings
    distancia = np.linalg.norm(embedding1 - embedding2)
    resultado = distancia < umbral
    return resultado  # Retorna True si la distancia es menor que el umbral

# Función para obtener el embedding de una imagen
def get_embedding(face_img):
    face_img = face_img.astype('float32')  # 3D(160x160x3)
    face_img = np.expand_dims(face_img, axis=0)  # 4D (None, 160, 160, 3)
    yhat = embedder.embeddings(face_img)
    return yhat[0]  # 512D image (1x1x512)

# Función para predecir la identidad de una imagen
def predict_identity(image_path):
    # Leer la imagen
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detectar rostros
    results = detector.detect_faces(img)

    if results:
        x, y, w, h = results[0]['box']
        face = img[y:y+h, x:x+w]
        face = cv2.resize(face, (160, 160))

        # Obtener el embedding
        test_im = get_embedding(face)
        test_im = np.array(test_im).reshape(1, -1)  # Reshape para la predicción
        #reshape(1, -1). 1 -> El arreglo solo tendra una fila; -1 ->

        # Predecir la identidad
        ypreds = model.predict(test_im)

        # Evaluar el umbral
        if UMBRAL == 1:
            coincidencias = []
            for embedding in EMBEDDED_X:
                if es_coincidencia(test_im, embedding, UMBRAL):
                    coincidencias.append(embedding)

            # Devolver resultados
            if coincidencias:
                return encoder.inverse_transform(ypreds)
            else:
                return None
        else:
            return "Error: El umbral debe ser igual a 1 para continuar."
    else:
        return "No se detectó ningún rostro."

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    # Generar un nombre de archivo único
    file.filename = f"{uuid.uuid4()}.jpg"
    contents = await file.read()

    # Guardar la imagen
    file_path = os.path.join(IMAGEDIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(contents)

    # Realizar reconocimiento facial
    recognition_result = predict_identity(file_path)

    # Formatear el resultado para la respuesta JSON
    if recognition_result is None:
        recognition_result = "No matching face detected."
    elif isinstance(recognition_result, np.ndarray):
        recognition_result = recognition_result.tolist()  # Convertir a lista

    return {
        recognition_result[0]  # Convertir a cadena
    }


@app.get("/show/")
async def readRandomfile():
    # get random file from the image directory
    files = os.listdir(IMAGEDIR)
    random_index = randint(0, len(files) - 1)

    path = f"{IMAGEDIR}{files[random_index]}"

    return FileResponse(path)

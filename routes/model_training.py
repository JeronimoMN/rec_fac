from fastapi import APIRouter, HTTPException
import os
import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

router = APIRouter()

# Inicializar detector y embebedor
detector = MTCNN()
embedder = FaceNet()

class FACELOADING:
    def __init__(self, directory):
        self.directory = directory
        self.target_size = (160, 160)
        self.X = []  # Almacenamiento de imágenes de los rostros
        self.Y = []  # Almacenamiento de etiquetas
        self.detector = MTCNN()  # Inicializar el detector aquí

    def extract_face(self, filename):
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.detector.detect_faces(img)
        if results:  # Verifica si se detectaron caras
            x, y, w, h = results[0]['box']
            x, y = abs(x), abs(y)
            face = img[y:y+h, x:x+w]
            face_arr = cv2.resize(face, self.target_size)
            return face_arr
        else:
            raise ValueError(f"No se detectó ningún rostro en {filename}")

    def load_faces(self, dir):
        FACES = []
        for im_name in os.listdir(dir):
            try:
                path = os.path.join(dir, im_name)
                single_face = self.extract_face(path)
                FACES.append(single_face)
            except Exception as e:
                print(f"Error al cargar {im_name}: {e}")
        return FACES

    def load_classes(self):
        for sub_dir in os.listdir(self.directory):
            path = os.path.join(self.directory, sub_dir)
            if os.path.isdir(path):  # Verifica si es un subdirectorio
                FACES = self.load_faces(path)
                labels = [sub_dir for _ in range(len(FACES))]
                print(f"Loaded successfully: {len(labels)} rostros de {sub_dir}")
                self.X.extend(FACES)
                self.Y.extend(labels)
        return np.asarray(self.X), np.asarray(self.Y)

@router.post("/train/")
async def create_model(dataset_dir: str = "files/dataset"):
    """
    Entrenar un nuevo modelo SVM basado en el dataset proporcionado.
    """
    # Validar directorio del dataset
    if not os.path.exists(dataset_dir):
        raise HTTPException(status_code=400, detail="El directorio del dataset no existe.")

    faceloading = FACELOADING(dataset_dir)
    X, Y = faceloading.load_classes()

    if len(X) == 0 or len(Y) == 0:
        raise HTTPException(status_code=400, detail="No se encontraron imágenes válidas en el dataset.")

    # Convertir las imágenes al formato esperado por el modelo
    EMBEDDED_X = []
    for face in X:
        face = face.astype("float32")  # Asegurar tipo float32
        face = np.expand_dims(face, axis=0)  # Expandir dimensiones a (1, 160, 160, 3)
        embedding = embedder.embeddings(face)[0]  # Obtener el embedding (512 dimensiones)
        EMBEDDED_X.append(embedding)

    EMBEDDED_X = np.array(EMBEDDED_X)

    # Guardar embeddings y etiquetas
    embeddings_path = "files/faces_embeddings.npz"
    np.savez_compressed(embeddings_path, EMBEDDED_X, Y)

    # Codificar etiquetas
    encoder = LabelEncoder()
    Y_encoded = encoder.fit_transform(Y)

    # Dividir datos
    X_train, X_test, Y_train, Y_test = train_test_split(EMBEDDED_X, Y_encoded, test_size=0.2, random_state=42)

    # Entrenar modelo SVM
    model = SVC(kernel="linear", probability=True)
    model.fit(X_train, Y_train)

    # Guardar el modelo
    model_path = "files/svm_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    return {
        "message": "Modelo entrenado y guardado con éxito.",
        "model_path": model_path,
        "embeddings_path": embeddings_path,
    }

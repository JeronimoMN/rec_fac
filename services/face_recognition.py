import cv2
import numpy as np
from keras_facenet import FaceNet
from mtcnn.mtcnn import MTCNN
import pickle
from sklearn.preprocessing import LabelEncoder

# Cargar el modelo SVM
with open("files/svm_model_pragmatic.pkl", "rb") as f:
    model = pickle.load(f)

# Cargar los embeddings y las etiquetas
data = np.load("files/faces_embeddings_pragmatic.npz")
EMBEDDED_X = data["arr_0"]
Y = data["arr_1"]

print("Etiquetas")
print(Y)


# Inicializar el detector de rostros y el embebedor
detector = MTCNN()
embedder = FaceNet()

# Inicializar el codificador
encoder = LabelEncoder()
encoder.fit(Y)

umbrall = 1

def get_embedding(face_img):
    """
    Obtener el embedding de una imagen facial.
    """
    face_img = face_img.astype("float32")
    face_img = np.expand_dims(face_img, axis=0)
    yhat = embedder.embeddings(face_img)
    return yhat[0]

def es_coincidencia(embedding1, embedding2, umbral):
    """
    Comparar dos embeddings para determinar coincidencia.
    """
    distancia = np.linalg.norm(embedding1 - embedding2)
    return distancia < umbral

def predict_identity(image_path):
    """
    Predecir la identidad de una imagen dada.
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detectar rostros
    results = detector.detect_faces(img)

    if results:
        x, y, w, h = results[0]["box"]
        face = img[y:y+h, x:x+w]
        face = cv2.resize(face, (160, 160))

        test_im = get_embedding(face)
        test_im = np.array(test_im).reshape(1, -1) #Convierte el embedding a un array de 2 dimensiones.

        ypreds = model.predict(test_im)
        identity = encoder.inverse_transform(ypreds)

        if umbrall == 1:
            coincidencias = [
                embedding
                for embedding in EMBEDDED_X
                if es_coincidencia(test_im, embedding, umbrall)
            ]

            if coincidencias:
                return True
            else:
                return False
        else:
            return "Error: El umbral debe ser igual a 1 para continuar."
    else:
        return "No se detectó ningún rostro."

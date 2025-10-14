# backend.py
from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import pickle
import uvicorn
import io
from PIL import Image

app = FastAPI()

# Charger modèle et label encoder une fois
model = load_model("model2_best.h5")
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return preprocess_input(image)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = preprocess_image(image_bytes)
    preds = model.predict(image)
    label_idx = np.argmax(preds, axis=1)[0]
    label = le[label_idx].replace("_", " ").upper()
    return {"prediction": label}
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

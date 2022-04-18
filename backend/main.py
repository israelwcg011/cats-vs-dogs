from fastapi import FastAPI
import numpy as np
from keras.models import load_model
from os.path import join
from config import BACKEND_FOLDER
from helpers import prepare_image

##### model #####

# load model for the predictions
model = load_model(join(BACKEND_FOLDER, "models", "cats_vs_dogs_model_2.h5"))

# create function to classify images
def identify(image_path):
    animal = {0: "Cat", 1: "Dog"}
    image = prepare_image(image_path)
    prediction = model.predict([image])[0]
    index = np.where(prediction == np.max(prediction))[0][0]
    return animal[index]

##### server initialization #####

app = FastAPI()

@app.get("/")
async def root():
    return {
        "message": "Hello World!"
    }

@app.get("/prediction")
async def make_prediction():
    result = identify(join(BACKEND_FOLDER, "uploads", "61.jpg"))
    return {
        "message": f"I think it is a {result}!"
    }
import numpy as np
from tensorflow.keras.models import model_from_json
import cv2


class Prediction:
    def __init__(self):
        print("running constructor ")

        json_file = open('mask_detector/models/model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)

        # load weights into new model
        self.model.load_weights("mask_detector/models/model.h5")
        print("Loaded model from disk")

    def preprocessing(self, image):
        """pre-processing of the input image as required"""
        image = cv2.resize(image, (128, 128))
        image = image / 255.0
        return image

    def predict(self, image):
        image = self.preprocessing(image)

        predictions = self.model.predict(np.expand_dims(image, axis=0))
        if np.argmax(predictions[0]) == 0:
            return {"result": "mask", "score": str(predictions[0][np.argmax(predictions[0])]), "maskDetected": True}
        else:
            return {"result": "unmask", "score": str(predictions[0][np.argmax(predictions[0])]) , "maskDetected": False}

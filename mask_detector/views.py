
from django.http import JsonResponse
import cv2
import json
import base64
import numpy as np

# Create your views here.
from mask_detector.modules.FaceDetectorMTCNN import FaceDetector
from mask_detector.modules.Prediction import Prediction

from rest_framework.decorators import api_view
FD = FaceDetector()
Inference = Prediction()

@api_view(["POST"])
def home(request):
    try:
        request_body = json.loads(request.body.decode("utf-8"))

        image_blob = request_body['blob']
        decoded_data = base64.b64decode(image_blob)

        image = cv2.imdecode(np.frombuffer(decoded_data, np.uint8), 1)
        face_data = FD.detect_face(image)

        if face_data["error"]:
            return JsonResponse({"error": True , "errorMessage": "NO FACE DETECTED"})
        else:
            result = Inference.predict(face_data["data"])
            return JsonResponse(result)
    except Exception as e:
        return JsonResponse({"error": True, "errorMessage": "BACKEND FAILURE PLEASE TRY AGAIN LATER "})
    finally:
        pass

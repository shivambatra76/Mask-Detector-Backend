from mtcnn.mtcnn import MTCNN
import cv2

class FaceDetector:
    def __init__(self):
        print("Loading face detector")
        self.detector = MTCNN()
        print("Loaded face detector")

    def detect_face(self, image):
        """ function to extract face from an image """
        try:
            rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            faces = self.detector.detect_faces(rgb_frame)

            x1, y1, width, height = faces[0]['box']
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height

            # extract the face
            face = rgb_frame[y1:y2, x1:x2]
            return {"error": False , "data": face}
        except IndexError as index:

            return {"error": True}
        finally:
            pass
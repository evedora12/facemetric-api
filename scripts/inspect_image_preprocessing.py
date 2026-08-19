from pathlib import Path

from io import BytesIO
from PIL import Image
import cv2 as cv

from app.services.image_preprocessing import preprocess_image
from app.services.landmark_detection import canny_edge

img_path = r"/Users/evedurant/Desktop/facemetric-api/tests/test_image.png"
img_path = r"/Users/evedurant/Desktop/facemetric-api/tests/face.jpg"

with open(img_path, "rb") as image_file:
    image_bytes = image_file.read()

img = preprocess_image(image_bytes)    
canny = canny_edge(img, 50, 200)
print(canny.shape)

# display image
cv.imshow("image view", canny)
cv.waitKey(0)
cv.destroyAllWindows()
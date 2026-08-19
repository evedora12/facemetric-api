from  dataclasses import dataclass
from io import BytesIO
import numpy as np
from PIL import Image
import cv2 as cv

from app.services.image_validation import ValidatedImage

@dataclass
class PreprocessedImage:
    pass

def preprocess_image(validated_img: ValidatedImage): 
    """ 
    Takes in an image in the form of byte informatio and converts it to 
    3 channel colour matrix
    """
    image_bytes = validated_img.image_bytes

    # convert raw bytes into a 1d array
    np_array = np.frombuffer(image_bytes, np.uint8)


    #converts array of bytes into a 3 channel colour matrix
    img = cv.imdecode(np_array, cv.IMREAD_COLOR)

    return img

    
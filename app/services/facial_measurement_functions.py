import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
# from mediapipe.tasks.python.vision import drawing_utils
#from mediapipe.tasks.python.vision import drawing_styles
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv



mp_drawing = mp.solutions.drawing_utils

from pathlib import Path

project_root = Path(__file__).resolve().parents[2]

model_path = project_root/"models"/"face_landmarker.task"


# Create a FaceLandmarker object

# Code adapted from google Mediapipe Face Landmarker example:

# this dewscribes where the trained model is stored
base_options = python.BaseOptions(model_asset_path=str(model_path))

# this configures how the face landmarker should behave
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
# detector uses options from above
detector = vision.FaceLandmarker.create_from_options(options)


def get_detection_result_takes_path(img_path:str):
    """
    Takes in an image path and runs mediapipe on image.
    
    returns detection_result
        
    """
    # make image
    bgr_image = cv.imread(img_path)
        
    if bgr_image is None:
        raise FileNotFoundError( f"opencv cant read image")
        
    rgb_image = cv.cvtColor(
        bgr_image,
        cv.COLOR_BGR2RGB,
    )
        
    rgb_image = np.ascontiguousarray(
        rgb_image,
        dtype=np.uint8,
    )
        
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_image, )
        
    # detect landmarks 
    detection_result = detector.detect(mp_image)
    return detection_result

def get_detection_result_takes_path(img_path:str):
    """
    Takes in an image path and runs mediapipe on image.
    
    returns detection_result
        
    """
    # make image
    bgr_image = cv.imread(img_path)
        
    if bgr_image is None:
        raise FileNotFoundError( f"opencv cant read image")
        
    rgb_image = cv.cvtColor(
        bgr_image,
        cv.COLOR_BGR2RGB,
    )
        
    rgb_image = np.ascontiguousarray(
        rgb_image,
        dtype=np.uint8,
    )
        
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_image, )
        
    # detect landmarks 
    detection_result = detector.detect(mp_image)
    return detection_result

def get_detection_result_takes_img(bgr_image):
    """
    Takes in an image path and runs mediapipe on image.
    
    returns detection_result
        
    """
    
    # preprocessed image will be a bgr image
        
    if bgr_image is None:
        raise FileNotFoundError( f"opencv cant read image")
        
    rgb_image = cv.cvtColor(
        bgr_image,
        cv.COLOR_BGR2RGB,
    )
        
    rgb_image = np.ascontiguousarray(
        rgb_image,
        dtype=np.uint8,
    )
        
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_image, )
        
    # detect landmarks 
    detection_result = detector.detect(mp_image)
    return detection_result


def run_mediapipe_extract_symmetry_points(detection_result, img) -> np.ndarray:
    """
    Takes in an detection result and an image returns an x array and y array
    of points along the centre of the face
    """

    # finds the first (and only in this case) face detected by detection_result
    first_face = detection_result.face_landmarks[0]

    h, w = img.shape[:2]

    # the list of features that are meant to be in the centre of the face and 
    # their index with detection_result
    FEATURES = {"nose_tip": 1, "upper_lip": 13, "upper_face": 10, "chin": 152,}

    
    x = []
    y = []

    for feature_name, landmark_index in FEATURES.items():

        # each separate point in list of face landmarks
        landmark = first_face[landmark_index]

        pixel_x =  int(landmark.x * w)
        pixel_y = int(landmark.y * h)

        x.append(pixel_x)
        y.append(pixel_y)
        
    return x, y


        
def get_angle_of_rotation(img, x, y):
    """ Takes in an image and x and y arrays and 
    returns the angle the face makes iwth the vertical
    and the change in x that the rotation would create
    """
    height, _, _ = img.shape
    # this is the line of best fit that goes through the point in 
    # the x and y array in the form y = ax + b
    a, b = np.polyfit(x, y, 1)

    # now we find points x0, y0 where the line would cross the 
    # top of the image by rearranging y = ax+b to x = (y-b)/a
    # y0 is zero at the top of the image
    y0 = 0
    x0 = int((y0-b)/a)

    # here we find the point where this line would cross the bottom
    #  of the image, i.e the height for the y value
    y1 = height
    x1 = int((y1 - b)/a)

    # find change in x and chnage in y
    delta_x = x1 - x0
    delta_y = y1 - y0

    # find angle that the line of best fit makes with the vertical
    #  using trigonomatry
    theta = np.arctan(delta_x/delta_y)

    return theta, delta_x



def rotate_landmarks(img_path):
    """ Takes in an img_path, detects key landmarks, detects
    the tilt of the face, theta, and rotates the facial landmarks
    by theta. Returns the rotated facial landmarks.
    """
    img = cv.imread(img_path)
    
    img = cv.cvtColor(
        img,
        cv.COLOR_BGR2RGB,
    )

    #rotated_img = rotate(img_path)
    detection_result = get_detection_result_takes_path(img_path)
    face = detection_result.face_landmarks[0]

    h, w, _ = img.shape
    centre = (w/2, h/2)
    x, y = run_mediapipe_extract_symmetry_points(detection_result, img)
    theta, delta_x = get_angle_of_rotation(img, x, y)
    theta = np.rad2deg(theta)
    rotation_matrix = cv.getRotationMatrix2D(centre ,angle=-theta,scale=1.0,)

    delta_x = abs(delta_x)

    for landmark in face:
        x = landmark.x * w
        y = landmark.y * h

        # each point needs to be a 3d matrix since the opencv matrix has 3 columns
        point = np.array([x, y, 1.0])

        rotated_x, rotated_y = np.dot(rotation_matrix, point)

        rotated_x = (rotated_x - delta_x)/w
        rotated_y = (rotated_y - delta_x)/h

        landmark.x = float(rotated_x)
        landmark.y = float(rotated_y)

    return face


def rotate_landmarks_takes_img(img):
    """ Takes in an 3D numpy array, detects key landmarks, detects
    the tilt of the face, theta, and rotates the facial landmarks
    by theta. Returns the rotated facial landmarks.
    """

    
    img = cv.cvtColor(
        img,
        cv.COLOR_BGR2RGB,
    )

    #rotated_img = rotate(img_path)
    detection_result = get_detection_result_takes_img(img)
    face = detection_result.face_landmarks[0]

    h, w, _ = img.shape
    centre = (w/2, h/2)
    x, y = run_mediapipe_extract_symmetry_points(detection_result, img)
    theta, delta_x = get_angle_of_rotation(img, x, y)
    theta = np.rad2deg(theta)
    rotation_matrix = cv.getRotationMatrix2D(centre ,angle=-theta,scale=1.0,)

    delta_x = abs(delta_x)

    for landmark in face:
        x = landmark.x * w
        y = landmark.y * h

        # each point needs to be a 3d matrix since the opencv matrix has 3 columns
        point = np.array([x, y, 1.0])

        rotated_x, rotated_y = np.dot(rotation_matrix, point)

        rotated_x = (rotated_x - delta_x)/w
        rotated_y = (rotated_y - delta_x)/h

        landmark.x = float(rotated_x)
        landmark.y = float(rotated_y)

    return face



def canthal_tilt(face) -> np.ndarray:
    """
    takes in the first face in detection result and determines the canthal tilt of left eye, right eye
    """

    # finds the first (and only in this case) face detected by detection_result
    #face = detection_result.face_landmarks[0]

    

    EYE_INDICES = {
    "left_eye_outer": 33,
    "left_eye_inner": 133,
    "right_eye_inner": 362,
    "right_eye_outer": 263,}

    # get left eye outer point information from its index
    left_eye_outer = face[33]
    left_eye_outer = [left_eye_outer.x, left_eye_outer.y]

    # get left eye inner point information from its index
    left_eye_inner = face[133]
    left_eye_inner = [left_eye_inner.x, left_eye_inner.y]


    # keeping positive tilt positive

    # subtract the outer left x value from the inner one (this will always be positive)
    delta_x_left = left_eye_inner[0] - left_eye_outer[0]

    # subtract outer left y value from the inner one (will be positive if canthal tilt is positive) 
    delta_y_left = left_eye_inner[1] - left_eye_outer[1]


    theta_left = np.arctan(delta_y_left/delta_x_left)
    theta_left = float(np.rad2deg(theta_left))



    right_eye_outer = face[263]
    right_eye_outer = [right_eye_outer.x, right_eye_outer.y]

    right_eye_inner = face[362]
    right_eye_inner = [right_eye_inner.x, right_eye_inner.y]


    delta_x_right = right_eye_outer[0] - right_eye_inner[0]
    delta_y_right = right_eye_inner[1] - right_eye_outer[1]

    theta_right = np.arctan(delta_y_right/delta_x_right)
    theta_right = float(np.rad2deg(theta_right))

    
   

    return theta_left, theta_right




KEY_FEATURE_INDICES = {
    "nose_bottom": 2,

    "upper_lip_top": 0,
    "lower_lip_bottom": 17,
    "mouth_left": 61,
    "mouth_right": 291,
    "upper_lip": 13,

    "right_brow_start": 107,
    "right_brow_arch": 105,
    "right_brow_end": 70,

    "intrabrow_bottom": 8,

    "left_brow_start": 336,
    "left_brow_arch": 334,
    "left_brow_end": 300,

    "right_nostril_outer": 98,
    "left_nostril_outer": 327,

    "chin": 152,
    "left_cheek": 234,
    "right_cheek": 454,

    "left_eye_outer": 33,
    "left_eye_inner": 133,
    "right_eye_inner": 362,
    "right_eye_outer": 263,

    "left_iris_outer": 471,
    "left_iris_inner": 469,

    "right_iris_inner": 476,
    "right_iris_outer": 474,
}




def top_bottom_lip_ratio(face):
    """ Takes in the first face in detection result and returns the ratio
    between the height of the toip and bottom lip
    """
    #face = detection_result.face_landmarks[0]

    # returns the index key associated with the point at top of upper lip
    upper_lip_index = KEY_FEATURE_INDICES["upper_lip_top"]
    upper_lip = face[upper_lip_index]
    upper_lip_y = upper_lip.y

    
    lower_lip_index = KEY_FEATURE_INDICES["lower_lip_bottom"]
    lower_lip = face[lower_lip_index]
    lower_lip_y = lower_lip.y


    middle_index = KEY_FEATURE_INDICES["upper_lip"]
    middle = face[middle_index]
    middle_y = middle.y


    top_lip_height = middle_y - upper_lip_y

    lower_lip_height = lower_lip_y - middle_y


    return lower_lip_height/top_lip_height

def bottom_two_thirds(face):
    """ 
    Takes in the first face in detection result and determines
    the ratio between the lower two "thirds" of the face.

    Returns the middle third divided by the bottom third
    """
    intrabrow_bottom_index = KEY_FEATURE_INDICES["intrabrow_bottom"]
    intrabrow = face[intrabrow_bottom_index]
    intrabrow_y = intrabrow.y
    
        
    nose_bottom_index = KEY_FEATURE_INDICES["nose_bottom"]
    nose_bottom = face[nose_bottom_index]
    nose_bottom_y = nose_bottom.y
    

    
    
    chin_index = KEY_FEATURE_INDICES["chin"]
    chin = face[chin_index]
    chin_y = chin.y

    
    middle_third = nose_bottom_y - intrabrow_y
    
    bottom_third = chin_y - nose_bottom_y
    
    
    return middle_third/bottom_third



def nose_to_chin(face):
    """Takes in the first face in detection result and returns the ratio between the 
    nose to the bottom of the lips and the bottom of the lips to the chin. This should be roughly 1
    """
    nose_bottom_index = KEY_FEATURE_INDICES["nose_bottom"]
    nose_bottom = face[nose_bottom_index]
    nose_bottom_y = nose_bottom.y

    lower_lip_index = KEY_FEATURE_INDICES["lower_lip_bottom"]
    lower_lip = face[lower_lip_index]
    lower_lip_y = lower_lip.y
    
     
        
    chin_index = KEY_FEATURE_INDICES["chin"]
    chin = face[chin_index]
    chin_y = chin.y
    
        
    nose_to_lip_bottom = lower_lip_y - nose_bottom_y
        
    lip_bottom_to_chin = chin_y - lower_lip_y
        
        
    return nose_to_lip_bottom/lip_bottom_to_chin





def arch_symmetry(face):
    """ Takes in the first face in detection result and returns the 
    x distance each brow arch is from the centre
    """

    right_arch_index = KEY_FEATURE_INDICES["right_brow_arch"]
    right_arch = face[right_arch_index]
    right_arch_x = right_arch.x
    print(f"right arch : {right_arch_x}")

    intrabrow_bottom_index = KEY_FEATURE_INDICES["intrabrow_bottom"]
    intrabrow = face[intrabrow_bottom_index]
    intrabrow_x = intrabrow.x
    
    
    left_arch_index = KEY_FEATURE_INDICES["left_brow_arch"]
    left_arch = face[left_arch_index]
    left_arch_x = left_arch.x
    

    left_width = abs(intrabrow_x - left_arch_x)
    right_width = abs(right_arch_x - intrabrow_x)

    return left_width, right_width 



def convert_rotate_matrix_to_euler(M):
    """
    converts a 4 by 4 rotation matrix
    into pitch, yaw and roll, returning 
    angles in degrees.
    """

    # M00 = cos(z)cos(y),  M10 = sin(z)sin(y)
    # and so abs_cosy = |cos(y)|
    abs_cosy = np.sqrt(M[0, 0]**2+M[1, 0]**2)

    # cos(0) = 90 which would mean a full rotation,
    #  this is a problematic case
    head_turned = abs_cosy < 1e-5

    if not head_turned:

        # This calculates the rotation angle around what 
        # we are calling the x axis

        # since M21 = cos(y)sin(x) and M22 = cos(y)cos(x),
        # arctan(sinx/cos(x)) will give x

        # same logic applies to getting angles y and z
        x = np.arctan2(M[2, 1], M[2, 2])

        #M20 = -sin(y) and so manipulating this gives y
        y = np.arctan2(-M[2, 0], abs_cosy)
        z = np.arctan2(M[1, 0], M[0, 0])

    else:

        # M 
        x = np.arctan2(-M[1, 2], M[1, 1])
        y = np.arctan2(-M[2, 0], abs_cosy)
        z = 0

    euler_angles_derees = np.degrees([x, y, z])
    return euler_angles_derees


def get_pitch_yaw_roll(img_path):
    """ takes in a ___ and returns the pitch, yaw and 
    roll of the face in the image
    """

    detection_result = get_detection_result_takes_path(img_path)

    matrix = detection_result.facial_transformation_matrixes[0]

    pitch, yaw, roll = convert_rotate_matrix_to_euler(matrix)

    return pitch, yaw, roll

def get_pitch_yaw_roll_takes_detection_result(detection_result):
    """ takes in a detection result and returns the pitch, yaw and 
    roll of the face in the image
    """

    matrix = detection_result.facial_transformation_matrixes[0]

    pitch, yaw, roll = convert_rotate_matrix_to_euler(matrix)

    return pitch, yaw, roll

def all_measurements(face: list):
    """
    Takes in a list of facial landmarks and 
    returns statements on a few key facial measurements
    """

    theta_left, theta_right = canthal_tilt(face)
    canthal = f" the canthal tilt of the left eye is {theta_left:.3g} and the right is {theta_right:.3g}. The ideal is 4 to 8 degrees and the eyes should have the same tilt"

    lip_ratio = top_bottom_lip_ratio(face)
    top_to_bottom_lip_ratio = f" the bottom lip is {lip_ratio:.3g} times as large as the top lip. The ideal is 1.6 "

    bottom_thirds_ratio = bottom_two_thirds(face)
    bottom_two_thirds_ratio = f" the middle third is {bottom_thirds_ratio:.3g} times as large as the bottom third. The ideal is roughly {1/0.9:.3g}. "

    nose_to_chin_ratio = nose_to_chin(face)
    nose_to_bottom_lip_to_chin = f" the ratio between the nose to the bottom of the lips and the bottom of the lips to the chin is {nose_to_chin_ratio:.3g}. The ideal is 1."

    left, right = arch_symmetry(face)
    arches = f"the horizontal distance form the centre to the left brow arch is {left:.3g} and to the right brow arch is {right:.3g}."

    return canthal, top_to_bottom_lip_ratio, bottom_two_thirds_ratio, nose_to_bottom_lip_to_chin, arches
from app.services.facial_measurement_functions import get_pitch_yaw_roll
from app.exceptions import InvalidFacePoseError

MAX_YAW = 2
MAX_PITCH = 10

def is_pose_okay(yaw, pitch):

    print("pose validation is running")
    if abs(yaw) > MAX_YAW:
        print("rejected because yaw is more than 2 degrees")
        raise InvalidFacePoseError(message="the face is turned by more than 2 degrees so we cannot assess facial symmetry",
                                   yaw=yaw, pitch=pitch, max_yaw=MAX_YAW, max_pitch=MAX_PITCH)

    if abs(pitch) > MAX_PITCH:
        print("rejected because pitch is more than 10 degrees")
        raise InvalidFacePoseError(message="the face is turned up or down by more than 10 degrees so we cannot assess facial symmetry",
                                   yaw=yaw, pitch=pitch, max_yaw=MAX_YAW, max_pitch=MAX_PITCH)


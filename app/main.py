from io import BytesIO

from app.models.responses import ImageAnalysisResponse
from app.services.image_validation import validate_image
from app.services.image_preprocessing import preprocess_image
from app.services.facial_measurement_functions import get_detection_result_takes_img

from app.services.facial_measurement_functions import rotate_landmarks_takes_img

from app.services.facial_measurement_functions import get_pitch_yaw_roll_takes_detection_result

from app.services.facial_measurement_functions import all_measurements

from app.services.pose_validation import is_pose_okay

from app.exceptions import InvalidFacePoseError

from fastapi import FastAPI, UploadFile, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from PIL import Image, UnidentifiedImageError

app = FastAPI(
    title="FaceMetric",
    description="analyses face",
    version="0.1.0",
)

ALLOWED_PHOTO_TYPES = {
    "image/jpeg",
    "image/png"
}

MAX_FILE_SIZE = 5 * 1024 *1024

@app.get("/")
def root() -> dict[str, str]:
    return{"message": "Hello, this is FaceMEtric!"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/about")
def about():
    return {"name": "Eve Durant", "Purpose": "I am making an app that analyses your face"}

@app.post("/analyse", response_model=ImageAnalysisResponse,)
async def analyse(file: UploadFile = File(...),  #
    ) -> ImageAnalysisResponse:
 # remove this code and move it into validate_image function
    validated_image = await validate_image(file)

    preprocessed_img = preprocess_image(validated_image)

    detection_result = get_detection_result_takes_img(preprocessed_img)

    pitch, yaw, _ = get_pitch_yaw_roll_takes_detection_result(detection_result)

    is_pose_okay(yaw, pitch)

    face = rotate_landmarks_takes_img(preprocessed_img)

    canthal, lip_ratio, bottom_two_thirds, nose_lip, arch = all_measurements(face)

    return ImageAnalysisResponse(
        # filename = file.filename,
        # content_type = file.content_type,
        # image_format = validated_image.image_format,
        # width = validated_image.width,
        # height = validated_image.height,
        # size_bytes = len(validated_image.image_bytes),
        # status = "validated",

        canthal_tilt = canthal,
        top_to_bottom_lip_ratio = lip_ratio,
        bottom_two_thirds_ratio = bottom_two_thirds,
        nose_to_bottom_lip_to_chin = nose_lip,
        arch_symmetry = arch,
    )

@app.exception_handler(InvalidFacePoseError)
async def invalid_face_angle(request, exclusion:InvalidFacePoseError):
    json_response = JSONResponse(status_code=422,
                                 content = {
                                     
                                 "detail": {"code": "face_angle_invalid",
                                 "message": exclusion.message,
                                 "yaw": exclusion.yaw,
                                 "pitch": exclusion.pitch,
                                 "max_yaw": exclusion.max_yaw,
                                 "max_pitch": exclusion.max_pitch}
                                 })
    return json_response
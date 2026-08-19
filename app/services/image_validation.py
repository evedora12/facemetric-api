from dataclasses import dataclass
from io import BytesIO

from fastapi import HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError

ALLOWED_PHOTO_TYPES = {
    "image/jpeg",
    "image/png"
}

MAX_FILE_SIZE = 5 * 1024 *1024

@dataclass
class ValidatedImage:
    image_bytes: bytes
    image_format: str
    width: int
    height: int

async def validate_image(file: UploadFile) -> ValidatedImage:
    if file.content_type not in ALLOWED_PHOTO_TYPES:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="only jpeg and png images are supported.",
        
            )
    
    image_bytes = await file.read()
    
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail="The uploaded image must be 5 MB or smaller.",
        )
    
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()
    
        with Image.open(BytesIO(image_bytes)) as image:
            image = image.convert("RGB")
            w, h = image.size
    
        with Image.open(BytesIO(image_bytes)) as image:
            image_type = image.format
    
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file isn't a valid image.",
        )
    if image_type is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The image format couldn't be determined",
        )

    # if not image_bytes:
    #     raise HTTPException(
    #         status_code=status.HTTP_400_BAD_REQUEST,
    #         detail="The image is empty."
    #     )
    
    return ValidatedImage(
        image_bytes=image_bytes,
        image_format=image_type,
        width=w,
        height=h,
    )

    
    
    
from pydantic import BaseModel

class ImageAnalysisResponse(BaseModel):
    #filename: str
    #content_type: str
    #image_format: str
    #width: int
    #height: int
    #size_bytes: int
    #status: str
    canthal_tilt: str
    top_to_bottom_lip_ratio: str
    bottom_two_thirds_ratio: str
    nose_to_bottom_lip_to_chin: str
    arch_symmetry: str

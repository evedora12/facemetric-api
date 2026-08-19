from fastapi.testclient import TestClient
from io import BytesIO
from PIL import Image
from app.main import app

client = TestClient(app)

# def create_test_image(
#         #creates a real image entirely in memory
#     image_format: str = "JPEG",
#     width: int = 100,
#     height: int = 100,
# ) -> bytes:    # this is documenting the fact that the funciton returns a bytes object
#     buffer = BytesIO()

#     image = Image.new(
#         mode="RGB",
#         size=(width, height),
#     )
#     image.save(buffer, format=image_format)

#     return buffer.getvalue()

def test_analyse_valid_jpeg_returns_200() -> None:

    #Create a test image
    image_path = (r"/Users/evedurant/Desktop/facemetric-api/tests/test_image.png")
    
    #image_bytes = create_test_image(image_format="JPEG", width = 100, height =150,)
    with Image.open(image_path) as image:
        w, h = image.size

    # Call endpoint - the digital location where the API recieves API requests
    # building a simulated file upload
    with open(image_path, "rb") as image_file:
        response = client.post(
            "/analyse",
            files={      # this means send this request as a multipart file upload
                "file": (       # since the parameter is called (file: UploadFile = File(...)):
                    "face.png",                 #file.filename
                    image_file,      #image_bytes
                    "image/png",               # file.content_type


                )
            },
        )

    # checking that the outcome is okay (200 means okay)
    assert response.status_code == 200

    response_data = response.json()

    assert response_data["filename"] == "face.png"
    assert response_data["content_type"] == "image/png"
    assert response_data["image_format"] == "PNG"
    assert response_data["width"] == w               

    assert response_data["height"] == h
   
    assert response_data["status"] == "validated"



def test_analyse_rejects_unsupported_content_type() -> None:

    file_path = (r"/Users/evedurant/Desktop/facemetric-api/tests/test_image.png")
        

    response = client.post(
        "/analyse",
        files={
            "file": (
                "test_pdf.pdf",
                file_path,
                "application/pdf",
            )
        },
    )

    assert response.status_code == 415
    assert response.json() == {
        "detail": "only jpeg and png images are supported."
    }


def test_analyse_rejects_oversized_file() -> None:

    image_path = (r"tests/oversized_img.jpg")

    with open(image_path, "rb") as image_file:
        response = client.post(
            "/analyse",
            files={
                "file": (
                    "oversized_img.jpg",
                    image_file,
                    "image/jpeg"
                )
            },
        )

    assert response.status_code == 413
    assert response.json() == {
        "detail": "The uploaded image must be 5 MB or smaller."
    }    

def test_analyse_rejects_empty_file() -> None:
    response = client.post(
        "/analyse",
            files={
                "file": (
                    "empty.jpg",
                    b"",
                    "image/jpeg"
                )
            },
        )
    assert response.status_code == 400
    assert response.json() == { 
        "detail": "The uploaded file isn't a valid image."
    }

def test_analyse_requires_file() -> None:
    response = client.post("/analyse")

    assert response.status_code == 422
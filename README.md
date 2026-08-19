# FaceMetric API

This is a compputer vision API which uses the MediaPipe facial landmark detection to identify the features of the face, and their coordinates, and therefore geometry facial measurements.

The project uses FastAPI, MediaPipe, OpenCV, NumPy, Pillow.

## The Process

### In the current v1 the project does the following:

- Image upload validation (rejects invalid images)
- Processes images into RGB NumPy arrays
- Identifies 468 key features of the face using MediaPipe Face Landmarker
- Estimates head pose using pitch, yaw, and roll, calculated from a rotation matrix given by MediaPipe.
- Rejects yaw (looking left/right) of more than 2 degrees and a pitch (looking up/down) of more than 10 degrees. Essentially rejects non-frontal faces
- Identifies roll (how much the head is turned towards the shoulder) and rotates it so that the face is completely vertical.
- Calculates facial geometry and ratio measurements
- Provides structured JSON response.

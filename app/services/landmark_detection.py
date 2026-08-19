import cv2 as cv


def canny_edge(img, lower=100, upper=200): 
    # Load image in grayscale


    gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred_image = cv.GaussianBlur(gray_image, (3, 3), 0)

    # Perform Canny edge detection
    edges = cv.Canny(blurred_image, lower, upper)

    # # Display results
    # cv.imshow('Edges', edges)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    return edges
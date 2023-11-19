from app import app
import pytest
from sklearn import datasets
import numpy as np
from io import BytesIO
from PIL import Image

# def get_image_bytes(image):
#     pil_image = Image.fromarray(image)
#     img_byte_arr = BytesIO()
#     pil_image.save(img_byte_arr, format='PNG')
#     img_byte_arr = img_byte_arr.getvalue()
#     return img_byte_arr
def get_image_bytes(image):
    pil_image = Image.fromarray(image.astype('uint8'))  # Ensure the data type is uint8
    img_byte_arr = BytesIO()
    pil_image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr

def test_post_compare_digits():
    digits = datasets.load_digits()
    X, y = digits.images, digits.target

    # Iterate over each digit
    for digit in range(10):
        print(f"----[DEBUG] PROCESSING: {digit}")
        # Find indices for the current digit
        indices = [i for i, label in enumerate(y) if label == digit]

        # Select the first two images for this digit
        image1_bytes = get_image_bytes(X[indices[0]])
        image2_bytes = get_image_bytes(X[indices[1]])

        # Send POST request with these images
        response = app.test_client().post(
            '/compare_digits', 
            data={'image1': (BytesIO(image1_bytes), 'image1.png'), 
                  'image2': (BytesIO(image2_bytes), 'image2.png')},
            content_type='multipart/form-data'
        )

        # Assert the status code and response
        assert response.status_code == 200
        print(f"----[DEBUG] ASSERRT SUCCESS FOR STATUS: {response.status_code}")
        assert response.get_json()['same_digit'] is True
        print(f"----[DEBUG] ASSERRT SUCCESS FOR DIGIT: {digit}")

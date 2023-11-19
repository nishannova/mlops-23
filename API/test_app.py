from app import app
import pytest
from sklearn import datasets
from PIL import Image
from io import BytesIO

def get_image_bytes(image):
    pil_image = Image.fromarray(image.astype('uint8'))
    img_byte_arr = BytesIO()
    pil_image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr

def test_post_predict_digit():
    digits = datasets.load_digits()
    X, y = digits.images, digits.target

    for digit in range(10):
        print(f"----[DEBUG] PROCESSING: {digit}")
        index = next(i for i, label in enumerate(y) if label == digit)

        image_bytes = get_image_bytes(X[index])

        response = app.test_client().post(
            '/predict_digit', 
            data={'image': (BytesIO(image_bytes), 'image.png')},
            content_type='multipart/form-data'
        )

        assert response.status_code == 200
        print(f"----[DEBUG] ASSERT SUCCESS FOR STATUS: {response.status_code}")
        assert response.get_json()['predicted_digit'] == digit
        print(f"----[DEBUG] ASSERT SUCCESS FOR DIGIT: {digit}")

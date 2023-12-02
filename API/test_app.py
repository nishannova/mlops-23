from app import app
import pytest
from sklearn import datasets
from PIL import Image
from io import BytesIO
import os
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.datasets import load_digits
import re

digits = load_digits()

# def get_image_bytes(image):
#     pil_image = Image.fromarray(image.astype('uint8'))
#     img_byte_arr = BytesIO()
#     pil_image.save(img_byte_arr, format='PNG')
#     img_byte_arr = img_byte_arr.getvalue()
#     return img_byte_arr

def get_image_bytes(image):
    pil_image = Image.fromarray(image.astype('uint8'))
    img_byte_arr = BytesIO()
    pil_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr

# def test_post_predict_digit():
#     digits = datasets.load_digits()
#     X, y = digits.images, digits.target

#     for digit in range(10):
#         print(f"----[DEBUG] PROCESSING: {digit}")
#         index = next(i for i, label in enumerate(y) if label == digit)

#         image_bytes = get_image_bytes(X[index])

#         response = app.test_client().post(
#             '/predict_digit', 
#             data={'image': (BytesIO(image_bytes), 'image.png')},
#             content_type='multipart/form-data'
#         )

#         assert response.status_code == 200
#         print(f"----[DEBUG] ASSERT SUCCESS FOR STATUS: {response.status_code}")
#         assert response.get_json()['predicted_digit'] == digit
#         print(f"----[DEBUG] ASSERT SUCCESS FOR DIGIT: {digit}")

def test_post_predict_model():
    digits = load_digits()
    X, y = digits.images, digits.target

    # Choose an index for testing
    index = 100  # Replace with your chosen index

    image_bytes = get_image_bytes(X[index])
    original_digit = y[index]

    model_types = ['svm', 'lr', 'tree']

    for model_type in model_types:
        response = app.test_client().post(
            f'/predict/{model_type}', 
            data={'image': (BytesIO(image_bytes.getvalue()), 'image.png')},
            content_type='multipart/form-data'
        )

        assert response.status_code == 200
        predicted_digit = response.get_json()['predicted_digit']
        print(f"Response for {model_type} model: {predicted_digit}")
        print(f"WHICH WAS ORIGINALLY: {original_digit}")


LR_MODEL_DIR = "../q2_models"

# def test_model_type():
#     model_filename = '<rollno>_lr_<solver_name>.joblib'  # Replace with actual filename
#     model = joblib.load(model_filename)
#     assert isinstance(model, LogisticRegression), "Loaded model is not a Logistic Regression model"

def test_solver_name_in_filename():
    model_filenames = [
        "m22aie208_lr_lbfgs.joblib",
        "m22aie208_lr_liblinear.joblib",
        "m22aie208_lr_newton-cg.joblib",
        "m22aie208_lr_sag.joblib",
        "m22aie208_lr_saga.joblib"
    ]
    for model_filename in model_filenames:
        filename_solver = model_filename.split("_")[-1].split(".")[0]
        print(f"\n[DEBUG]: FILENAME SOLVER: {filename_solver}")
        # Load the model and get its solver parameter
        model = joblib.load(os.path.join(LR_MODEL_DIR, model_filename))
        model_solver = model.get_params()['solver']
        print(f"[DEBUG]: ACTUAL SOLVER: {model_solver}")
        assert filename_solver == model_solver, f"Solver name in filename ({filename_solver}) does not match the model's solver ({model_solver})"
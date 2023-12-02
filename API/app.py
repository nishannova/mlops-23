from flask import Flask, request, jsonify
import joblib
from PIL import Image
import numpy as np
from io import BytesIO
import os

# Load the model
# model = joblib.load('/digits/API/best_model.pkl')
model = joblib.load('best_model.pkl')
SAVED_MODEL_DIR = "../saved_model"
app = Flask(__name__)

from sklearn.preprocessing import Normalizer

normalizer = Normalizer(norm='l2')

models = {}


def load_model():
    models['svm'] = joblib.load(os.path.join(SAVED_MODEL_DIR, 'svm_model.joblib'))
    models['lr'] = joblib.load(os.path.join(SAVED_MODEL_DIR, 'lr_model.joblib'))
    models['tree'] = joblib.load(os.path.join(SAVED_MODEL_DIR, 'tree_model.joblib'))

# Load models
load_model()

# @app.route('/predict_digit', methods=['POST'])
# def predict_digit():
#     if 'image' not in request.files:
#         return jsonify(error='Please provide an image.'), 400

#     image_bytes = request.files['image'].read()
#     image = Image.open(BytesIO(image_bytes)).convert('L')
#     image = image.resize((8, 8), Image.LANCZOS)
    
#     # image_arr = np.array(image).reshape(1, -1)
#     # pred = model.predict(image_arr)
    
#     image_arr = np.array(image).reshape(1, -1)
    # image_arr_normalized = normalizer.transform(image_arr)
    # pred = model.predict(image_arr_normalized)

#     return jsonify(predicted_digit=int(pred[0]))

@app.route('/predict/<model_type>', methods=['POST'])
def predict_digit(model_type):
    if model_type not in models:
        return jsonify(error='Model type not supported.'), 400

    if 'image' not in request.files:
        return jsonify(error='Please provide an image.'), 400

    model = models[model_type]

    image_bytes = request.files['image'].read()
    image = Image.open(BytesIO(image_bytes)).convert('L')
    image = image.resize((8, 8), Image.LANCZOS)
    
    image_arr = np.array(image).reshape(1, -1)
    image_arr_normalized = normalizer.transform(image_arr)
    pred = model.predict(image_arr_normalized)

    return jsonify(predicted_digit=int(pred[0]))


@app.route('/compare_digits', methods=['POST'])
def compare_digits():
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify(error='Please provide two images.'), 400

    image1_bytes = request.files['image1'].read()
    image2_bytes = request.files['image2'].read()

    image1 = Image.open(BytesIO(image1_bytes)).convert('L')
    image2 = Image.open(BytesIO(image2_bytes)).convert('L')

    image1 = image1.resize((8, 8), Image.LANCZOS)
    image2 = image2.resize((8, 8), Image.LANCZOS)
    
    # image1_arr = np.array(image1).reshape(1, -1)
    # image2_arr = np.array(image2).reshape(1, -1)

    # pred1 = model.predict(image1_arr)
    # pred2 = model.predict(image2_arr)
    
    image1_arr = np.array(image1).reshape(1, -1)
    image2_arr = np.array(image2).reshape(1, -1)

    image1_arr_normalized = normalizer.transform(image1_arr)
    image2_arr_normalized = normalizer.transform(image2_arr)
    print(f"[DEBUG]: Normalized the image")
    pred1 = model.predict(image1_arr_normalized)
    pred2 = model.predict(image2_arr_normalized)

    result = pred1 == pred2

    return jsonify(same_digit=bool(result[0]))

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5001)

from flask import Flask, request, jsonify
import joblib
from PIL import Image
import numpy as np
from io import BytesIO
# Load the model
# model = joblib.load('/digits/API/best_model.pkl')
model = joblib.load('best_model.pkl')

app = Flask(__name__)


@app.route('/predict_digit', methods=['POST'])
def predict_digit():
    if 'image' not in request.files:
        return jsonify(error='Please provide an image.'), 400

    image_bytes = request.files['image'].read()
    image = Image.open(BytesIO(image_bytes)).convert('L')
    image = image.resize((8, 8), Image.LANCZOS)
    
    image_arr = np.array(image).reshape(1, -1)
    pred = model.predict(image_arr)

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
    
    image1_arr = np.array(image1).reshape(1, -1)
    image2_arr = np.array(image2).reshape(1, -1)

    pred1 = model.predict(image1_arr)
    pred2 = model.predict(image2_arr)

    result = pred1 == pred2

    return jsonify(same_digit=bool(result[0]))

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, jsonify
import joblib
from PIL import Image
import numpy as np
from io import BytesIO
# Load the model
model = joblib.load('../best_model.pkl')

app = Flask(__name__)



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

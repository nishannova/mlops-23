import requests
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from io import BytesIO

# Load the digits dataset
digits = load_digits()

# # Select two digits
# index1, index2 = 0, 1  # Replace with your chosen indices

# # Convert the images to bytes
# image1_bytes = BytesIO()
# image2_bytes = BytesIO()
# plt.imsave(image1_bytes, digits.images[index1], cmap='gray', format='png')
# plt.imsave(image2_bytes, digits.images[index2], cmap='gray', format='png')
# image1_bytes.seek(0)
# image2_bytes.seek(0)

# # Prepare the request
# # url = 'http://ml-ops-asignment-5-app.azurewebsites.net:5001/compare_digits'
# url = 'http://0.0.0.0:5000/compare_digits'

# files = {
#     'image1': ('image1.png', image1_bytes, 'image/png'),
#     'image2': ('image2.png', image2_bytes, 'image/png')
# }

# # Send the request
# response = requests.post(url, files=files)

# # Process the response
# if response.status_code == 200:
#     print(response.json())
# else:
#     print('Failed to get a response from the server:', response.status_code)

def send_prediction_request(image_bytes, model_type, url_base="http://0.0.0.0:5001/predict/"):
    """
    Send a prediction request to the specified model endpoint with an image.

    :param image_bytes: BytesIO object containing the image
    :param model_type: Type of the model ('svm', 'lr', 'tree')
    :param url_base: Base URL of the Flask API
    :return: Response from the API
    """
    url = url_base + model_type
    image_bytes.seek(0)
    files = {'image': ('image.png', image_bytes, 'image/png')}
    response = requests.post(url, files=files)
    return response

# Load the digits dataset
digits = load_digits()

# Select an image
index = 100  # Replace with your chosen index

# Convert the image to bytes
image_bytes = BytesIO()
plt.imsave(image_bytes, digits.images[index], cmap='gray', format='png')
image_bytes.seek(0)

# Model types
model_types = ['svm', 'lr', 'tree']

# Send the request for each model type
for model_type in model_types:
    response = send_prediction_request(image_bytes, model_type)
    if response.status_code == 200:
        print(f"Response for {model_type} model:", response.json())
        print(f"WHICH WAS ORIGINALLY: {digits['target'][index]}")
    else:
        print(f"Failed to get a response from the {model_type} model server:", response.status_code)


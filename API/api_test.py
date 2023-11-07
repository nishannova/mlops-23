import requests
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from io import BytesIO

# Load the digits dataset
digits = load_digits()

# Select two digits
index1, index2 = 0, 1  # Replace with your chosen indices

# Convert the images to bytes
image1_bytes = BytesIO()
image2_bytes = BytesIO()
plt.imsave(image1_bytes, digits.images[index1], cmap='gray', format='png')
plt.imsave(image2_bytes, digits.images[index2], cmap='gray', format='png')
image1_bytes.seek(0)
image2_bytes.seek(0)

# Prepare the request
url = 'http://0.0.0.0:5000/compare_digits'
files = {
    'image1': ('image1.png', image1_bytes, 'image/png'),
    'image2': ('image2.png', image2_bytes, 'image/png')
}

# Send the request
response = requests.post(url, files=files)

# Process the response
if response.status_code == 200:
    print(response.json())
else:
    print('Failed to get a response from the server:', response.status_code)

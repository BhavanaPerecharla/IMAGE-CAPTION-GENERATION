from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
model_path = "IMAGE_CAPTION_best_model.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

model = load_model(model_path)

def preprocess_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        return img_array
    except Exception as e:
        raise ValueError(f"Error processing image: {e}")

def generate_caption(img_path):
    try:
        img_array = preprocess_image(img_path)
        prediction = model.predict(img_array)
        caption = "Generated caption"  # Replace with actual caption generation logic
        return caption
    except Exception as e:
        raise ValueError(f"Error generating caption: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected for uploading'}), 400

    try:
        if file:
            img_path = os.path.join("/tmp", file.filename)
            file.save(img_path)
            caption = generate_caption(img_path)
            os.remove(img_path)
            return jsonify({'caption': caption})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

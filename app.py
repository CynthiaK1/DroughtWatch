from flask import Flask, render_template, request
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet50 import preprocess_input
from keras.models import load_model
import numpy as np
import os

app = Flask(__name__)
model = load_model('KerasImage_classification_model.h5')

# Use the same data generators for preprocessing
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    # Save uploaded image
    image_file = request.files['imagefile']
    image_path = os.path.join("images", image_file.filename)
    image_file.save(image_path)

    # Preprocess the image for prediction using the same preprocessing as during training
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, *image.shape))
    image = datagen.standardize(image)

    # Make prediction
    prediction = model.predict(image)
    class_index = np.argmax(prediction)
    classes = {0: 'normal', 1: 'low', 2: 'critical'}  # Update with your actual class labels
    classification = classes[class_index]

    return render_template('index.html', prediction=classification)

if __name__ == '__main__':
    app.run(port=3000, debug=True)

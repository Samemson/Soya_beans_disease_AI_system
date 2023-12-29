Soybean Disease Detection
This project uses a deep learning model to detect diseases in soybean leaf images. It can classify images into 6 classes:

    •	Bacterial Pustule
    •   Frogeye Leaf Spot
    •   Healthy
    •   Sudden Death Syndrome
    •   Target Spot
    •   Yellow Mosaic

Setup

    1. Install dependencies:
        •   Keras
        •   TensorFlow
        •   OpenCV
        •   PIL
        •   NumPy
        •   Pyttsx3
    2. Download the pre-trained Keras model `keras_model.h5` and label mappings `labels.txt`.
    3. Place test images in the `test/` folder and solution texts in `solutions/`.
Usage
    1. Run `main.py` and select an image path.
    2. The program will classify the image, draw a bounding box, and print/speak the results.
    3. To detect objects in real-time video, uncomment the `video()` function call.
Model Details
The model is a convolutional neural network trained on 224x224 RGB images. It was trained on a dataset of over 1000 soybean leaf images.

The model architecture and training procedure are described in 
model.py
.

Future Improvements
    •   Add a web/mobile interface for easier use
    •   Improve model performance with data augmentation and transfer learning
    •   Detect multiple objects per image with a bounding box around each
    •   Deploy model to a cloud service for on-device/server inference
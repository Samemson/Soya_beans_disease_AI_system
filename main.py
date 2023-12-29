from keras.models import load_model
import numpy as np
import cv2
from PIL import Image, ImageOps
import tensorflow as tf
import pyttsx3 as ts
import os

engine = ts.init()

# Image detection path; select only one
# image_path = "test/Bacterial_Pustule_test.png"
# image_path = "test/Frogeye_Leaf_Spot_test.jpg"
# image_path = "test/Healthy_test.jpg"
# image_path = "test/Sudden_Death_Syndrome_test.jpg"
image_path = "test/Target_ls_test.jpg"
# image_path = "test/Yellow_Mosaic_test.png"

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_model.h5", compile=False)

# Load the labels
with open('labels.txt') as f:
    class_names = f.readlines()

# Create the array of the right shape to feed into the keras model
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Predicts the model
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]

# Prints and say solutions of image detection
def print_detection_text(image_path):
    if image_path == "test/Bacterial_Pustule_test.png":
        print(open('solutions/bacterial_pustule.txt').read())
        engine.say(open('solutions/bacterial_pustule.txt').read())
        engine.runAndWait()

    elif image_path == "test/Sudden_Death_Syndrome_test.jpg":
        print(open('solutions/sudden_death_syndrome.txt').read())
        engine.say(open('solutions/sudden_death_syndrome.txt').read())
        engine.runAndWait()

    elif image_path == "test/Target_ls_test.jpg":
        print(open('solutions/target_leaf_spot.txt').read())
        engine.say(open('solutions/target_leaf_spot.txt').read())
        engine.runAndWait()

    elif image_path == "test/Yellow_Mosaic_test.png":
        print(open('solutions/yellow_mosaic.txt').read())
        engine.say(open('solutions/yellow_mosaic.txt').read())
        engine.runAndWait()

    elif image_path == "test/Frogeye_Leaf_Spot_test.jpg":
        print(open('solutions/frogeye_leaf_spot.txt').read())
        engine.say(open('solutions/frogeye_leaf_spot.txt').read())
        engine.runAndWait()

    elif image_path == "test/Healthy_test.jpg":
        print("Healthy Plant")
        engine.say("Healthy Plant")
        engine.runAndWait()


# Function that carries out image detecton
def detect_objects_in_image(image_path):
    # Load the image
    with Image.open(image_path) as image:
        image = image.convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    predictions = model.predict(data)
    index = np.argmax(predictions)
    class_name = class_names[index]
    confidence_score = predictions[0][index]

    print("Class:", class_name[2:])
    print("Confidence Score:", confidence_score)

    # Speech; saying what was detected
    engine.say('Leaf detected is ' + class_name[2:])
    engine.runAndWait()
    print(f'Leaf detected is {class_name[2:]}')
    print_detection_text(image_path)
        

    # Extract bounding box coordinates
    bbox = (0, 0, 224, 224)
    ymin, xmin, ymax, xmax = bbox

    # Rescale bounding box coordinates to the original image size
    imH, imW, _ = image_array.shape
    xmin, xmax, ymin, ymax = (xmin * imW, xmax * imW, ymin * imH, ymax * imH)
    xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)

    # Draw bounding box and class label on the image
    result_image = image_array.copy()
    cv2.rectangle(result_image, (xmin, ymin), (xmax, ymax), color=(255, 0, 255), thickness=2)
    cv2.putText(result_image, f'{class_name[2:-1]}: {confidence_score:.2f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 1)

    # Display the image
    cv2.imshow("Object Detection Result", result_image)
    print_detection_text(class_name)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Perform object detection on the input image
def createBoundingBox(image, threshold=0.5):
    # Preprocess the image
    resized_image = cv2.resize(image, (224, 224))
    input_image = Image.fromarray(resized_image)
    input_image = ImageOps.exif_transpose(input_image)
    input_image = ImageOps.fit(input_image, (224, 224), Image.Resampling.LANCZOS)
    input_array = np.array(input_image)

    # Normalize the image
    normalized_image_array = (input_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    print("Class:", class_name[2:], end="")
    print("Confidence Score:", confidence_score)

    bbox = (0, 0, image.shape[1], image.shape[0])

    # Draw bounding box and class label
    cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color=(0, 255, 0), thickness=2)

    # Display the result on the image
    cv2.putText(image, f'{class_name[2:-1]}: {confidence_score:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)

    return image

# Function that carries out video detecton
def video():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error opening file..")
        return
    (success, image) = cap.read()
    while success:
        # engine.say('Leaf detected is' + class_name[2:])
        # engine.runAndWait()
        # print("Speech Running")

        # Perform object detection and draw bounding box on the frame
        image_with_boxes = createBoundingBox(image)

        cv2.imshow("Object Detection", image_with_boxes)
    
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        (success, image) = cap.read()
    
    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Call the video function; select either image or video detecton
detect_objects_in_image(image_path)
# video()

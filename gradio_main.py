import cv2
import gradio as gr
import numpy as np
import joblib

from api import is_beautiful
from facial_image import FacialImage, convert_to_numpy_array

title = "Sytoss System: Beauty Recognition"
description = "A model to classify is face beautiful or not."

target_width = 512
facial = FacialImage()
loaded_svm_classifier = joblib.load('svm_model.pkl')


def process_image(input_image: np.ndarray):
    if input_image is not None:

        output_image = input_image

        input_height, input_width, _ = input_image.shape
        if input_width > target_width:
            scale_factor = float(target_width / input_width)
            output_width = int(input_width * scale_factor)
            output_height = int(input_height * scale_factor)
            dsize = (output_width, output_height)
            output_image = cv2.resize(input_image, dsize)

        ratios, output_image = facial.calculate_ratios(output_image)
        ratios_vector = convert_to_numpy_array(ratios)

        class_probabilities = loaded_svm_classifier.predict_proba([ratios_vector])

        predicted_class = np.argmax(class_probabilities)

        probability_of_predicted_class = class_probabilities[0, predicted_class]

        return output_image, predicted_class, probability_of_predicted_class
    return None, None


iface = gr.Interface(
    process_image,
    inputs=gr.inputs.Image(),
    outputs=["image", gr.Number(label="Predicted class"), gr.Number(label="Probability")],
    title=title,
    description=description)

iface.launch()

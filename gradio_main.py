import cv2
import gradio as gr
import numpy

from facial_image import FacialImage

# Create title, description and article strings
title = "Sytoss system: Recognition beauty"
description = "An model to classify images of face beauty or no."
article = "Created at me."

target_width = 512
facial = FacialImage()


# Функция для обработки входных данных
def process_image(input_image: numpy.ndarray):
    print(input_image.shape)

    # Image
    output_image = input_image

    input_height, input_width, _ = input_image.shape
    if input_width > target_width:
        scale_factor = float(target_width / input_width)
        output_width = int(input_width * scale_factor)
        output_height = int(input_height * scale_factor)
        dsize = (output_width, output_height)
        output_image = cv2.resize(input_image, dsize)

    output_image = facial.landmarks(output_image)

    prediction_class = 1.0  # Замените на ваше значение
    prediction_proba = 2.0  # Замените на ваше значение
    return output_image, prediction_class, prediction_proba


### Создание интерфейса Gradio
iface = gr.Interface(
    process_image,
    inputs=gr.inputs.Image(),
    outputs=["image", gr.Number(label="Predicted beauty"), gr.Number(label="Prediction probability")],
    title=title,
    description=description,
    article=article)

iface.launch()

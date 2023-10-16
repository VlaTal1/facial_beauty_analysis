import cv2
import gradio as gr
import numpy

from api import is_beautiful
from facial_image import FacialImage

# Create title, description and article strings
title = "Sytoss system: Recognition beauty"
description = "An model to classify images of face beauty or no."
article = "Created at me."

target_width = 512
facial = FacialImage()


# Функция для обработки входных данных
def process_image(input_image: numpy.ndarray):
    if input_image is not None:
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

        ratios, output_image = facial.calculate_ratios(output_image)
        ratios = facial.convert_to_numpy_array(ratios)

        return output_image, is_beautiful(ratios)
    return None, None


### Создание интерфейса Gradio
iface = gr.Interface(
    process_image,
    inputs=gr.inputs.Image(),
    outputs=["image", gr.Number(label="Predicted beauty")],
    title=title,
    description=description,
    article=article)

iface.launch()

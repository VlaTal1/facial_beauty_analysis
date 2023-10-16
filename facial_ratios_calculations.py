import cv2

from facial_image import FacialImage
from utils import write_line_to_csv

facial = FacialImage()

with open('SCUT-FBP5500_v2/train_test_files/split_of_60%training and 40%testing/train.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        split_line = line.split(' ')
        img_name = split_line[0]
        mark = round(float(split_line[1].replace('\n', '')), 0)
        image = cv2.imread(f'SCUT-FBP5500_v2/Images/{img_name}')
        ratios = facial.calculate_ratios(image=image, is_normalized=True)
        write_line_to_csv(ratios, f"csv/mark{mark}_normalized")

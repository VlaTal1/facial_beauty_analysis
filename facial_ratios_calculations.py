from facial_image import FacialImage
from utils import write_line_to_csv

facial = FacialImage()

with open('SCUT-FBP5500_v2/train_test_files/split_of_60%training and 40%testing/train.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        split_line = line.split(' ')
        img_name = split_line[0]
        mark = round(float(split_line[1].replace('\n', '')), 0)
        if mark == 5:
            ratios = facial.calculate_ratios(image_path=f'SCUT-FBP5500_v2/Images/{img_name}', is_normalized=True)
            write_line_to_csv(ratios, "csv/mark5_normalized")
        elif mark == 4:
            ratios = facial.calculate_ratios(image_path=f'SCUT-FBP5500_v2/Images/{img_name}', is_normalized=True)
            write_line_to_csv(ratios, "csv/mark4_normalized")
        elif mark == 3:
            ratios = facial.calculate_ratios(image_path=f'SCUT-FBP5500_v2/Images/{img_name}', is_normalized=True)
            write_line_to_csv(ratios, "csv/mark3_normalized")
        elif mark == 2:
            ratios = facial.calculate_ratios(image_path=f'SCUT-FBP5500_v2/Images/{img_name}', is_normalized=True)
            write_line_to_csv(ratios, "csv/mark2_normalized")
        elif mark == 1:
            ratios = facial.calculate_ratios(image_path=f'SCUT-FBP5500_v2/Images/{img_name}', is_normalized=True)
            write_line_to_csv(ratios, "csv/mark1_normalized")

from facial_image import FacialImage
from utils import write_to_csv

facial = FacialImage()

mark5 = []
mark4 = []
mark3 = []
mark2 = []
mark1 = []

with open('SCUT-FBP5500_v2/train_test_files/split_of_60%training and 40%testing/train.txt',
          'r') as file:
    lines = file.readlines()
    for line in lines:
        split_line = line.split(' ')
        img_name = split_line[0]
        mark = round(float(split_line[1].replace('\n', '')), 0)
        if mark == 5 and len(mark5) != 10:
            ratios = facial.calculate_ratios(image_path=f'SCUT-FBP5500_v2/Images/{img_name}')
            mark5.append(ratios)
        elif mark == 4 and len(mark4) != 10:
            ratios = facial.calculate_ratios(image_path=f'SCUT-FBP5500_v2/Images/{img_name}')
            mark4.append(ratios)
        elif mark == 3 and len(mark3) != 10:
            ratios = facial.calculate_ratios(image_path=f'SCUT-FBP5500_v2/Images/{img_name}')
            mark3.append(ratios)
        elif mark == 2 and len(mark2) != 10:
            ratios = facial.calculate_ratios(image_path=f'SCUT-FBP5500_v2/Images/{img_name}')
            mark2.append(ratios)
        elif mark == 1 and len(mark1) != 10:
            ratios = facial.calculate_ratios(image_path=f'SCUT-FBP5500_v2/Images/{img_name}')
            mark1.append(ratios)

    write_to_csv(mark5, "csv/mark5")
    write_to_csv(mark4, "csv/mark4")
    write_to_csv(mark3, "csv/mark3")
    write_to_csv(mark2, "csv/mark2")
    write_to_csv(mark1, "csv/mark1")
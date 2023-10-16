import numpy
import csv


def write_to_csv(ratios, file_name):
    rows = []

    for i in range(len(ratios)):
        row = []
        for key, value in ratios[i].items():
            row.append(value)
        rows.append(row)

    with open(file_name, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(rows)
        csvfile.close()


def write_line_to_csv(ratios, file_name):
    row = []

    for key, value in ratios.items():
        row.append(value)

    with open(file_name, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(row)


def standard_deviation(values):
    _mean = numpy.mean(values)
    differences = [(value - _mean) ** 2 for value in values]
    sum_of_differences = sum(differences)
    return (sum_of_differences / (len(values) - 1)) ** 0.5


def normalization(ratios_list):
    _mean = numpy.mean(list(ratios_list.values()))
    _deviation = standard_deviation(list(ratios_list.values()))

    z_score = {}
    for z_key, z_i in ratios_list.items():
        z_score[z_key] = (z_i - _mean) / _deviation

    lb = 0
    ub = 1.618
    _min = min(z_score.values())
    _max = max(z_score.values())

    linear = {}
    for z_key, z_i in z_score.items():
        linear[z_key] = lb + (z_i - _min) * (ub - lb) / (_max - _min)

    return linear

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

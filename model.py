import csv

file_path = "SmartRocketTrainingData20260507_191122.csv"
data = []

with open(file_path,'r') as file:
    reader = csv.reader(file)
    header = next(reader)

    for row in reader:
        try:
            presed_row_value = [float(row_value) for row_value in row]
            data.append(presed_row_value)
        except ValueError:
            continue

print(data)
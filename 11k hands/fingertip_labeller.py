# automatically label fingertips on 11k hands
# dorsal view only
import csv

with open('HandInfo.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    reader.readline() # skip header
    for row in reader:
        aspect = row[6]
        filename = row[7]
        
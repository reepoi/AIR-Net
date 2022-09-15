import csv
import numpy as np
import matplotlib.pyplot as plt

FILE_LOCATION = 'vel_2Daneu_crop.0.csv'

with open(FILE_LOCATION) as f:
    reader = csv.reader(f, delimiter=',')
    rows = []
    for row in reader:
        rows.append(row)

# skip the CSV header
rows = rows[1:]

# skip the CSV header
data = np.array(rows, dtype=np.float64)

xs, ys = data[:, 3], data[:, 4] # spatial coordinates
us, vs = data[:, 0], data[:, 1] # velocity vector components

# you may adjust the scale parameter, or remove it to use matplotlib's auto-scaling
# the larger the scale parameter value, the smaller the vectors are in the plot
plt.quiver(xs, ys, us, vs, scale=400)

plt.show()

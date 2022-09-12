import csv
import numpy as np
import matplotlib.pyplot as plt

x, y = np.meshgrid(np.linspace(-5,5,40),np.linspace(-5,5,40))

with open('amir/vel_2Daneu_crop.csv') as f:
    reader = csv.reader(f, delimiter=',')
    rows = []
    for row in reader:
        rows.append(row)

data = np.array(rows, dtype=np.float64)

data = data[:3200, :]

v_x, v_y = data[0::2], data[1::2]

v = np.array([[[x, y] for x, y in zip(r_x, r_y)] for r_x, r_y in zip(v_x, v_y)])

v0 = v[:, 0, :]

print(v0, v0.shape)

plt.quiver(x, y, v0[:, 0], v0[:, 1])

plt.show()

# np.savetxt('testpy_x.csv', v0[:,0], delimiter=',')

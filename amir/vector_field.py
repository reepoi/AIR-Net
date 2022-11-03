import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.interpolate as interp

FILE_LOCATION = 'cropped_2D_aneurysm/vel_2Daneu_crop.0.csv'

data_pd = pd.read_csv(FILE_LOCATION)[0:100]
u_col, v_col, _, x_col, y_col, _ = data_pd.columns
data_pd = data_pd.set_index([x_col, y_col]) # set index to (x, y)
data_pd = data_pd.groupby(level=data_pd.index.names).first() # remove duplicate (x, y)
# data_pd = data_pd[~data_pd.index.duplicated(keep='first')] # remove duplicate points
# data_dict = data_pd.to_dict('index') # create mapping (x, y) -> (u, v)

xs = data_pd.index.get_level_values(x_col).to_numpy()
ys = data_pd.index.get_level_values(y_col).to_numpy()
us = data_pd[u_col].to_numpy()
vs = data_pd[v_col].to_numpy()

interp_us = interp.interp2d(xs, ys, us)
interp_vs = interp.interp2d(xs, ys, vs)

grid_density = 100
grid_xs = np.linspace(np.min(xs), np.max(xs), num=grid_density)
grid_ys = np.linspace(np.min(ys), np.max(ys), num=grid_density)
grid_us = interp_us(grid_xs, grid_ys)
grid_vs = interp_vs(grid_xs, grid_ys)

fig, ax = plt.subplots()
ax.quiver(grid_xs, grid_ys, grid_us, grid_vs) #, scale=400)

fig.savefig('vector_field.png')

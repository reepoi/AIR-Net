import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.interpolate as interp

FILE_LOCATION = 'cropped_2D_aneurysm/vel_2Daneu_crop.0.csv'
FILE_LOCATION = 'vel2Daneu_crop_recovered_0.9.csv'


def nmae(yhat, y):
    return np.abs(yhat - y).mean() / (np.max(y) - np.min(y))


def aneu_timeframe_df(time):
    FILE_LOCATION = f'cropped_2D_aneurysm/vel_2Daneu_crop.{time}.csv'
    data_pd = pd.read_csv(FILE_LOCATION)
    u_col, v_col, _, x_col, y_col, _ = data_pd.columns
    data_pd = data_pd.set_index([x_col, y_col]) # set index to (x, y)
    return data_pd


def aneu_timeframe(time):
    data_pd = aneu_timeframe_df(time)
    data_pd = data_pd.groupby(level=data_pd.index.names).first() # remove duplicate (x, y)

    (x_col, y_col), (u_col, v_col) = data_pd.index.names, data_pd.columns
    xs = data_pd.index.get_level_values(x_col).to_numpy()
    ys = data_pd.index.get_level_values(y_col).to_numpy()
    us = data_pd[u_col].to_numpy()
    vs = data_pd[v_col].to_numpy()

    return xs, ys, us, vs


def aneu_flattened(time):
    FILE_LOCATION = 'vel2Daneu_crop_recovered_0.6.csv'
    data = pd.read_csv(FILE_LOCATION, header=None).to_numpy()

    data_pd = aneu_timeframe_df(time)
    x_col, y_col = data_pd.index.names
    xs = data_pd.index.get_level_values(x_col).to_numpy()
    ys = data_pd.index.get_level_values(y_col).to_numpy()
    us = data[0::2, time]
    vs = data[1::2, time]

    return xs, ys, us, vs


if __name__ == '__main__':
    time = 0
    func = aneu_flattened
    xs, ys, us, vs = func(time)

    data_pd = aneu_timeframe_df(time)
    u_col, v_col, *_ = data_pd.columns
    nmae_original_us = nmae(us, data_pd[u_col])
    nmae_original_vs = nmae(vs, data_pd[v_col])
    print(f'nmae with original: {nmae_original_us:.5f}')
    print(f'nmae with original: {nmae_original_vs:.5f}')

    fig, ax = plt.subplots()
    ax.quiver(xs, ys, us, vs, scale=400)

    fig.savefig(f'{func.__name__}__{time}.png')

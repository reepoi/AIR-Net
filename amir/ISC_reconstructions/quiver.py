import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def nmae(pre,rel,mask=None):
    if mask is None:
        mask = np.ones(pre.shape)
    def translate_mask(mask):
        u,v = np.where(mask == 1)
        return u,v
    u,v = translate_mask(1-mask)
    return np.abs(pre-rel)[u,v].mean()/(np.max(rel)-np.min(rel))


if __name__ == '__main__':
    mask_rate = 0.9
    grid_density = 300
    ts = np.linspace(-2, 2, num=grid_density)
    xx, yy = np.meshgrid(ts, ts)
    us = np.sin(2 * xx + 2 * yy)
    vs = np.cos(2 * xx - 2 * yy)
    usMask = pd.read_csv(f'{mask_rate}DropMask_us.csv', header=None).to_numpy()
    vsMask = pd.read_csv(f'{mask_rate}DropMask_vs.csv', header=None).to_numpy()
    usRec = pd.read_csv(f'{mask_rate}DropRecovered_us.csv', header=None).to_numpy()
    vsRec = pd.read_csv(f'{mask_rate}DropRecovered_vs.csv', header=None).to_numpy()
    print(f'NMAE (us): {nmae(usRec, us, mask=usMask)}')
    print(f'NMAE (vs): {nmae(vsRec, vs, mask=vsMask)}')

    # fig, ax = plt.subplots()
    # ax.quiver(xx, yy, us, vs)
    # fig.savefig(f'{mask_rate}DropRecovered.png')

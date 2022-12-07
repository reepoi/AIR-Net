from pathlib import Path
import io

import PIL
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import train


def matplotlib_to_PIL_Image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img = PIL.Image.open(buf)
    return img


def quiver(X, Y, U, V, scale=None, save_path=None):
    fig, ax = plt.subplots()

    if scale:
        ax.quiver(X, Y, U, V, scale=scale)
    else:
        ax.quiver(X, Y, U, V)

    if save_path:
        fig.savefig(save_path)

    return fig, ax


if __name__ == '__main__':
    OUTPUT_DIR = Path(__file__).parent / 'output'

    data_set = train.DataSet.ANEURYSM

    algorithm = train.Algorithm.IST

    algorithm_nmae_files = (OUTPUT_DIR / data_set.value / algorithm.value).rglob('nmae.json')
    technique = train.Technique.IDENTITY
    technique_nmae_files = [p for p in algorithm_nmae_files if technique.value in p.parts]
    nmae_jsons = [pd.read_json(p, typ='series') for p in technique_nmae_files]
    mask_rates = [0.3, 0.5, 0.7, 0.9]
    for j, p in zip(nmae_jsons, technique_nmae_files):
        mask_rate = float(str(p).partition('mask_rate')[2].partition('/')[0])
        j['mask_rate'] = mask_rate
    nmae = pd.concat(nmae_jsons)
    print(nmae)

    fig, axes = plt.subplots(nrows=2, ncols=3)
    for a in np.nditer(axes, flags=['refs_ok']):
        a.item().set_xticks(mask_rates)
    axes[0, 0].scatter(nmae['mask_rate'], nmae['velx_by_time'], label=algorithm.value)
    axes[0, 1].scatter(nmae['mask_rate'], nmae['vely_by_time'], label=algorithm.value)
    fig.savefig('test/figure.png')
    plt.close(fig)

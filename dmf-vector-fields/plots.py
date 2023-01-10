import itertools
from pathlib import Path
import io

import PIL
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import pandas as pd
import numpy as np
import train
import data


FONT_SIZE = 14


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


def plot_report_data(ax, data_set, time, algorithm, technique, grid_density, component, num_factors=1):
    assert component in {'velx', 'vely'}
    algorithm_nmae_files = (OUTPUT_DIR / data_set.value / algorithm.value).rglob(f'report_data.{time}.json')
    if algorithm is train.Algorithm.DMF:
        algorithm_nmae_files = [p for p in algorithm_nmae_files if f'num_factors{num_factors}' in p.parts]
    technique_nmae_files = [p for p in algorithm_nmae_files if technique.value in p.parts]
    grid_density_nmae_files = [p for p in technique_nmae_files if f'grid_density{grid_density}' in p.parts]
    files = sorted(grid_density_nmae_files)
    assert len(files) != 0, f'No files found! {data_set.value}, {algorithm.value}, {technique.value}, {grid_density}'
    nmae_jsons = [{'nmae': pd.read_json(p)[component].loc['nmae'][-1]} for p in files]
    mask_rates = [0.3, 0.5, 0.7, 0.9]
    for j, p in zip(nmae_jsons, files):
        mask_rate = float(str(p).partition('mask_rate')[2].partition('/')[0])
        j['mask_rate'] = mask_rate
    nmae = pd.concat([pd.Series(nj) for nj in nmae_jsons])

    ax.set_xticks(mask_rates)
    ax.set_title(f'Grid Density: {grid_density}')
    ax.set_xlabel('Mask Rate')
    ax.set_ylabel('NMAE')
    label = algorithm.value
    if algorithm is train.Algorithm.DMF:
        label = f'{algorithm.value}_{num_factors}'
    ax.scatter(nmae['mask_rate'], nmae['nmae'], label=label)


def plot_nmae(ax, data_set, algorithm, technique, num_factors=1, grid_density=100, num_timeframes=None):
    data_set_name = data_set.value
    if num_timeframes is not None:
        data_set_name = f'{data_set_name}_tf{num_timeframes}'
    algorithm_nmae_files = (OUTPUT_DIR / data_set_name / algorithm.value).rglob('nmae.json')
    if algorithm is train.Algorithm.DMF:
        algorithm_nmae_files = [p for p in algorithm_nmae_files if f'num_factors{num_factors}' in p.parts]
    if technique is train.Technique.INTERPOLATED:
        algorithm_nmae_files = [p for p in algorithm_nmae_files if f'grid_density{grid_density}' in p.parts]
    technique_nmae_files = [p for p in algorithm_nmae_files if technique.value in p.parts]
    files = sorted(technique_nmae_files)
    if len(files) == 0:
        print(f'No files found! {data_set.value}, {algorithm.value}, {technique.value}, {num_factors}, {grid_density}')
        return
    nmae_jsons = [pd.read_json(p, typ='series') for p in files]
    mask_rates = [0.3, 0.5, 0.7, 0.9]
    for j, p in zip(nmae_jsons, files):
        mask_rate = float(str(p).partition('mask_rate')[2].partition('/')[0])
        j['mask_rate'] = mask_rate
    nmae = pd.concat(nmae_jsons)

    ax.set_xticks(mask_rates)
    ax.tick_params(axis='both', labelsize=FONT_SIZE)
    ax.set_xlabel('Mask Rate', fontsize=FONT_SIZE)
    ax.set_ylabel('log10(NMAE)', fontsize=FONT_SIZE)
    label = algorithm.value.upper()
    if algorithm is train.Algorithm.DMF:
        label = f'{label} (F={num_factors})'
    data = lambda c: np.log10(nmae[c].to_numpy() if not isinstance(nmae[c], np.float64) else nmae[c])
    ax.plot(
        nmae['mask_rate'], sum(data(c) for c in ['velx_by_time', 'vely_by_time']),
        marker='o', linestyle='dashed',
        linewidth=1, markersize=7,
        label=label
    )


def vec_field_aneurysm():
    args = dict(data_dir=Path('data'))
    time = 0
    return data.TimeframeAneurysm(time=time, filepath=args['data_dir'] / train.DataSet.ANEURYSM.value / f'vel_2Daneu_crop.{time}.csv').vec_field


def vec_field_double_gyre():
    return data.double_gyre().timeframe(0).vec_field


def plot_vec_field(field: data.VectorField):
    rng = np.random.RandomState(seed=20210909)
    subsample = rng.randint(field.coords.x.size, size=int(np.floor(field.coords.x.size * 0.3)))
    x = field.coords.x[subsample]
    y = field.coords.y[subsample]
    u = field.velx[subsample]
    v = field.vely[subsample]
    colors = np.sqrt(u**2 + v**2)
    u = u / colors
    v = v / colors

    # we need to normalize our colors array to match it colormap domain
    # which is [0, 1]
    norm = Normalize()
    norm.autoscale(colors)

    colormap = cm.viridis
    # pick your colormap here, refer to 
    # http://matplotlib.org/examples/color/colormaps_reference.html
    # and
    # http://matplotlib.org/users/colormaps.html
    # for details
    fig, ax = plt.subplots()
    ax.quiver(x, y, u, v, color=colormap(norm(colors)), angles='xy', scale=45, pivot='mid')
    fig.savefig('test.pdf', format='pdf', bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    plot_vec_field(vec_field_aneurysm())
    plot_vec_field(vec_field_double_gyre())
    OUTPUT_DIR = Path(__file__).parent / '..' / 'out' / 'output'
    image_format = 'pdf'
    kwargs = dict(grid_density=100, num_timeframes=None)
    for t, d in itertools.product(train.Technique, [train.DataSet.ANEURYSM, train.DataSet.DOUBLE_GYRE]):
        fig, ax = plt.subplots()
        plot_nmae(ax, d, train.Algorithm.IST, t, **kwargs)
        for nf in [2, 3, 4, 5]:
            plot_nmae(ax, d, train.Algorithm.DMF, t, num_factors=nf, **kwargs)
        plot_name = f'figures/{d.value}_{t.value}'
        if kwargs['num_timeframes'] is not None:
            plot_name = f'{plot_name}_tf{kwargs["num_timeframes"]}'
        if t is train.Technique.INTERPOLATED:
            plot_name = f'{plot_name}_gd{kwargs["grid_density"]}'
        ax.legend(fontsize=FONT_SIZE)
        fig.savefig(f'{plot_name}.{image_format}', format=image_format, bbox_inches='tight')
        plt.close(fig)

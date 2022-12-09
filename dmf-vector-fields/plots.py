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


def plot_nmae(ax, data_set, algorithm, technique, component, num_factors=1):
    algorithm_nmae_files = (OUTPUT_DIR / data_set.value / algorithm.value).rglob('nmae.json')
    if algorithm is train.Algorithm.DMF:
        algorithm_nmae_files = [p for p in algorithm_nmae_files if f'num_factors{num_factors}' in p.parts]
    technique_nmae_files = [p for p in algorithm_nmae_files if technique.value in p.parts]
    files = sorted(technique_nmae_files)
    assert len(files) != 0, 'No files found!'
    nmae_jsons = [pd.read_json(p, typ='series') for p in files]
    mask_rates = [0.3, 0.5, 0.7, 0.9]
    for j, p in zip(nmae_jsons, files):
        mask_rate = float(str(p).partition('mask_rate')[2].partition('/')[0])
        j['mask_rate'] = mask_rate
    nmae = pd.concat(nmae_jsons)

    ax.set_xticks(mask_rates)
    ax.set_xlabel('Mask Rate')
    ax.set_ylabel('NMAE')
    label = algorithm.value
    if algorithm is train.Algorithm.DMF:
        label = f'{algorithm.value}_{num_factors}'
    ax.scatter(nmae['mask_rate'], nmae[component], label=label)


if __name__ == '__main__':
    OUTPUT_DIR = Path(__file__).parent / '..' / 'out' / 'output'

    data_set = train.DataSet.FUNC1

    technique = train.Technique.IDENTITY

    # fig, axes = plt.subplots(nrows=2, ncols=3)
    # for a in np.nditer(axes, flags=['refs_ok']):
    #     a.item().set_xticks(mask_rates)

    fig, ax = plt.subplots()
    # component = 'velx_by_time'
    # plot_nmae(ax, data_set, train.Algorithm.IST, technique, component)
    # plot_nmae(ax, data_set, train.Algorithm.DMF, technique, component, num_factors=2)
    # plot_nmae(ax, data_set, train.Algorithm.DMF, technique, component, num_factors=3)
    # plot_nmae(ax, data_set, train.Algorithm.DMF, technique, component, num_factors=4)
    # plot_nmae(ax, data_set, train.Algorithm.DMF, technique, component, num_factors=5)
    component = 'velx'
    time = 0
    grid_density = 100
    technique = train.Technique.INTERPOLATED
    plot_report_data(ax, data_set, time, train.Algorithm.IST, technique, grid_density, component)
    plot_report_data(ax, data_set, time, train.Algorithm.DMF, technique, grid_density, component, num_factors=2)
    plot_report_data(ax, data_set, time, train.Algorithm.DMF, technique, grid_density, component, num_factors=3)
    # plot_report_data(ax, data_set, time, train.Algorithm.DMF, technique, grid_density, component, num_factors=4)
    # plot_report_data(ax, data_set, time, train.Algorithm.DMF, technique, grid_density, component, num_factors=5)
    ax.legend()
    fig.savefig(f'figures/{data_set.value}_{technique.value}_timeframe_{component}.png')
    plt.close(fig)

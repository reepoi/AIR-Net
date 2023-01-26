import itertools
from pathlib import Path
import io

import PIL
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import pandas as pd
import numpy as np
from dmf_vector_fields import enums, model


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
    if algorithm is enums.Algorithm.DMF:
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
    if algorithm is enums.Algorithm.DMF:
        label = f'{algorithm.value}_{num_factors}'
    ax.scatter(nmae['mask_rate'], nmae['nmae'], label=label)


def plot_nmae(ax, data_set, algorithm, technique, num_factors=1, grid_density=100, num_timeframes=None):
    data_set_name = data_set.value
    if num_timeframes is not None:
        data_set_name = f'{data_set_name}_tf{num_timeframes}'
    algorithm_nmae_files = (OUTPUT_DIR / data_set_name / algorithm.value).rglob('nmae.json')
    if algorithm is enums.Algorithm.DMF:
        algorithm_nmae_files = [p for p in algorithm_nmae_files if f'num_factors{num_factors}' in p.parts]
    if technique is enums.Technique.INTERPOLATED:
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
    if algorithm is enums.Algorithm.DMF:
        label = f'{label} (F={num_factors})'
    data = lambda c: np.log10(nmae[c].to_numpy() if not isinstance(nmae[c], np.float32) else nmae[c])
    ax.plot(
        nmae['mask_rate'], sum(data(c) for c in ['velx_by_time', 'vely_by_time']),
        marker='o', linestyle='dashed',
        linewidth=1, markersize=7,
        label=label
    )


def plot_reconstructed_rank(csv_filename, data_set, technique, num_factors=1, grid_density=100, num_timeframes=None):
    data_set_name = data_set.value
    if num_timeframes is not None:
        data_set_name = f'{data_set_name}_tf{num_timeframes}'
    files = (OUTPUT_DIR / data_set_name).rglob(csv_filename)
    files = [p for p in files if technique.value in p.parts]
    if len(files) == 0:
        print(f'No files found! {data_set.value}, {technique.value}')
        return
    matrices = [pd.read_csv(p, header=None).to_numpy() for p in files]
    data = []
    for m, p in zip(matrices, files):
        algorithm = enums.Algorithm.IST if enums.Algorithm.IST.value in str(p) else enums.Algorithm.DMF
        mask_rate = float(str(p).partition('mask_rate')[2].partition('/')[0])
        num_factors = ''
        if algorithm is enums.Algorithm.DMF:
            num_factors = f' (F={str(p).partition("num_factors")[2].partition("/")[0]})'
        data.append(dict(
            mask_rate=mask_rate,
            label=f'{algorithm.value.upper()}{num_factors}',
            rank=np.linalg.matrix_rank(m)
        ))

    plot_data = dict()
    for d in data:
        label = d['label']
        if label not in plot_data:
            plot_data[label] = dict(mask_rates=[], ranks=[])
        plot_data[label]['mask_rates'].append(d['mask_rate'])
        plot_data[label]['ranks'].append(d['rank'])

    mask_rates = [0.3, 0.5, 0.7, 0.9]
    fig, ax = plt.subplots()
    ax.set_xticks(mask_rates)
    ax.tick_params(axis='both', labelsize=FONT_SIZE)
    ax.set_xlabel('Mask Rate', fontsize=FONT_SIZE)
    ax.set_ylabel('Matrix Rank', fontsize=FONT_SIZE)
    for label, p in plot_data.items():
        ax.plot(
            p['mask_rates'], p['ranks'],
            marker='o', linestyle='dashed',
            linewidth=1, markersize=7,
            label=label
        )
    ax.legend(fontsize=FONT_SIZE)
    fig.savefig(f'rank_{data_set.value}_{technique.value}_{csv_filename}.pdf', format='pdf', bbox_inches='tight')
    plt.close(fig)


def vel_by_time_aneurysm():
    args = dict(data_dir=Path('data'))
    time = 0
    tf = data.TimeframeAneurysm(time=time, filepath=args['data_dir'] / enums.DataSet.ANEURYSM.value / f'vel_2Daneu_crop.{time}.csv')
    return data.VelocityByTimeAneurysm(
        coords=tf.vec_field.coords,
        filepath_vel_by_time=args['data_dir'] / enums.DataSet.ANEURYSM.value / 'vel_by_time_2Daneu_crop.csv',
    )


def plot_heatmap(name, vec_field):
    arr = vec_field.ravel().vel_axes[0]
    length = int(arr.size**(1/2))
    assert length**2 == arr.size, 'Can only plot a square.'
    arr = arr.reshape(length, -1)

    fig, ax = plt.subplots()
    ax.imshow(arr)
    fig.savefig(f'{name}.pdf', format='pdf', bbox_inches='tight')
    plt.close(fig)


def plot_hist2d(name, vec_field):
    fig, ax = plt.subplots()
    x, y = vec_field.coords.axes
    z = vec_field.vel_axes[0]
    ax.pcolormesh(x, y, z)
    fig.savefig(f'{name}.pdf', format='pdf', bbox_inches='tight')
    plt.close(fig)


def plot_vec_field(name, vec_field, scale=45, subsample=0.3):
    if len(vec_field.components) == 1:
        plot_heatmap(name, vec_field)
        return
    assert len(vec_field.components) == 2, 'Can only plot 2D vector fields'
    vec_field = vec_field.ravel()
    x, y = vec_field.coords.axes
    velx, vely = vec_field.vel_axes
    rng = np.random.RandomState(seed=20210909)
    subsample = rng.randint(x.size, size=int(np.floor(y.size * subsample)))
    x = x[subsample]
    y = y[subsample]
    u = velx[subsample]
    v = vely[subsample]
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
    ax.quiver(x, y, u, v, color=colormap(norm(colors)), angles='xy', scale=scale, pivot='mid')
    fig.savefig(f'{name}.pdf', format='pdf', bbox_inches='tight')
    plt.close(fig)


def rank_over_vel_by_time_columns(name, vel_by_time, masked=False, mask_rate=0.9):
    if masked:
        mask = model.get_bit_mask(vel_by_time.shape_as_completable(interleaved=False), mask_rate)
        vel_by_time = vel_by_time.transform(lambda vel: vel * mask, interleaved=False)
    times = range(vel_by_time.timeframes)
    ranks_velx = [np.linalg.matrix_rank(vel_by_time.velx_by_time[:, :t+1]) for t in times]
    ranks_vely = [np.linalg.matrix_rank(vel_by_time.vely_by_time[:, :t+1]) for t in times]
    interleaved = vel_by_time.completable_matrices()
    ranks_interleaved = [np.linalg.matrix_rank(interleaved[:, :t+1]) for t in times]
    fig, ax = plt.subplots()
    ax.plot(times, ranks_velx, label='velx')
    ax.plot(times, ranks_vely, label='vely')
    ax.plot(times, ranks_interleaved, label='intrlvd')
    ax.legend(fontsize=FONT_SIZE)
    fig.savefig(f'{name}.pdf', format='pdf', bbox_inches='tight')
    plt.close(fig)


def rank_over_timeframes(name, vel_by_time, masked=False, mask_rate=0.9):
    if masked:
        mask = model.get_bit_mask(vel_by_time.shape_as_completable(interleaved=False), mask_rate)
        vel_by_time = vel_by_time.transform(lambda vel: vel * mask, interleaved=False)
    times = range(vel_by_time.timeframes)
    tf_vec_fields = [vel_by_time.timeframe(t).vec_field for t in times]
    ranks_velx = [np.linalg.matrix_rank(vf.velx) for vf in tf_vec_fields]
    ranks_vely = [np.linalg.matrix_rank(vf.vely) for vf in tf_vec_fields]
    fig, ax = plt.subplots()
    ax.plot(times, ranks_velx, label='velx')
    ax.plot(times, ranks_vely, label='vely')
    ax.legend(fontsize=FONT_SIZE)
    fig.savefig(f'{name}.pdf', format='pdf', bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    # aneurysm = vel_by_time_aneurysm()
    # double_gyre = data.double_gyre(num_timeframes=22)
    # plot_vec_field(f'vf_{enums.DataSet.ANEURYSM.value}', aneurysm.timeframe(0).vec_field)
    # plot_vec_field(f'vf_{enums.DataSet.DOUBLE_GYRE.value}', double_gyre.timeframe(0).vec_field)
    # plot_vec_field(f'vf_all_{enums.DataSet.ANEURYSM.value}', aneurysm.timeframe(0).vec_field, scale=100, subsample=1)
    # plot_vec_field(f'vf_all_{enums.DataSet.DOUBLE_GYRE.value}', double_gyre.timeframe(0).vec_field, scale=100, subsample=1)
    # rank_over_vel_by_time_columns(f'rank_{enums.DataSet.ANEURYSM.value}', aneurysm)
    # rank_over_vel_by_time_columns(f'rank_{enums.DataSet.DOUBLE_GYRE.value}', double_gyre)
    # rank_over_timeframes(f'rank_tf_{enums.DataSet.ANEURYSM.value}', aneurysm)
    # rank_over_timeframes(f'rank_tf_{enums.DataSet.DOUBLE_GYRE.value}', double_gyre)
    # mask_rate = 0.9
    # rank_over_vel_by_time_columns(f'rank_masked{mask_rate}_{enums.DataSet.ANEURYSM.value}', aneurysm, masked=True, mask_rate=mask_rate)
    # rank_over_vel_by_time_columns(f'rank_masked{mask_rate}_{enums.DataSet.DOUBLE_GYRE.value}', double_gyre, masked=True, mask_rate=mask_rate)
    # rank_over_timeframes(f'rank_tf_masked{mask_rate}_{enums.DataSet.ANEURYSM.value}', aneurysm, masked=True, mask_rate=mask_rate)
    # rank_over_timeframes(f'rank_tf_masked{mask_rate}_{enums.DataSet.DOUBLE_GYRE.value}', double_gyre, masked=True, mask_rate=mask_rate)
    OUTPUT_DIR = Path(__file__).parent / '..' / 'out' / 'output'
    # image_format = 'pdf'
    # kwargs = dict(grid_density=100, num_timeframes=None)
    for t, d in itertools.product(enums.Technique, [enums.DataSet.ANEURYSM, enums.DataSet.DOUBLE_GYRE]):
        plot_reconstructed_rank('reconstructed_velx_by_time.csv', d, t, num_factors=1, grid_density=100, num_timeframes=None)
        plot_reconstructed_rank('reconstructed_vely_by_time.csv', d, t, num_factors=1, grid_density=100, num_timeframes=None)
    # for t, d in itertools.product(enums.Technique, [enums.DataSet.ANEURYSM, enums.DataSet.DOUBLE_GYRE]):
    #     fig, ax = plt.subplots()
    #     plot_nmae(ax, d, enums.Algorithm.IST, t, **kwargs)
    #     for nf in [2, 3, 4, 5]:
    #         plot_nmae(ax, d, enums.Algorithm.DMF, t, num_factors=nf, **kwargs)
    #     plot_name = f'figures/{d.value}_{t.value}'
    #     if kwargs['num_timeframes'] is not None:
    #         plot_name = f'{plot_name}_tf{kwargs["num_timeframes"]}'
    #     if t is enums.Technique.INTERPOLATED:
    #         plot_name = f'{plot_name}_gd{kwargs["grid_density"]}'
    #     ax.legend(fontsize=FONT_SIZE)
    #     fig.savefig(f'{plot_name}.{image_format}', format=image_format, bbox_inches='tight')
    #     plt.close(fig)

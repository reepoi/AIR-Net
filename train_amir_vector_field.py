import argparse
import json
import numpy as np
import pandas as pd
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import torch
import main
from MinPy import reg
import wandb
import io
import pdb
import PIL


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_argparser():
    parser = argparse.ArgumentParser(description='Run paper\'s implementation of AIR-Net.')
    parser.add_argument('--weight-decay-total-variation', type=float,
                        help='coefficients of the total variation regularizers.')
    parser.add_argument('--weight-decay-dirichlet-energy', type=float,
                        help='coefficients of the dirichlet energy regularizers.')
    parser.add_argument('--num-factors', type=int,
                        help='number minus one of matrix factors with dimension rows x rows.')
    parser.add_argument('--save-dir', type=str,
                        help='where to save the figures.')
    parser.add_argument('--epochs', type=int,
                        help='number of epochs.')
    parser.add_argument('--mask-rate', type=float, default=None,
                        help='the expected precentage of matrix entries to be hidden.')
    parser.add_argument('--aneu-path', type=str,
                        help='path to aneurysm data.')
    return parser


def matplotlib_to_PIL_Image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img = PIL.Image.open(buf)
    return img


def interp_griddata(spatial_coordinates, func_values, new_points, **kwargs):
    return interp.griddata(spatial_coordinates, func_values, new_points, method='linear', **kwargs)


def load_aneu_timeframe(path, grid_density):
    data_pd = pd.read_csv(path)
    u_col, v_col, w_col, x_col, y_col, z_col = data_pd.columns
    data_pd = data_pd[(c for c in data_pd.columns if c not in {w_col, z_col})] # drop extra columns
    data_pd = data_pd.set_index([x_col, y_col]) # set index to (x, y)
    data_pd = data_pd.groupby(level=data_pd.index.names).first() # remove duplicate (x, y)

    xs = data_pd.index.get_level_values(x_col).to_numpy()
    ys = data_pd.index.get_level_values(y_col).to_numpy()
    us = data_pd[u_col].to_numpy()
    vs = data_pd[v_col].to_numpy()

    grid_xs, grid_ys = np.meshgrid(
        np.linspace(np.min(xs), np.max(xs), num=grid_density),
        np.linspace(np.min(ys), np.max(ys), num=grid_density)
    )
    grid_us = interp_griddata((xs, ys), us, (grid_xs, grid_ys), fill_value=0)
    grid_vs = interp_griddata((xs, ys), vs, (grid_xs, grid_ys), fill_value=0)

    return {'data_pd': data_pd, 'grid_data': [grid_xs, grid_ys, grid_us, grid_vs]}


def interpolate_vector_field_points(data_pd, grid_points):
    grid_xs, grid_ys, grid_us, grid_vs = grid_points
    (x_col, y_col), (u_col, v_col) = data_pd.index.names, data_pd.columns
    data_xs = data_pd.index.get_level_values(x_col).to_numpy()
    data_ys = data_pd.index.get_level_values(y_col).to_numpy()
    spatial_coordinates, func_values  = (
        (grid_xs.ravel(), grid_ys.ravel()),
        (grid_us.ravel(), grid_vs.ravel())
    )
    data_us = interp_griddata(spatial_coordinates, func_values[0], (data_xs, data_ys))
    data_vs = interp_griddata(spatial_coordinates, func_values[1], (data_xs, data_ys))
    return data_xs, data_ys, data_us, data_vs


def num_large_singular_values(matrix, threshold=5e-1):
    u, s, vh = np.linalg.svd(matrix, full_matrices=False)
    return np.sum(np.where(s > threshold, 1, 0))


def do(mask_rate, args):
    grid_density = 200
    data = load_aneu_timeframe(args.aneu_path, grid_density)
    data_pd = data['data_pd']
    (x_col, y_col), (u_col, v_col) = data_pd.index.names, data_pd.columns
    grid_xs, grid_ys, grid_us, grid_vs = data['grid_data']
    matrix = grid_us

    print(f'grid_us Rank: {num_large_singular_values(grid_us)}')
    print(f'grid_vs Rank: {num_large_singular_values(grid_us)}')

    rows, cols = matrix.shape
    matrix_factor_dimensions = [main.Shape(rows=rows, cols=rows) for _ in range(args.num_factors - 1)]
    matrix_factor_dimensions.append(main.Shape(rows=rows, cols=cols))

    wandb.config = {
        'dataset': 'Amir Interpolated',
        'mask_rate': mask_rate,
        'epochs': args.epochs,
        'weight_decay_total_variation_row': args.weight_decay_total_variation,
        'weight_decay_total_variation_col': args.weight_decay_total_variation,
        'weight_decay_dirichlet_energy_row': args.weight_decay_dirichlet_energy,
        'weight_decay_dirichlet_energy_col': args.weight_decay_dirichlet_energy,
        'matrix_factor_dimensions': [(s.rows, s.cols) for s in matrix_factor_dimensions]
    }
    wandb.init(mode="disabled", project="AIR-Net Vector Fields", entity="taost", config=wandb.config)

    mask = main.get_bit_mask(matrix, rate=mask_rate)
    mask_numpy = mask.cpu().numpy()

    fig, ax = main.plot.quiver(grid_xs, grid_ys, grid_us, grid_vs, scale=400)
    fig.savefig(f'{args.save_dir}/AmirInterp.png')

    fig, ax = main.plot.quiver(grid_xs, grid_ys, grid_us * mask_numpy, grid_vs * mask_numpy, scale=400)
    fig.savefig(f'{args.save_dir}/{mask_rate}DropAmir.png')

    wandb.log({'masked': wandb.Image(matplotlib_to_PIL_Image(fig))})
    print(matrix_factor_dimensions)

    def meets_stop_criteria(epoch, loss):
        return loss < 1e-5

    report_frequency = 500
    def report(reconstructed_matrix, epoch, loss, column):
        data_xs = data_pd.index.get_level_values(x_col).to_numpy()
        data_ys = data_pd.index.get_level_values(y_col).to_numpy()
        data_func = interp_griddata((grid_xs.ravel(), grid_ys.ravel()), reconstructed_matrix.ravel(), (data_xs, data_ys))
        nmae_against_original = main.lossm.my_nmae(data_func, data_pd[column].to_numpy())
        print(f'Column: {column}, Epoch: {epoch}, MSE+REG: {loss:.5f}, NMAE (Original): {nmae_against_original:.5f}')
    
    print(f'Mask Rate: {mask_rate}')
    print('Train grid_us')
    RCgrid_us = main.run_test(args.epochs, matrix_factor_dimensions, torch.tensor(grid_us).to(device), mask,
                              meets_stop_criteria=meets_stop_criteria,
                              report_frequency=report_frequency, report=lambda *args: report(*args, column=u_col))

    print('Train grid_vs')
    RCgrid_vs = main.run_test(args.epochs, matrix_factor_dimensions, torch.tensor(grid_vs).to(device), mask,
                              meets_stop_criteria=meets_stop_criteria,
                              report_frequency=report_frequency, report=lambda *args: report(*args, column=v_col))

    data_xs, data_ys, data_us, data_vs = interpolate_vector_field_points(data_pd, [grid_xs, grid_ys, RCgrid_us, RCgrid_vs])

    fig, ax = main.plot.quiver(data_xs, data_ys, data_us, data_vs, scale=400)
    fig.savefig(f'{args.save_dir}/{mask_rate}Recovered.png')
    wandb.log({'recovered': wandb.Image(matplotlib_to_PIL_Image(fig))})

    wandb.finish()


if __name__ == '__main__':
    mask_rates = [0.3, 0.5, 0.7, 0.9]

    args = get_argparser().parse_args()
    if (mr := args.mask_rate):
        mask_rates = [mr]

    for mr in mask_rates:
        do(mr, args)

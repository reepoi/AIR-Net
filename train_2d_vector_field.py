import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import main
import wandb
import io
import PIL


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_argparser():
    parser = argparse.ArgumentParser(description='Run paper\'s implementation of AIR-Net.')
    parser.add_argument('--weight-decay', type=float,
                        help='coefficients of the regularizers.')
    parser.add_argument('--num-factors', type=int,
                        help='number minus one of matrix factors with dimension rows x rows.')
    parser.add_argument('--save-dir', type=str,
                        help='where to save the figures.')
    parser.add_argument('--epochs', type=int,
                        help='number of epochs.')
    parser.add_argument('--grid-density', type=int,
                        help='density of the grid to plot the vector field on.')
    return parser


def matplotlib_to_PIL_Image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img = PIL.Image.open(buf)
    return img


def do(mask_rate, epochs, weight_decay, num_factors, grid_density, save_dir):
    ts = np.linspace(-2, 2, num=grid_density)
    xx, yy = np.meshgrid(ts, ts)
    us = torch.tensor(np.sin(2 * xx + 2 * yy)).to(device)
    vs = torch.tensor(np.cos(2 * xx - 2 * yy)).to(device)

    rows, cols = us.shape
    matrix_factor_dimensions = [main.Shape(rows=rows, cols=rows) for _ in range(num_factors - 1)]
    matrix_factor_dimensions.append(main.Shape(rows=rows, cols=cols))

    wandb.config = {
        'dataset': 'F = (sin(2x + 2y), cos(2x - 2y))',
        'grid_density': grid_density,
        'mask_rate': mask_rate,
        'epochs': epochs,
        'weight_decay_dirichlet_energy_row': weight_decay,
        'weight_decay_dirichlet_energy_col': weight_decay,
        'matrix_factor_dimensions': [(s.rows, s.cols) for s in matrix_factor_dimensions]
    }
    wandb.init(project="AIR-Net Vector Fields", entity="taost", config=wandb.config)

    mask = main.get_bit_mask(us, rate=mask_rate)

    fig, ax = main.plot.quiver(xx, yy, (us * mask).cpu(), (vs * mask).cpu())
    wandb.log({'masked': wandb.Image(matplotlib_to_PIL_Image(fig))})
    print(matrix_factor_dimensions)

    print('Train us')
    RCus, us_losses = main.run_paper_test(epochs, matrix_factor_dimensions, us, mask, regularizers=[
        main.paper_regularization(weight_decay, rows, 'row'),
        main.paper_regularization(weight_decay, cols, 'col'),
    ], loss_log_suffix='x-component')
    print('\n')
    print('Train vs')
    RCvs, vs_losses = main.run_paper_test(epochs, matrix_factor_dimensions, vs, mask, regularizers=[
        main.paper_regularization(weight_decay, rows, 'row'),
        main.paper_regularization(weight_decay, cols, 'col'),
    ], loss_log_suffix='y-component')

    fig, ax = main.plot.quiver(xx, yy, RCus.detach().cpu(), RCvs.detach().cpu())
    wandb.log({'recovered': wandb.Image(matplotlib_to_PIL_Image(fig))})

    wandb.finish()


if __name__ == '__main__':
    mask_rates = [0.3, 0.5, 0.7, 0.9]
    args = get_argparser().parse_args()
    for mr in mask_rates:
        do(mr, **args.__dict__)

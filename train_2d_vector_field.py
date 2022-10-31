import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import main
from MinPy import reg
import wandb
import io
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
    parser.add_argument('--grid-density', type=int,
                        help='density of the grid to plot the vector field on.')
    return parser


def matplotlib_to_PIL_Image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img = PIL.Image.open(buf)
    return img


def do(mask_rate, epochs, weight_decay_total_variation, weight_decay_dirichlet_energy,
       num_factors, grid_density, save_dir):
    ts = torch.linspace(-2, 2, grid_density).to(device)
    xx, yy = torch.meshgrid(ts, ts)
    us = torch.sin(2 * xx + 2 * yy).to(device)
    vs = torch.cos(2 * xx - 2 * yy).to(device)
    points = torch.hstack((xx.reshape(-1, 1), yy.reshape(-1, 1))).to(device)

    rows, cols = us.shape
    matrix_factor_dimensions = [main.Shape(rows=rows, cols=rows) for _ in range(num_factors - 1)]
    matrix_factor_dimensions.append(main.Shape(rows=rows, cols=cols))

    wandb.config = {
        'dataset': 'F = (sin(2x + 2y), cos(2x - 2y))',
        'grid_density': grid_density,
        'mask_rate': mask_rate,
        'epochs': epochs,
        'weight_decay_total_variation_row': weight_decay_total_variation,
        'weight_decay_total_variation_col': weight_decay_total_variation,
        'weight_decay_dirichlet_energy_row': weight_decay_dirichlet_energy,
        'weight_decay_dirichlet_energy_col': weight_decay_dirichlet_energy,
        'matrix_factor_dimensions': [(s.rows, s.cols) for s in matrix_factor_dimensions]
    }
    # wandb.init(project="AIR-Net Vector Fields", entity="taost", config=wandb.config)
    wandb.init(mode="disabled", project="AIR-Net Vector Fields", entity="taost", config=wandb.config)

    mask = main.get_bit_mask(us, rate=mask_rate)

    fig, ax = main.plot.quiver(xx.cpu(), yy.cpu(), (us * mask).cpu(), (vs * mask).cpu())
    fig.savefig(f'{save_dir}/{mask_rate}Drop.png')
    wandb.log({'masked': wandb.Image(matplotlib_to_PIL_Image(fig))})
    print(matrix_factor_dimensions)

    # print(f'Mask Rate: {mask_rate}')
    # print('Train us')
    # RCus, us_losses = main.run_paper_test(epochs, matrix_factor_dimensions, us, mask, regularizers=[
    #     main.paper_regularization(weight_decay_dirichlet_energy, rows, 'row'),
    #     main.paper_regularization(weight_decay_dirichlet_energy, cols, 'col'),
    # ], loss_log_suffix='x-component')
    # print(f'us nmae: {us_losses[-1]:.5f}')
    # print('Train vs')
    # RCvs, vs_losses = main.run_paper_test(epochs, matrix_factor_dimensions, vs, mask, regularizers=[
    #     main.paper_regularization(weight_decay_dirichlet_energy, rows, 'row'),
    #     main.paper_regularization(weight_decay_dirichlet_energy, cols, 'col'),
    # ], loss_log_suffix='y-component')
    # print(f'vs nmae: {vs_losses[-1]:.5f}')

    print(f'Mask Rate: {mask_rate}')
    print('Train us')
    RCus, us_losses = main.run_test(epochs, matrix_factor_dimensions, us, mask, regularizers=[
        main.dirichlet_energy_regularization(weight_decay_dirichlet_energy, rows, reg.DirichletEnergyRegularizationMode.ROW_SIMILARITY),
        main.dirichlet_energy_regularization(weight_decay_dirichlet_energy, cols, reg.DirichletEnergyRegularizationMode.COL_SIMILARITY),
        main.total_variation_regularization(weight_decay_total_variation, rows, reg.TotalVariationRegularizationMode.ROW_VARIATION),
        main.total_variation_regularization(weight_decay_total_variation, cols, reg.TotalVariationRegularizationMode.COL_VARIATION),
    ])
    print(f'us nmae: {us_losses[-1]:.5f}')
    print('Train vs')
    RCvs, vs_losses = main.run_test(epochs, matrix_factor_dimensions, vs, mask, regularizers=[
        main.dirichlet_energy_regularization(weight_decay_dirichlet_energy, rows, reg.DirichletEnergyRegularizationMode.ROW_SIMILARITY),
        main.dirichlet_energy_regularization(weight_decay_dirichlet_energy, cols, reg.DirichletEnergyRegularizationMode.COL_SIMILARITY),
        main.total_variation_regularization(weight_decay_total_variation, rows, reg.TotalVariationRegularizationMode.ROW_VARIATION),
        main.total_variation_regularization(weight_decay_total_variation, cols, reg.TotalVariationRegularizationMode.COL_VARIATION),
    ])
    print(f'vs nmae: {vs_losses[-1]:.5f}')

    fig, ax = main.plot.quiver(xx.cpu(), yy.cpu(), RCus.detach().cpu(), RCvs.detach().cpu())
    fig.savefig(f'{save_dir}/{mask_rate}Recovered.png')
    wandb.log({'recovered': wandb.Image(matplotlib_to_PIL_Image(fig))})

    wandb.finish()


if __name__ == '__main__':
    mask_rates = [0.3, 0.5, 0.7, 0.9]
    mask_rates = [0.9]
    args = get_argparser().parse_args()
    for mr in mask_rates:
        do(mr, **args.__dict__)

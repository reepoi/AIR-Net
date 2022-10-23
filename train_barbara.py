import argparse
import json
import numpy as np
import torch
import main
import wandb


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
                        help='number of epochs')
    return parser


def do(mask_rate, epochs, weight_decay, num_factors, save_dir):
    matrix = main.dataloader.get_data(height=240,width=240,pic_name='./train_pics/Barbara.jpg').to(device)

    rows, cols = matrix.shape
    matrix_factor_dimensions = [main.Shape(rows=rows, cols=rows) for _ in range(num_factors - 1)]
    matrix_factor_dimensions.append(main.Shape(rows=rows, cols=cols))

    wandb.config = {
        'dataset': 'Barbara.jpg',
        'mask_rate': mask_rate,
        'epochs': epochs,
        'weight_decay_dirichlet_energy_row': weight_decay,
        'weight_decay_dirichlet_energy_col': weight_decay,
        'matrix_factor_dimensions': [(s.rows, s.cols) for s in matrix_factor_dimensions]
    }
    wandb.init(project="AIR-Net Vector Fields", entity="taost", config=wandb.config)

    mask = main.get_bit_mask(matrix, rate=mask_rate)

    wandb.log({'masked': wandb.Image((matrix * mask).cpu(), caption='Masked Matrix')})

    RCmatrix, PaperDMFAIR_losses = main.run_paper_test(epochs, matrix_factor_dimensions, matrix, mask, regularizers=[
        main.paper_regularization(weight_decay, rows, 'row'),
        main.paper_regularization(weight_decay, cols, 'col')
    ])

    wandb.log({'recovered': wandb.Image(RCmatrix.detach().cpu(), caption='Recovered Matrix')})

    wandb.finish()

if __name__ == '__main__':
    mask_rates = [0.3, 0.5, 0.7, 0.9]
    args = get_argparser().parse_args()
    for mr in mask_rates:
        do(mr, **args.__dict__)

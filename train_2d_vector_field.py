import argparse
import json
import numpy as np
import torch
import main


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_argparser():
    parser = argparse.ArgumentParser(description='Run paper\'s implementation of AIR-Net.')
    parser.add_argument('--weight-decay', type=float,
                        help='coefficients of the regularizers.')
    parser.add_argument('--num-factors', type=int,
                        help='number minus one of matrix factors with dimension rows x rows.')
    parser.add_argument('--save-dir', type=str,
                        help='where to save the figures.')
    return parser


def do(mask_rates, weight_decay, num_factors, path):
    ts = np.linspace(-2, 2, num=300)
    xx, yy = np.meshgrid(ts, ts)
    us = sin(2 * xx + 2 * yy)
    vs = cos(2 * xx - 2 * yy)
    for mask_rate in mask_rates:

        mask = main.get_bit_mask(matrix, rate=mask_rate)

        main.plot.gray_im((matrix * mask).cpu(), save_if=True, path=f'{path}/{mask_rate}Drop.png')

        rows, cols = matrix.shape
        matrix_factor_dimensions = [main.Shape(rows=rows, cols=rows) for _ in range(num_factors - 1)]
        matrix_factor_dimensions.append(main.Shape(rows=rows, cols=cols))

        epochs = 20000
        RCMatrix_PaperDMFAIR, PaperDMFAIR_losses = main.run_paper_test(epochs, matrix_factor_dimensions, matrix, mask, regularizers=[
            main.paper_regularization(weight_decay, rows, 'row'),
            main.paper_regularization(weight_decay, cols, 'col')
        ])
        main.plot.gray_im(RCMatrix_PaperDMFAIR.detach().cpu(), save_if=True, path=f'{path}/{mask_rate}DropRecovered.png')
        RCMatrices = {
            'PaperDMFAIR': (PaperDMFAIR_losses, RCMatrix_PaperDMFAIR)
        }
        losses = {n: t[0] for n, t in RCMatrices.items()}
        losses['x_plot'] = np.arange(0, epochs, 1)
        main.plot.lines(losses, save_if=True, path=f'{path}/{mask_rate}DropNMAE.png', black_if=True, ylabel_name='NMAE')

if __name__ == '__main__':
    args = get_argparser().parse_args()
    with open(f'{args.save_dir}/args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    do([0.3, 0.5, 0.7, 0.9], args.weight_decay, args.num_factors, args.save_dir)
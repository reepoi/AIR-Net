import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
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
    parser.add_argument('--epochs', type=int,
                        help='number of epochs.')
    parser.add_argument('--grid-density', type=int,
                        help='density of the grid to plot the vector field on.')
    return parser


def do(mask_rates, epochs, weight_decay, num_factors, grid_density, path):
    ts = np.linspace(-2, 2, num=grid_density)
    xx, yy = np.meshgrid(ts, ts)
    us = torch.tensor(np.sin(2 * xx + 2 * yy)).to(device)
    vs = torch.tensor(np.cos(2 * xx - 2 * yy)).to(device)
    main.plot.quiver(xx, yy, us.cpu(), vs.cpu(), save_if=True, path=f'{path}/vectorfield.png')
    for mask_rate in mask_rates:
        mask = main.get_bit_mask(us, rate=mask_rate)

        main.plot.quiver(xx, yy, (us * mask).cpu(), (vs * mask).cpu(), save_if=True, path=f'{path}/{mask_rate}Drop.png')

        rows, cols = us.shape
        matrix_factor_dimensions = [main.Shape(rows=rows, cols=rows) for _ in range(num_factors - 1)]
        matrix_factor_dimensions.append(main.Shape(rows=rows, cols=cols))
        print(matrix_factor_dimensions)

        print('Train us')
        RCus, us_losses = main.run_paper_test(epochs, matrix_factor_dimensions, us, mask, regularizers=[
            main.paper_regularization(weight_decay, rows, 'row'),
            main.paper_regularization(weight_decay, cols, 'col')
        ])
        print('\n')
        print('Train vs')
        RCvs, vs_losses = main.run_paper_test(epochs, matrix_factor_dimensions, vs, mask, regularizers=[
            main.paper_regularization(weight_decay, rows, 'row'),
            main.paper_regularization(weight_decay, cols, 'col')
        ])

        main.plot.quiver(xx, yy, RCus.detach().cpu(), RCvs.detach().cpu(), save_if=True, path=f'{path}/{mask_rate}DropRecovered.png')

        RCMatrices = {
            'us': (us_losses, RCus),
            'vs': (vs_losses, RCvs)
        }

        losses = {n: t[0] for n, t in RCMatrices.items()}
        losses['x_plot'] = np.arange(0, epochs, 1)

        main.plot.lines(losses, save_if=True, path=f'{path}/{mask_rate}DropNMAE.png', black_if=True, ylabel_name='NMAE')

if __name__ == '__main__':
    args = get_argparser().parse_args()
    with open(f'{args.save_dir}/args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    do([0.3, 0.5, 0.7, 0.9], args.epochs, args.weight_decay, args.num_factors, args.grid_density, args.save_dir)

import argparse
from collections import namedtuple, OrderedDict
import itertools
import os
import json
import pdb
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import scipy.interpolate as interp

import model
import data


device = 'cuda' if torch.cuda.is_available() else 'cpu'


Coordinates = namedtuple('Coordinates', ['x', 'y'])
VectorField = namedtuple('VectorField', ['coords', 'velx', 'vely'])


def get_argparser():
    parser = argparse.ArgumentParser(description='Run deep matrix factorization with 2d aneurysm.')
    parser.add_argument('--max-epochs', type=int, default=20_000,
                        help='maximum number of epochs.')
    parser.add_argument('--desired-loss', type=float, default=1e-4,
                        help='desired loss for early training termination.')
    parser.add_argument('--num-factors', type=int, default=3,
                        help='number of matrix factors.')
    parser.add_argument('--grid-density', type=int, default=200,
                        help='the number of points on the side of the interpolation grid.')
    parser.add_argument('--mask-rate', type=float, default=None,
                        help='the expected precentage of matrix entries to be set to zero.')
    parser.add_argument('--report-frequency', type=int, default=500,
                        help='the number of epochs to pass before printing a report to the console.')
    parser.add_argument('--data-dir', type=Path,
                        help='path to the matrix completion test data.')
    parser.add_argument('--save-dir', type=str,
                        help='where to save the figures.')
    parser.add_argument('--run-matrix-config', type=bool, default=False,
                        help='run over various test configurations.')
    return parser


def num_large_singular_values(matrix, threshold=5e-1):
    """
    Returns the number of singular values greater than some
    threshold value.

    Parameters
    ----------
    matrix: numeric
        The matrix to be analyzed.
    
    threshold: scalar, default 5e-1
        The threshold that a singular value must be greater
        than to be counted.

    Returns
    -------
        int
    """
    u, s, vh = np.linalg.svd(matrix, full_matrices=False)
    return np.sum(np.where(s > threshold, 1, 0))


def vec_field_component_names():
    names = ['velx', 'vely']
    for n in names:
        yield n


def run_test(**args):
    run_velocity_by_time(**args)


def run_timeframe(tf, **args):
    save_dir_timeframe = lambda p: f'{args["save_dir"]}/{p}.{tf.time}'

    tf_grid = tf.as_completable(grid_density=args['grid_density'])
    tf_grid.vec_field.save(save_dir_timeframe('interpolated'))

    rows, cols = tf_grid.vec_field.velx.shape
    matrix_factor_dimensions = [model.Shape(rows=rows, cols=rows) for _ in range(args['num_factors'] - 1)]
    matrix_factor_dimensions.append(model.Shape(rows=rows, cols=cols))
    print(matrix_factor_dimensions)

    def meets_stop_criteria(epoch, loss):
        return loss < args['desired_loss']

    def report(reconstructed_matrix, epoch, loss, last_report: bool, component):
        vel = data.interp_griddata(tf_grid.vec_field.coords, reconstructed_matrix, tf.vec_field.coords)
        nmae_against_original = model.norm_mean_abs_error(vel, getattr(tf.vec_field, component), lib=np)
        print(f'Component: {component}, Epoch: {epoch}, Loss: {loss:.5e}, NMAE (Original): {nmae_against_original:.5e}')
        if last_report:
            print(f'\n*** END {component} ***\n')
    
    training_names = vec_field_component_names()
    def trainer(vel):
        name = next(training_names)
        return model.train(
            max_epochs=args['max_epochs'],
            matrix_factor_dimensions=matrix_factor_dimensions,
            masked_matrix=vel,
            mask=mask,
            meets_stop_criteria=meets_stop_criteria,
            report_frequency=args['report_frequency'],
            report=lambda *args: report(*args, component=name)
        )

    mask = model.get_bit_mask((rows, cols), args['mask_rate'])
    mask_numpy = mask.cpu().numpy()

    tf_grid_masked = tf_grid.transform(lambda vel: vel * mask_numpy)
    tf_grid_masked.vec_field.save(save_dir_timeframe('masked_interpolated'))

    print(f'Mask Rate: {args["mask_rate"]}')
    tf_grid_masked_rec = tf_grid_masked.numpy_to_torch().transform(trainer).torch_to_numpy()
    tf_grid_masked_rec.vec_field.save(save_dir_timeframe('reconstructed_interpolated'))
    tf_grid_masked_rec.vec_field.interp(coords=tf.vec_field.coords).save(save_dir_timeframe('reconstructed'))


def run_velocity_by_time(**args):
    save_dir = lambda p: f'{args["save_dir"]}/{p}'
    time = 0
    tf = data.AneurysmTimeframe(time=time, filepath=args['data_dir'] / 'aneurysm' / f'vel_2Daneu_crop.{time}.csv')
    vbt = data.AneurysmVelocityByTime(
        coords=tf.vec_field.coords,
        filepath_vel_by_time=args['data_dir'] / 'aneurysm' / 'my_vel_by_time.csv',
        # filepath_vel_by_time=args['data_dir'] / 'aneurysm' / 'vel_by_time.csv',
    )

    # tfs = []
    for t in range(vbt.timeframes):
        # tf = data.AneurysmTimeframe(time=time, filepath=args['data_dir'] / 'aneurysm' / f'vel_2Daneu_crop.{time}.csv')
        # tfs.append(tf)
        tf = vbt.timeframe(t)
        # tf.vec_field.save(save_dir('test_vector_field'))
        run_timeframe(tf, **args)
        break
    # velxs = [tf.vec_field.velx.reshape(-1, 1) for tf in tfs]
    # velys = [tf.vec_field.vely.reshape(-1, 1) for tf in tfs]
    # vbt = data.AneurysmVelocityByTime(coords=vbt.coords, velx_by_time=np.hstack(velxs), vely_by_time=np.hstack(velys))
    # matrix = vbt.as_completable(interleved=True)
    # save = lambda name, arr: np.savetxt(f'{args["save_dir"]}_{name}.csv', arr, delimiter=',')
    # save('new_vbt', matrix)


if __name__ == '__main__':
    args = get_argparser().parse_args().__dict__
    if args['run_matrix_config']:
        config_values = OrderedDict(
            num_factors=[2, 3, 4, 5],
            mask_rate=[0.3, 0.5, 0.7, 0.9],
            grid_density = [100, 200, 300, 400, 500]
        )
        config_values_keys = config_values.keys()
        save_dir = args['save_dir']
        for params in itertools.product(*config_values.values()):
            folder_name = f'{save_dir}/'
            for k, p in zip(config_values_keys, params):
                args[k] = p
                folder_name += f'{k}_{p}__'
            os.mkdir(folder_name)
            args['save_dir'] = folder_name
            with open(f'{folder_name}/config.json', 'w') as f:
                json.dump(args, f, indent=2)
            run_test(**args)
    else:
        run_test(**args)

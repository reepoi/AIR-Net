import argparse
from collections import namedtuple, OrderedDict
import enum
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


class DataSet(enum.Enum):
    ANEURYSM = 'aneurysm'
    FUNC1 = 'func1'
    FUNC2 = 'func2'


class Algorithm(enum.Enum):
    DMF = 'dmf'
    IST = 'ist'


def get_argparser():
    parser = argparse.ArgumentParser(description='Run deep matrix factorization with 2d aneurysm.')
    parser.add_argument('--max-epochs', type=int, default=20_000,
                        help='maximum number of epochs.')
    parser.add_argument('--desired-loss', type=float, default=1e-4,
                        help='desired loss for early training termination.')
    parser.add_argument('--num-factors', type=int, default=3,
                        help='number of matrix factors.')
    parser.add_argument('--mask-rate', type=float, default=None,
                        help='the expected precentage of matrix entries to be set to zero.')
    parser.add_argument('--algorithm', type=Algorithm,
                        help='the algorithm to use for matrix completion.')
    parser.add_argument('--data-set', type=DataSet,
                        help='the data set in the data dir to use.')
    parser.add_argument('--data-dir', type=Path,
                        help='path to the matrix completion data.')
    parser.add_argument('--save-dir', type=str,
                        help='where to save the figures.')
    parser.add_argument('--run-matrix-config', type=bool, default=False,
                        help='run over various test configurations.')
    parser.add_argument('--report-frequency', type=int, default=500,
                        help='the number of epochs to pass before printing a report to the console.')

    # Timeframe options
    parser.add_argument('--grid-density', type=int, default=200,
                        help='the number of points on the side of the interpolation grid.')
    parser.add_argument('--timeframe', type=int, default=None,
                        help='the timeframe to use when the vector field is time dependent.')

    # VelocityByTime options
    parser.add_argument('--interleaved', dest='interleaved', action='store_true')
    parser.add_argument('--no-interleaved', dest='interleaved', action='store_false')
    parser.set_defaults(interleaved=None)
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


def run_timeframe(tf, **args):
    def meets_stop_criteria(epoch, loss):
        return loss < args['desired_loss']

    def report(reconstructed_matrix, epoch, loss, last_report: bool, component):
        vel = data.interp_griddata(tf_grid.vec_field.coords, reconstructed_matrix, tf.vec_field.coords)
        nmae_against_original = model.norm_mean_abs_error(vel, getattr(tf.vec_field, component), lib=np)
        print(f'Component: {component}, Epoch: {epoch}, Loss: {loss:.5e}, NMAE (Original): {nmae_against_original:.5e}')
        if last_report:
            print(f'\n*** END {component} ***\n')

    save_dir_timeframe = lambda p: f'{args["save_dir"]}/{p}.{tf.time}'

    tf.vec_field.save(save_dir_timeframe('original'))

    tf_grid = tf.as_completable(grid_density=args['grid_density'])
    tf_grid.vec_field.save(save_dir_timeframe('interpolated'))

    rows, cols = tf_grid.vec_field.velx.shape

    mask = model.get_bit_mask((rows, cols), args['mask_rate'])
    mask_numpy = mask.cpu().numpy()

    training_names = (c for c in tf.vec_field.components)

    tf_grid_masked = tf_grid.transform(lambda vel: vel * mask_numpy)
    tf_grid_masked.vec_field.save(save_dir_timeframe('masked_interpolated'))

    print(f'Mask Rate: {args["mask_rate"]}')

    if args['algorithm'] is Algorithm.DMF:
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
        matrix_factor_dimensions = [model.Shape(rows=rows, cols=rows) for _ in range(args['num_factors'] - 1)]
        matrix_factor_dimensions.append(model.Shape(rows=rows, cols=cols))
        print(matrix_factor_dimensions)
        tf_grid_masked_rec = tf_grid_masked.numpy_to_torch().transform(trainer).torch_to_numpy()
    elif args['algorithm'] is Algorithm.IST:
        def trainer(vel):
            name = next(training_names)
            return model.iterated_soft_thresholding(
                masked_matrix=vel,
                mask=mask_numpy,
                report_frequency=args['report_frequency'],
                report=lambda *args: report(*args, component=name)
            )
        tf_grid_masked_rec = tf_grid_masked.transform(trainer)

    tf_grid_masked_rec.vec_field.save(save_dir_timeframe('reconstructed_interpolated'))
    tf_grid_masked_rec.vec_field.interp(coords=tf.vec_field.coords).save(save_dir_timeframe('reconstructed'))


def run_velocity_by_time(vbt, **args):
    def meets_stop_criteria(epoch, loss):
        return loss < args['desired_loss']

    def report(reconstructed_matrix, epoch, loss, last_report: bool, component):
        if interleaved:
            num_components = len(vbt.components)
            vbt_reported = vbt.__class__(
                coords=vbt.coords,
                **{c: reconstructed_matrix[i::num_components] for i, c in enumerate(vbt.components)}
            )
            nmaes = {c: model.norm_mean_abs_error(getattr(vbt, c), getattr(vbt_reported, c), lib=np) for c in vbt.components}
            print(f'Component: all, Epoch: {epoch}, Loss: {loss:.5e},', *(f'NMAE_{c} (Original): {nmae:.5e}' for c, nmae in nmaes.items()))
        else:
            nmae_against_original = model.norm_mean_abs_error(reconstructed_matrix, getattr(vbt, component), lib=np)
            print(f'Component: {component}, Epoch: {epoch}, Loss: {loss:.5e}, NMAE (Original): {nmae_against_original:.5e}')
        if last_report:
            print(f'\n*** END {"all" if interleaved else component} ***\n')

    plot_time = 0
    interleaved = args['interleaved']
    save_dir = lambda p: f'{args["save_dir"]}/{p}'

    vbt.save(save_dir('original'), plot_time=plot_time)

    rows, cols = vbt.shape_as_completable(interleaved=interleaved)

    mask = model.get_bit_mask((rows, cols), args['mask_rate'])
    mask_numpy = mask.cpu().numpy()

    training_names = (c for c in vbt.components)
    vbt_masked = vbt.transform(lambda vel: vel * mask_numpy, interleaved=interleaved)
    vbt_masked.save(save_dir('masked'), plot_time=plot_time)

    print(f'Mask Rate: {args["mask_rate"]}')

    if args['algorithm'] is Algorithm.DMF:
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
        matrix_factor_dimensions = [model.Shape(rows=rows, cols=rows) for _ in range(args['num_factors'] - 1)]
        matrix_factor_dimensions.append(model.Shape(rows=rows, cols=cols))
        print(matrix_factor_dimensions)
        vbt_rec = vbt_masked.numpy_to_torch().transform(trainer, interleaved=interleaved).torch_to_numpy()
    elif args['algorithm'] is Algorithm.IST:
        def trainer(vel):
            name = next(training_names)
            return model.iterated_soft_thresholding(
                masked_matrix=vel,
                mask=mask_numpy,
                report_frequency=args['report_frequency'],
                report=lambda *args: report(*args, component=name)
            )
        vbt_rec = vbt_masked.transform(trainer, interleaved=interleaved)

    vbt_rec.save(save_dir('reconstructed'), plot_time=plot_time)


def run_test(**args):
    ds = args['data_set']
    if ds is DataSet.ANEURYSM:
        time = 0
        tf = data.TimeframeAneurysm(time=time, filepath=args['data_dir'] / DataSet.ANEURYSM.value / f'vel_2Daneu_crop.{time}.csv')
        vbt = data.VelocityByTimeAneurysm(
            coords=tf.vec_field.coords,
            filepath_vel_by_time=args['data_dir'] / DataSet.ANEURYSM.value / 'my_vel_by_time.csv',
        )
    elif ds is DataSet.FUNC1:
        func_x = lambda t, x, y: np.sin(2 * x + 2 * y)
        func_y = lambda t, x, y: np.cos(2 * x - 2 * y)
        vbt = data.velocity_by_time_function(func_x, func_y, (-2, 2), args['grid_density'])
    elif ds is DataSet.FUNC2:
        func_x = lambda t, x, y, z: np.sin(2 * x + 2 * y)
        func_y = lambda t, x, y, z: np.cos(2 * x - 2 * y)
        func_z = lambda t, x, y, z: np.cos(2 * x - 2 * z)
        vbt = data.velocity_by_time_function_3d(func_x, func_y, func_z, (-2, 2), args['grid_density'])
    
    if args['interleaved'] is None:
        if (t := args['timeframe']) >= 0:
            timeframes = [t]
        else:
            timesframes = range(vbt.timeframes)
        for t in timeframes:
            run_timeframe(vbt.timeframe(t), **args)
            break
    else:
        run_velocity_by_time(vbt, **args)


if __name__ == '__main__':
    args = get_argparser().parse_args().__dict__

    if args['run_matrix_config']:
        config_values = OrderedDict(
            mask_rate=[0.3, 0.5, 0.7, 0.9],
            num_factors=[2, 3, 4, 5],
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

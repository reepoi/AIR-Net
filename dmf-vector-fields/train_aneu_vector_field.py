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


def ravel_Coordinates(coords: Coordinates):
    """
    Flattens the Coordinate's components' shape.

    Parameters
    ----------
    coords: Coordinates
        A Coordinates instance.
    
    Returns
    -------
        A new Coordinates instance with components whose
        shape is flattened.
    """
    return Coordinates(x=coords.x.ravel(), y=coords.y.ravel())


def ravel_VectorField(vec_field: VectorField):
    """
    Flattens the shape of the VectorField's Coordinates,
    and its velx and vely components.

    Parameters
    ----------
    vec_field: VectorField
        A VectorField instance.
    
    Returns
    -------
        A new VectorField instance with components whose
        shape is flattened.
    """
    return VectorField(
        coords=ravel_Coordinates(vec_field.coords),
        velx=vec_field.velx.ravel(),
        vely=vec_field.vely.ravel()
    )


def list_VectorField(vec_field: VectorField):
    """
    Lists out all the data fields of a VectorField.

    Parameters
    ----------
    vec_field: VectorField
        A VectorField instance.

    Returns
    -------
        A tuple of all the data fields of a VectorField.
    """
    return vec_field.coords.x, vec_field.coords.y, vec_field.velx, vec_field.vely


def save_VectorField(vec_field: VectorField, directory):
    """
    Saves the data of a VectorField using ``np.savetxt``.

    Parameters
    ----------
    vec_field: VectorField
        A VectorField instance.
    
    directory: str
        The directory where the files should be saved.
    """
    save = lambda name, arr: np.savetxt(f'{directory}_{name}.csv', arr, delimiter=',')
    save('coords_x', vec_field.coords.x)
    save('coords_y', vec_field.coords.y)
    save('velx', vec_field.velx)
    save('vely', vec_field.vely)


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


def run_test(**args):
    run_aneursym(**args)


def run_aneursym(**args):
    save_dir = lambda p: f'{args["save_dir"]}/{p}'
    time = 0
    at = data.AneurysmTimeframe(time=time, filepath=args['data_dir'] / 'aneurysm' / f'vel_2Daneu_crop.{time}.csv')
    at_grid = at.as_completable(grid_density=args['grid_density'])

    at_grid.vec_field.save(save_dir('interpolated'))

    rows, cols = at_grid.vec_field.velx.shape
    matrix_factor_dimensions = [model.Shape(rows=rows, cols=rows) for _ in range(args['num_factors'] - 1)]
    matrix_factor_dimensions.append(model.Shape(rows=rows, cols=cols))

    print(matrix_factor_dimensions)

    mask = model.get_bit_mask((rows, cols), args['mask_rate'])
    mask_numpy = mask.cpu().numpy()

    at_grid_masked = at_grid.transform(lambda vel: vel * mask_numpy)

    at_grid_masked.vec_field.save(save_dir('masked_interpolated'))

    def meets_stop_criteria(epoch, loss):
        return loss < args['desired_loss']

    def report(reconstructed_matrix, epoch, loss, last_report: bool, component):
        vel = data.interp_griddata(at_grid.vec_field.coords, reconstructed_matrix, at.vec_field.coords)
        nmae_against_original = model.norm_mean_abs_error(vel, getattr(at.vec_field, component), lib=np)
        print(f'Component: {component}, Epoch: {epoch}, Loss: {loss:.5e}, NMAE (Original): {nmae_against_original:.5e}')
        if last_report:
            print(f'\n*** END {component} ***\n')
    
    print(f'Mask Rate: {args["mask_rate"]}')
    training_names = (n for n in ('velx', 'vely'))
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
    at_grid_masked_rec = at_grid_masked.numpy_to_torch().transform(trainer).torch_to_numpy()

    at_grid_masked_rec.vec_field.save(save_dir('reconstructed_interpolated'))
    at_grid_masked_rec.vec_field.interp(coords=at.vec_field.coords).save(save_dir('reconstructed'))


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

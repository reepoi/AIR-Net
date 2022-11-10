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
import plots
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


def interp_griddata(coords: Coordinates, func_values, new_coords: Coordinates, **kwargs):
    """
    Runs SciPy Interpolate's griddata. This method is to
    make sure the same interpolation method is used throughout
    the script.

    Parameters
    ----------
    coords: Coordinates
        The Coordinates where the values of the interpolated
        function are defined.
    
    func_values: numeric
        The values for each of the points of Coordinates.
    
    new_coords: Coordinates
        The Coordinates where the an interpolated function value
        should be produced.
    
    Returns
    -------
        numeric
        The interpolated function values.
    """
    return interp.griddata(coords, func_values, new_coords, method='linear', **kwargs)


def interp_vector_field(vec_field: VectorField, coords: Coordinates, **interp_opts):
    """
    Uses interpolation to change the grid on which a vector field
    is defined.

    Parameters
    ----------
    vec_field: VectorField
        The current vector field that will be used to create an
        interpolation funciton.
    
    coords: Coordinates
        The new grid for the vector field to be defined on.
    
    Returns
    -------
        A VectorField on defined on new Coordinates.
    """
    new_velx = interp_griddata(vec_field.coords, vec_field.velx, coords, **interp_opts)
    new_vely = interp_griddata(vec_field.coords, vec_field.vely, coords, **interp_opts)
    return VectorField(coords=coords, velx=new_velx, vely=new_vely)


def load_aneu_v_by_t():
    # duplicate point at row 39, 41
    # x   y   velx      vely
    # 1.9 0.0 -0.000152 -8.057502e-07
    data = pd.read_csv('../amir/vel_2Daneu_crop.csv')
    data = data.drop(41) # remove the duplicate
    tf_vec_field = load_aneu_timeframe('../amir/cropped_2D_aneurysm/vel_2Daneu_crop.0.csv')
    tf_vec_field = ravel_VectorField(tf_vec_field)
    velx, vely = data[0::2], data[1::2]
    return VectorField(
        coords=tf_vec_field.coords,
        velx=velx,
        vely=vely
    )


def load_aneu_timeframe(path):
    """
    Load and preprocess one timeframe of the 2d aneurysm data set.
    Any duplicate spatial coordinates are removed.

    Parameters
    ----------
    path : str
        The path where to find the 2d aneurysm data set timeframe.
    
    Returns
    -------
        VectorField
    """
    data = pd.read_csv(path)
    col_idxs, col_names = [0, 1, 3, 4], ['velx', 'vely', 'x', 'y']
    data = data.rename(columns={data.columns[i]: n for i, n in zip(col_idxs, col_names)})
    data = data[col_names] # drop extra columns
    data = data.set_index(['x', 'y']) # set index to (x, y)
    data = data.groupby(level=data.index.names).first() # remove duplicate (x, y)
    vec_field = VectorField(
        coords=Coordinates(
            x=data.index.get_level_values('x').to_numpy(),
            y=data.index.get_level_values('y').to_numpy()
        ),
        velx=data['velx'].to_numpy(),
        vely=data['vely'].to_numpy()
    )
    return vec_field


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
    time = 0
    aneurysm_timeframe = data.AneurysmTimeframe(time=time, filepath=args['data_dir'] / 'aneurysm' / f'vel_2Daneu_crop.{time}.csv')
    vel_by_time = data.AneurysmVelocityByTime(filepath_vel_by_time=args['data_dir'] / 'aneurysm' / f'vel_by_time.csv', aneurysm_timeframe=aneurysm_timeframe)


    # vec_field = load_aneu_timeframe(args['aneu_path'], args['grid_density'])
 
    # grid_coords = Coordinates(*np.meshgrid(
    #     np.linspace(np.min(vec_field.coords.x), np.max(vec_field.coords.x), num=args['grid_density']),
    #     np.linspace(np.min(vec_field.coords.y), np.max(vec_field.coords.y), num=args['grid_density'])
    # ))

    # grid_vec_field = interp_vector_field(vec_field, grid_coords, fill_value=0)
    # save_VectorField(grid_vec_field, f'{args["save_dir"]}/interpolated')
    # fig, _ = plots.quiver(*list_VectorField(grid_vec_field), scale=400,
    #                       save_path=f'{args["save_dir"]}/interpolated.png')

    # matrix = grid_vec_field.velx

    # print(f'velx Rank: {num_large_singular_values(grid_vec_field.velx)}')
    # print(f'vely Rank: {num_large_singular_values(grid_vec_field.vely)}')


    matrix = vel_by_time.as_completable1()
    rows, cols = matrix.shape
    matrix_factor_dimensions = [model.Shape(rows=rows, cols=rows) for _ in range(args['num_factors'] - 1)]
    matrix_factor_dimensions.append(model.Shape(rows=rows, cols=cols))

    print(matrix_factor_dimensions)

    mask = model.get_bit_mask(matrix.shape, args['mask_rate'])
    mask_numpy = mask.cpu().numpy()

    # grid_vec_field_masked = VectorField(coords=grid_vec_field.coords,
    #                                     velx=grid_vec_field.velx * mask_numpy,
    #                                     vely=grid_vec_field.vely * mask_numpy)
    # save_VectorField(grid_vec_field_masked, f'{args["save_dir"]}/interpolated_masked')
    # fig, _ = plots.quiver(*list_VectorField(grid_vec_field_masked), scale=400,
    #                        save_path=f'{args["save_dir"]}/masked_interpolated.png')

    def meets_stop_criteria(epoch, loss):
        return loss < args['desired_loss']

    def report(reconstructed_matrix, epoch, loss, last_report: bool, column):
        # vel = interp_griddata(ravel_Coordinates(grid_vec_field.coords), reconstructed_matrix.ravel(), vec_field.coords)
        # nmae_against_original = model.norm_mean_abs_error(vel, getattr(vec_field, column), lib=np)
        # print(f'Column: {column}, Epoch: {epoch}, Loss: {loss:.5e}, NMAE (Original): {nmae_against_original:.5e}')
        if last_report:
            print(f'\n*** END {column} ***\n')
    
    print(f'Mask Rate: {args["mask_rate"]}')
    rec_matrix=model.iterated_soft_thresholding(matrix, mask_numpy,
                                                report_frequency=args['report_frequency'],
                                                report=lambda *args: vel_by_time.accuracy_report1(*args, column='velx', ground_truth_matrix=matrix))
    reconstructed_grid_vec_field = VectorField(
        coords=grid_vec_field.coords,
        velx=rec_matrix[0::2, 0],
        vely=rec_matrix[1::2, 0]
        # velx=model.iterated_soft_thresholding(grid_vec_field.velx, mask_numpy,
        #                                       report_frequency=args['report_frequency'],
        #                                       report=lambda *args: report(*args, column='velx')),
        # vely=model.iterated_soft_thresholding(grid_vec_field.vely, mask_numpy,
        #                                       report_frequency=args['report_frequency'],
        #                                       report=lambda *args: report(*args, column='vely'))
        # velx=model.train(args['max_epochs'], matrix_factor_dimensions, torch.tensor(grid_vec_field.velx).to(device), mask,
        #                  meets_stop_criteria=meets_stop_criteria,
        #                  report_frequency=args['report_frequency'], report=lambda *args: report(*args, column='velx')),
        # vely=model.train(args['max_epochs'], matrix_factor_dimensions, torch.tensor(grid_vec_field.vely).to(device), mask,
        #                  meets_stop_criteria=meets_stop_criteria,
        #                  report_frequency=args['report_frequency'], report=lambda *args: report(*args, column='vely'))
    )
    save_VectorField(reconstructed_grid_vec_field, f'{args["save_dir"]}/reconstructed_interpolated')
    fig, _ = plots.quiver(*list_VectorField(reconstructed_grid_vec_field), scale=400,
                           save_path=f'{args["save_dir"]}/reconstructed_interpolated.png')

    reconstructed_vec_field = interp_vector_field(ravel_VectorField(reconstructed_grid_vec_field), vec_field.coords)
    save_VectorField(reconstructed_vec_field, f'{args["save_dir"]}/reconstructed')
    fig, _ = plots.quiver(reconstructed_vec_field.coords.x, reconstructed_vec_field.coords.y,
                           reconstructed_vec_field.velx, reconstructed_vec_field.vely, scale=400,
                           save_path=f'{args["save_dir"]}/reconstructed.png')

    plots.plt.close('all')


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

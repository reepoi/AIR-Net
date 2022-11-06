import argparse
from collections import namedtuple
import pdb

import torch
import numpy as np
import pandas as pd
import scipy.interpolate as interp
import wandb

import model
import plots


device = 'cuda' if torch.cuda.is_available() else 'cpu'


Coordinates = namedtuple('Coordinates', ['x', 'y'])
VectorField = namedtuple('VectorField', ['coords', 'velx', 'vely'])


def get_argparser():
    parser = argparse.ArgumentParser(description='Run deep matrix factorization with 2d aneurysm.')
    parser.add_argument('--max-epochs', type=int, default=20_000,
                        help='maximum number of epochs.')
    parser.add_argument('--num-factors', type=int, default=3,
                        help='number of matrix factors.')
    parser.add_argument('--mask-rate', type=float, default=None,
                        help='the expected precentage of matrix entries to be set to zero.')
    parser.add_argument('--aneu-path', type=str,
                        help='path to the 2d aneurysm data.')
    parser.add_argument('--save-dir', type=str,
                        help='where to save the figures.')
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


def run_test(mask_rate, grid_density, report_frequency, args):
    vec_field = load_aneu_timeframe(args.aneu_path, grid_density)
 
    grid_coords = Coordinates(*np.meshgrid(
        np.linspace(np.min(vec_field.coords.x), np.max(vec_field.coords.x), num=grid_density),
        np.linspace(np.min(vec_field.coords.y), np.max(vec_field.coords.y), num=grid_density)
    ))

    grid_vec_field = interp_vector_field(vec_field, grid_coords, fill_value=0)

    matrix = grid_vec_field.velx

    print(f'velx Rank: {num_large_singular_values(grid_vec_field.velx)}')
    print(f'vely Rank: {num_large_singular_values(grid_vec_field.vely)}')

    rows, cols = matrix.shape
    matrix_factor_dimensions = [model.Shape(rows=rows, cols=rows) for _ in range(args.num_factors - 1)]
    matrix_factor_dimensions.append(model.Shape(rows=rows, cols=cols))

    wandb.config = {
        'dataset': 'Aneursym Interpolated',
        'mask_rate': mask_rate,
        'epochs': args.max_epochs,
        'matrix_factor_dimensions': [(s.rows, s.cols) for s in matrix_factor_dimensions]
    }
    print(matrix_factor_dimensions)

    wandb.init(mode="disabled", project="AIR-Net Vector Fields", entity="taost", config=wandb.config)

    mask = model.get_bit_mask(matrix.shape, mask_rate)
    mask_numpy = mask.cpu().numpy()

    fig, _ = plots.quiver(grid_vec_field.coords.x, grid_vec_field.coords.y,
                           grid_vec_field.velx, grid_vec_field.vely, scale=400,
                           save_path=f'{args.save_dir}/AneuInterp.png')
    fig, _ = plots.quiver(grid_vec_field.coords.x, grid_vec_field.coords.y,
                           grid_vec_field.velx * mask_numpy, grid_vec_field.vely * mask_numpy, scale=400,
                           save_path=f'{args.save_dir}/{mask_rate}DropAneu.png')
    wandb.log({'masked': wandb.Image(plots.matplotlib_to_PIL_Image(fig))})

    def meets_stop_criteria(epoch, loss):
        return loss < 1e-5

    def report(reconstructed_matrix, epoch, loss, last_report: bool, column):
        vel = interp_griddata(ravel_Coordinates(grid_vec_field.coords), reconstructed_matrix.ravel(), vec_field.coords)
        nmae_against_original = model.norm_mean_abs_error(vel, getattr(vec_field, column), lib=np)
        print(f'Column: {column}, Epoch: {epoch}, MSE+REG: {loss:.5f}, NMAE (Original): {nmae_against_original:.5f}')
        if last_report:
            print(f'\n*** END {column} ***\n')
    
    print(f'Mask Rate: {mask_rate}')
    reconstructed_grid_vec_field = VectorField(
        coords=grid_vec_field.coords,
        velx=model.train(args.max_epochs, matrix_factor_dimensions, torch.tensor(grid_vec_field.velx).to(device), mask,
                         meets_stop_criteria=meets_stop_criteria,
                         report_frequency=report_frequency, report=lambda *args: report(*args, column='velx')),
        vely=model.train(args.max_epochs, matrix_factor_dimensions, torch.tensor(grid_vec_field.vely).to(device), mask,
                         meets_stop_criteria=meets_stop_criteria,
                         report_frequency=report_frequency, report=lambda *args: report(*args, column='vely'))
    )

    reconstructed_vec_field = interp_vector_field(ravel_VectorField(reconstructed_grid_vec_field), vec_field.coords)

    fig, _ = plots.quiver(reconstructed_vec_field.coords.x, reconstructed_vec_field.coords.y,
                           reconstructed_vec_field.velx, reconstructed_vec_field.vely, scale=400,
                           save_path=f'{args.save_dir}/{mask_rate}RecoveredAneu.png')
    wandb.log({'recovered': wandb.Image(plots.matplotlib_to_PIL_Image(fig))})

    wandb.finish()


if __name__ == '__main__':
    mask_rates = [0.3, 0.5, 0.7, 0.9]

    args = get_argparser().parse_args()
    if (mr := args.mask_rate):
        mask_rates = [mr]

    for mr in mask_rates:
        run_test(mr, 200, 500, args)

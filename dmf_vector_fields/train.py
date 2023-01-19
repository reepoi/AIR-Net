import argparse
import enum
import json
from pathlib import Path

import numpy as np

from dmf_vector_fields.settings import torch, device
from dmf_vector_fields import model
from dmf_vector_fields import data


class DataSet(enum.Enum):
    ANEURYSM = 'aneurysm'
    FUNC1 = 'func1'
    FUNC2 = 'func2'
    DOUBLE_GYRE = 'double-gyre'
    ARORA2019_5 = 'arora2019-rank5'


class DataSetType(enum.Enum):
    VectorField = 'vector-field'
    Matrix = 'matrix'


class Algorithm(enum.Enum):
    IST = 'ist'
    DMF = 'dmf'


class Technique(enum.Enum):
    IDENTITY = 'identity'
    INTERLEAVED = 'interleaved'
    INTERPOLATED = 'interpolated'


def get_argparser():
    parser = argparse.ArgumentParser(description='Run deep matrix factorization with 2d aneurysm.')
    parser.add_argument('--algorithm', type=Algorithm,
                        help='the algorithm to use for matrix completion.')
    parser.add_argument('--technique', type=Technique,
                        help='the pre-processing technique to use when creating hte matrices for completion.')
    parser.add_argument('--mask-rate', type=float, default=None,
                        help='the expected precentage of matrix entries to be set to zero.')
    parser.add_argument('--data-set', type=DataSet,
                        help='the data set in the data dir to use.')
    parser.add_argument('--data-set-type', type=DataSetType, default=DataSetType.VectorField,
                        help='the type of the data set.')
    parser.add_argument('--timeframes', type=int, default=None,
                        help='the number of time frames to use for a data set.')
    parser.add_argument('--data-dir', type=Path, default=Path('data'),
                        help='path to the matrix completion data.')
    parser.add_argument('--save-dir', type=str, default=Path('..') / 'out' / 'output',
                        help='where to save the figures.')
    parser.add_argument('--report-frequency', type=int, default=500,
                        help='the number of epochs to pass before printing a report to the console.')
    parser.add_argument('--run-all', type=int, default=0,
                        help='given 1, a mask rate, and a data set run all combinations of the other experiment args.')
    parser.add_argument('--min-time-identity-interleaved', type=int, default=2,
                        help='the minimum number of times defined by the vector field to use techinques IDENTITY or INTERPOLATED.')

    # Deep Matrix Factorization options
    parser.add_argument('--max-epochs', type=int, default=1_000_000,
                        help='maximum number of epochs.')
    parser.add_argument('--desired-loss', type=float, default=1e-6,
                        help='desired loss for early training termination.')
    parser.add_argument('--num-factors', type=int, default=3,
                        help='number of matrix factors.')

    # Timeframe options (if technique is Technique.INTERPOLATED)
    parser.add_argument('--grid-density', type=int, default=200,
                        help='the number of points on the side of the interpolation grid.')
    parser.add_argument('--timeframe', type=int, default=-1,
                        help='the timeframe to use when the vector field is time dependent.')
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


def no_requires_grad(tensor):
    tensor.requires_grad = False
    return tensor


def save_json(filename, d):
    with open(f'{filename}.json', 'w') as f:
        json.dump(d, f, indent=4)


def record_report_data(report_data, component, epoch, loss, nmae):
    rd = report_data[component] if component in report_data else {'epoch': [], 'loss': [], 'nmae': []}
    rd['epoch'].append(epoch)
    rd['loss'].append(loss)
    rd['nmae'].append(nmae)
    report_data[component] = rd


def run_timeframe(tf, tf_masked, tf_mask, **args):
    def meets_stop_criteria(epoch, loss):
        return loss < args['desired_loss']

    def report(reconstructed_matrix, epoch, loss, last_report: bool, component):
        vel = data.interp_griddata(tf_grid.vec_field.coords, reconstructed_matrix, tf.vec_field.coords)
        nmae_against_original = model.norm_mean_abs_error(vel, getattr(tf.vec_field, component), lib=np)
        print(f'Component: {component}, Epoch: {epoch}, Loss: {loss:.5e}, NMAE: {nmae_against_original:.5e}')
        record_report_data(report_data, component, epoch, loss.item(), nmae_against_original)
        if last_report:
            print(f'\n*** END {component} ***\n')

    report_data = dict()

    save_dir_timeframe = lambda p: f'{args["save_dir"]}/{p}.{tf.time}'

    # tf.save(save_dir_timeframe('original'))
    tf_grid = tf.as_completable(grid_density=args['grid_density'])
    # tf_grid.save(save_dir_timeframe('interpolated'))

    # tf_masked.save(save_dir_timeframe('masked'))
    tf_masked_grid = tf_masked.as_completable(grid_density=args['grid_density'])
    # tf_masked_grid.save(save_dir_timeframe('masked_interpolated'))

    print(f'Mask Rate: {args["mask_rate"]}')

    training_names = iter(tf.vec_field.components)
    mask = tf_mask.as_completable(grid_density=args['grid_density'], method='linear').vec_field.velx
    mask_torch = no_requires_grad(torch.tensor(mask, dtype=torch.float64).to(device))
    if args['algorithm'] is Algorithm.DMF:
        def trainer(vel):
            name = next(training_names)
            return model.train(
                max_epochs=args['max_epochs'],
                matrix_factor_dimensions=matrix_factor_dimensions,
                masked_matrix=vel,
                mask=mask_torch,
                meets_stop_criteria=meets_stop_criteria,
                report_frequency=args['report_frequency'],
                report=lambda *args: report(*args, component=name)
            )
        rows, cols = tf_grid.vec_field.velx.shape
        min_dim = min(rows, cols)
        matrix_factor_dimensions = [model.Shape(rows=rows, cols=min_dim)]
        matrix_factor_dimensions += [model.Shape(rows=min_dim, cols=min_dim) for _ in range(1, args['num_factors'] - 1)]
        matrix_factor_dimensions.append(model.Shape(rows=min_dim, cols=cols))
        print(matrix_factor_dimensions)
        tf_grid_masked_rec = tf_masked_grid.numpy_to_torch().transform(trainer).torch_to_numpy()
    elif args['algorithm'] is Algorithm.IST:
        def trainer(vel):
            name = next(training_names)
            return model.iterated_soft_thresholding(
                masked_matrix=vel,
                mask=mask_torch,
                normfac=np.max(mask),
                report_frequency=args['report_frequency'],
                report=lambda *args: report(*args, component=name)
            )
        tf_grid_masked_rec = tf_masked_grid.numpy_to_torch().transform(no_requires_grad).transform(trainer).torch_to_numpy()

    tf_grid_masked_rec.save(save_dir_timeframe('reconstructed_interpolated'))
    tf_grid_masked_rec.vec_field.interp(coords=tf.vec_field.coords).save(save_dir_timeframe('reconstructed'))

    save_json(save_dir_timeframe('report_data'), report_data)

    return tf_grid_masked_rec


def run_velocity_by_time(vbt, vbt_masked, vbt_mask, **args):
    def meets_stop_criteria(epoch, loss):
        return loss < args['desired_loss']

    def report(reconstructed_matrix, epoch, loss, last_report: bool, component):
        if interleaved:
            num_components = len(vbt.components)
            vbt_reported = vbt.__class__(
                coords=vbt.coords,
                **{c: reconstructed_matrix[i::num_components] for i, c in enumerate(vbt.components)}
            )
            nmaes = {c: model.norm_mean_abs_error(getattr(vbt_reported, c), getattr(vbt, c), lib=np) for c in vbt.components}
            for c in vbt.components:
                record_report_data(report_data, c, epoch, loss.item(), nmaes[c])
            print(f'Component: all, Epoch: {epoch}, Loss: {loss:.5e}, ' + ', '.join(f'NMAE_{c}: {nmae:.5e}' for c, nmae in nmaes.items()))
        else:
            nmae_against_original = model.norm_mean_abs_error(reconstructed_matrix, getattr(vbt, component), lib=np)
            record_report_data(report_data, component, epoch, loss.item(), nmae_against_original)
            print(f'Component: {component}, Epoch: {epoch}, Loss: {loss:.5e}, NMAE: {nmae_against_original:.5e}')
        if last_report:
            print(f'\n*** END {"all" if interleaved else component} ***\n')

    report_data = dict()

    plot_time = 0
    interleaved = args['interleaved']
    save_dir = lambda p: f'{args["save_dir"]}/{p}'

    # vbt.save(save_dir('original'), plot_time=plot_time)
    # vbt_masked.save(save_dir('masked'), plot_time=plot_time)

    print(f'Mask Rate: {args["mask_rate"]}')

    training_names = iter(vbt.components)
    mask = vbt_mask.completable_matrices(interleaved=interleaved)
    mask = mask if interleaved else mask[0]
    mask_torch = no_requires_grad(torch.tensor(mask, dtype=torch.float64).to(device))
    if args['algorithm'] is Algorithm.DMF:
        def trainer(vel):
            name = next(training_names)
            return model.train(
                max_epochs=args['max_epochs'],
                matrix_factor_dimensions=matrix_factor_dimensions,
                masked_matrix=vel,
                mask=mask_torch,
                meets_stop_criteria=meets_stop_criteria,
                report_frequency=args['report_frequency'],
                report=lambda *args: report(*args, component=name)
            )
        rows, cols = vbt.shape_as_completable(interleaved=interleaved)
        min_dim = min(rows, cols)
        matrix_factor_dimensions = [model.Shape(rows=rows, cols=min_dim)]
        matrix_factor_dimensions += [model.Shape(rows=min_dim, cols=min_dim) for _ in range(1, args['num_factors'] - 1)]
        matrix_factor_dimensions.append(model.Shape(rows=min_dim, cols=cols))
        print(matrix_factor_dimensions)
        vbt_rec = vbt_masked.numpy_to_torch().transform(trainer, interleaved=interleaved).torch_to_numpy()
    elif args['algorithm'] is Algorithm.IST:
        def trainer(vel):
            name = next(training_names)
            return model.iterated_soft_thresholding(
                masked_matrix=vel,
                mask=mask_torch,
                report_frequency=args['report_frequency'],
                report=lambda *args: report(*args, component=name)
            )
        vbt_rec = vbt_masked.numpy_to_torch().transform(no_requires_grad).transform(trainer, interleaved=interleaved).torch_to_numpy()

    vbt_rec.save(save_dir('reconstructed'), plot_time=plot_time)

    save_json(save_dir('report_data'), report_data)

    return vbt_rec


def skip_test(vbt, **args):
    tq = args['technique']
    if tq in {Technique.IDENTITY, Technique.INTERLEAVED}:
        if vbt.timeframes < args['min_time_identity_interleaved']:
            return 'data set has too few timeframes. See --min-time-identity-interleaved'
    if tq is Technique.INTERPOLATED:
        if len(vbt.components) >= 3:
            return 'data sets with 3 or more spatial dimensions not supported by Technique.INTERPOLATED'
    return ''


mask = None
def run_test(**args):
    # Select vector field
    ds = args['data_set']
    num_timeframes = args['timeframes']
    if ds is DataSet.ANEURYSM:
        vbt = data.VelocityByTimeAneurysm.load_from(args['data_dir'] / DataSet.ANEURYSM.value / 'vel_by_time')
        if num_timeframes is not None:
            vbt.velx_by_time, vbt.vely_by_time = vbt.velx_by_time[:, :num_timeframes], vbt.vely_by_time[:, :num_timeframes]
    elif ds is DataSet.FUNC1:
        func_x = lambda t, x, y: np.sin(2 * x + 2 * y)
        func_y = lambda t, x, y: np.cos(2 * x - 2 * y)
        vbt = data.velocity_by_time_function(func_x, func_y, [(-2, 2)] * 2, grid_density=100)
    elif ds is DataSet.DOUBLE_GYRE:
        if num_timeframes is None:
            vbt = data.double_gyre()
        else:
            vbt = data.double_gyre(num_timeframes=num_timeframes)
    elif ds is DataSet.FUNC3:
        func_x = lambda t, x, y, z: np.sin(2 * x + 2 * y)
        func_y = lambda t, x, y, z: np.cos(2 * x - 2 * y)
        func_z = lambda t, x, y, z: np.cos(2 * x - 2 * z)
        vbt = data.velocity_by_time_function_3d(func_x, func_y, func_z, (-2, 2), grid_density=100)

    # Check if test should be skipped
    if (msg := skip_test(vbt, **args)) != '':
        print(f'TEST SKIPPED: {msg}.')
        return

    # Mask vector field
    # The mask should be created once so that is the same for all experiments
    global mask
    if mask is None:
        mask = model.get_bit_mask(vbt.shape_as_completable(interleaved=False), args['mask_rate'])
    vbt_mask = data.VelocityByTime(coords=vbt.coords, velx_by_time=mask, vely_by_time=mask)
    vbt_masked = vbt.transform(lambda vel: vel * mask, interleaved=False)

    # Select pre-processing technique and algorithm
    technique = args['technique']
    data_set_name = ds.value
    if num_timeframes:
        data_set_name = f'{ds.value}_tf{num_timeframes}'
    save_dir = (
        Path(args['save_dir']) / data_set_name / args['algorithm'].value
        / (f'num_factors{args["num_factors"]}' if args['algorithm'] is Algorithm.DMF else '.')
        / f'mask_rate{args["mask_rate"]}' / technique.value
    )
    if technique is Technique.IDENTITY:
        args['interleaved'] = False
        args['save_dir'] = save_dir
        task = lambda: run_velocity_by_time(vbt, vbt_masked, vbt_mask, **args)
    elif technique is Technique.INTERLEAVED:
        args['interleaved'] = True
        args['save_dir'] = save_dir
        task = lambda: run_velocity_by_time(vbt, vbt_masked, vbt_mask, **args)
    elif technique is Technique.INTERPOLATED:
        save_dir = save_dir / f'grid_density{args["grid_density"]}'
        if (t := args['timeframe']) >= 0:
            timeframes = [t]
            save_dir = save_dir / f'time{t}'
        else:
            timeframes = range(vbt.timeframes)
        args['save_dir'] = save_dir
        def task():
            vbt.save(save_dir / 'original')
            tfs = []
            for t in timeframes:
                print(f'***** BEGIN TIME {t} *****')
                tfs.append(run_timeframe(vbt.timeframe(t), vbt_masked.timeframe(t), vbt_mask.timeframe(t), **args))
                print(f'***** END TIME {t} *****')
            vbt_rec = vbt.__class__(coords=vbt.coords, vec_fields=[tf.vec_field.interp(coords=vbt.coords) for tf in tfs])
            vbt_rec.save(save_dir / 'reconstructed')
            return vbt_rec

    # Create save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    # Run algorithm
    vbt_rec = task()

    # Record final NMAE results
    nmaes = {c: model.norm_mean_abs_error(getattr(vbt_rec, c), getattr(vbt, c), lib=np) for c in vbt.components}
    save_json(save_dir / 'nmae', nmaes)


def run_matrix(m, m_masked, **args):
    def meets_stop_criteria(epoch, loss):
        return loss < args['desired_loss']

    def report(reconstructed_matrix, epoch, loss, last_report: bool):
        nmae_against_original = model.norm_mean_abs_error(reconstructed_matrix, m.data, lib=np)
        record_report_data(report_data, 'matrix_data', epoch, loss.item(), nmae_against_original)
        print(f'Component: matrix_data, Epoch: {epoch}, Loss: {loss:.5e}, NMAE: {nmae_against_original:.5e}')
        if last_report:
            print('\n*** END matrix_data ***\n')

    report_data = dict()

    save_dir = lambda p: f'{args["save_dir"]}/{p}'

    # vbt.save(save_dir('original'), plot_time=plot_time)
    # vbt_masked.save(save_dir('masked'), plot_time=plot_time)

    print(f'Mask Rate: {args["mask_rate"]}')

    mask = np.zeros_like(m.data)
    mask[m.sampled_index_u, m.sampled_index_v] = 1
    mask_torch = no_requires_grad(torch.tensor(mask, dtype=torch.float64).to(device))
    if args['algorithm'] is Algorithm.DMF:
        def trainer(vel):
            return model.train(
                max_epochs=args['max_epochs'],
                matrix_factor_dimensions=matrix_factor_dimensions,
                masked_matrix=vel,
                mask=mask_torch,
                meets_stop_criteria=meets_stop_criteria,
                report_frequency=args['report_frequency'],
                report=lambda *args: report(*args)
            )
        rows, cols = m.data.shape
        min_dim = min(rows, cols)
        matrix_factor_dimensions = [model.Shape(rows=rows, cols=min_dim)]
        matrix_factor_dimensions += [model.Shape(rows=min_dim, cols=min_dim) for _ in range(1, args['num_factors'] - 1)]
        matrix_factor_dimensions.append(model.Shape(rows=min_dim, cols=cols))
        print(matrix_factor_dimensions)
        m_rec = m_masked.numpy_to_torch().transform(trainer).torch_to_numpy()
    elif args['algorithm'] is Algorithm.IST:
        def trainer(vel):
            return model.iterated_soft_thresholding(
                masked_matrix=vel,
                mask=mask_torch,
                report_frequency=args['report_frequency'],
                report=lambda *args: report(*args)
            )
        m_rec = m_masked.numpy_to_torch().transform(no_requires_grad).transform(trainer).torch_to_numpy()

    m_rec.save(save_dir('reconstructed'))

    save_json(save_dir('report_data'), report_data)

    return m_rec


matrix_mask = None
def run_matrix_test(**args):
    # Select vector field
    ds = args['data_set']
    mask_func = None
    if ds is DataSet.ARORA2019_5:
        m = data.MatrixArora2019(
            filepath=args['data_dir'] / 'arora2019',
            rank=5, mask_rate=args['mask_rate']
        )

        def mask_func(x):
            masked = np.zeros_like(x)
            masked[m.sampled_index_u, m.sampled_index_v] = x[m.sampled_index_u, m.sampled_index_v]
            return masked

    # Mask vector field
    # The mask should be created once so that is the same for all experiments
    global matrix_mask
    if matrix_mask is None:
        mask = model.get_bit_mask(m.data.shape, args['mask_rate'])
    if mask_func is None:
        mask_func = lambda x: x * mask

    m_masked = m.transform(mask_func)

    # Select pre-processing technique and algorithm
    technique = args['technique']
    save_dir = (
        Path(args['save_dir']) / ds.value / args['algorithm'].value
        / (f'num_factors{args["num_factors"]}' if args['algorithm'] is Algorithm.DMF else '.')
        / f'mask_rate{args["mask_rate"]}'
    )

    # Create save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    # Run algorithm
    m_rec = run_matrix(m, m_masked, **args)

    # Record final NMAE results
    nmaes = {'matrix_data': model.norm_mean_abs_error(m_rec.data, m.data, lib=np)}
    save_json(save_dir / 'nmae', nmaes)


if __name__ == '__main__':
    args = get_argparser().parse_args().__dict__
    if args['data_set_type'] is DataSetType.VectorField:
        if args['run_all'] == 1:
            grid_density = [100, 200, 300, 400, 500]
            # grid_density = [50]
            for a in [Algorithm.IST, Algorithm.DMF]:
            # for a in Algorithm:
                args['algorithm'] = a
                num_factors = [2, 3, 4, 5] if a is Algorithm.DMF else [1]
                for nf in num_factors:
                    args['num_factors'] = nf

                    # Run identity
                    args['technique'] = Technique.IDENTITY
                    run_test(**args)

                    # Run interleaved
                    args['technique'] = Technique.INTERLEAVED
                    run_test(**args)

                # Run interpolated
                # args['technique'] = Technique.INTERPOLATED
                # for gd in grid_density:
                #     args['grid_density'] = gd
                #     for nf in num_factors:
                #         args['num_factors'] = nf
                #         run_test(**args)
        else:
            run_test(**args)
    elif args['data_set_type'] is DataSetType.Matrix:
        run_matrix_test(**args)

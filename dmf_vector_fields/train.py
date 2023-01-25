import argparse
import json
from pathlib import Path

import numpy as np

from dmf_vector_fields.settings import torch, device
from dmf_vector_fields import model, data
from dmf_vector_fields.enums import DataSet, Algorithm, Technique


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


def meets_stop_criteria(epoch, loss):
    return loss < args['desired_loss']


def run_trainer(components, mask, report, to_train, args, transform_kwargs=None):
    transform_kwargs = transform_kwargs or {}
    components = enumerate(components)
    a = args['algorithm']
    if a is Algorithm.IST:
        def trainer(vel):
            c = next(components)
            return model.iterated_soft_thresholding(
                masked_matrix=vel,
                mask=mask,
                normfac=torch.max(mask),
                report_frequency=args['report_frequency'],
                report=lambda *args: report(*args, component_info=c)
            )
        return (
            to_train
            .numpy_to_torch()
            .transform(no_requires_grad)
            .transform(trainer, **transform_kwargs)
            .torch_to_numpy()
        )
    elif a is Algorithm.DMF:
        def trainer(vel):
            c = next(components)
            return model.train(
                max_epochs=args['max_epochs'],
                matrix_factor_dimensions=matrix_factor_dimensions,
                masked_matrix=vel,
                mask=mask,
                meets_stop_criteria=meets_stop_criteria,
                report_frequency=args['report_frequency'],
                report=lambda *args: report(*args, component_info=c)
            )
        rows, cols = mask.shape
        min_dim = min(rows, cols)
        matrix_factor_dimensions = [model.Shape(rows=rows, cols=min_dim)]
        matrix_factor_dimensions += [model.Shape(rows=min_dim, cols=min_dim) for _ in range(1, args['num_factors'] - 1)]
        matrix_factor_dimensions.append(model.Shape(rows=min_dim, cols=cols))
        print(matrix_factor_dimensions)
        return (
            to_train
            .numpy_to_torch()
            .transform(trainer, **transform_kwargs)
            .torch_to_numpy()
        )


def run_timeframe(tf, tf_masked, tf_mask, **args):
    def report(reconstructed_matrix, epoch, loss, last_report: bool, component_info):
        component_idx, component_name = component_info
        vel = data.interp_griddata(tf_grid.vec_field.coords, reconstructed_matrix, tf.vec_field.coords)
        nmae_against_original = model.norm_mean_abs_error(vel, tf.vec_field.vel_axes[component_idx], lib=np)
        print(f'Component: {component_name}, Epoch: {epoch}, Loss: {loss:.5e}, NMAE: {nmae_against_original:.5e}')
        record_report_data(report_data, component_name, epoch, loss.item(), nmae_against_original)
        if last_report:
            print(f'\n*** END {component_name} ***\n')

    report_data = dict()

    save_dir_timeframe = lambda p: f'{args["save_dir"]}/{p}.{tf.time}'

    tf_grid = tf.as_completable(grid_density=args['grid_density'])

    tf_masked_grid = tf_masked.as_completable(grid_density=args['grid_density'])

    print(f'Mask Rate: {args["mask_rate"]}')

    tf_grid_masked_rec = run_trainer(
        tf.vec_field.components,
        tf_mask.as_completable(grid_density=args['grid_density'], method='linear').numpy_to_torch().vec_field.vel_axes[0],
        report,
        tf_masked_grid,
        args
    )

    tf_grid_masked_rec.save(save_dir_timeframe('reconstructed_interpolated'))
    tf_grid_masked_rec.vec_field.interp(coords=tf.vec_field.coords).save(save_dir_timeframe('reconstructed'))

    save_json(save_dir_timeframe('report_data'), report_data)

    return tf_grid_masked_rec


def run_velocity_by_time(vbt, vbt_masked, vbt_mask, **args):
    def report(reconstructed_matrix, epoch, loss, last_report: bool, component_info):
        component_idx, component_name = component_info
        if interleaved:
            dims = len(vbt.components)
            vbt_reported = vbt.__class__(
                coords=vbt.coords,
                vel_by_time_axes=tuple(reconstructed_matrix[i::dims] for i in range(dims)),
                components=vbt.components
            )
            nmaes = {c: model.norm_mean_abs_error(rep_a, a, lib=np) for c, rep_a, a in zip(vbt.components, vbt_reported.vel_by_time_axes, vbt.vel_by_time_axes)}
            for c in vbt.components:
                record_report_data(report_data, c, epoch, loss.item(), nmaes[c])
            print(f'Component: all, Epoch: {epoch}, Loss: {loss:.5e}, ' + ', '.join(f'NMAE_{c}: {nmae:.5e}' for c, nmae in nmaes.items()))
        else:
            nmae_against_original = model.norm_mean_abs_error(reconstructed_matrix, vbt.vel_by_time_axes[component_idx], lib=np)
            record_report_data(report_data, component_name, epoch, loss.item(), nmae_against_original)
            print(f'Component: {component_name}, Epoch: {epoch}, Loss: {loss:.5e}, NMAE: {nmae_against_original:.5e}')
        if last_report:
            print(f'\n*** END {"all" if interleaved else component_name} ***\n')

    report_data = dict()

    plot_time = 0
    interleaved = args['interleaved']
    save_dir = lambda p: f'{args["save_dir"]}/{p}'

    print(f'Mask Rate: {args["mask_rate"]}')

    vbt_rec = run_trainer(
        vbt.components,
        vbt_mask.as_completable(interleaved=interleaved).numpy_to_torch().vel_by_time_axes[0],
        report,
        vbt_masked,
        args,
        transform_kwargs=dict(interleaved=interleaved)
    )

    vbt_rec.save(save_dir('reconstructed'), plot_time=plot_time)

    save_json(save_dir('report_data'), report_data)

    return vbt_rec


def skip_test(vbt, **args):
    tq = args['technique']
    if tq in {Technique.IDENTITY, Technique.INTERLEAVED}:
        if vbt.timeframes < args['min_time_identity_interleaved']:
            return 'data set has too few timeframes. See --min-time-identity-interleaved'
        if tq is Technique.INTERLEAVED and vbt.vel_by_time_axes == 1:
            return 'data set has too few dimensions'
    if tq is Technique.INTERPOLATED:
        if len(vbt.components) >= 3:
            return 'data sets with 3 or more spatial dimensions not supported by Technique.INTERPOLATED'
    return ''


def choose_dataset(args):
    mask = None
    ds = args['data_set']
    num_timeframes = args['timeframes']
    if ds is DataSet.ANEURYSM:
        vbt = data.VelocityByTimeAneurysm.load_from(args['data_dir'] / DataSet.ANEURYSM.value / 'vel_by_time')
        if num_timeframes is not None:
            for a in vbt.vel_by_time_axes:
                a = a[:, :num_timeframes]
    elif ds is DataSet.FUNC1:
        vbt = data.func1()
    elif ds is DataSet.DOUBLE_GYRE:
        if num_timeframes is None:
            vbt = data.double_gyre()
        else:
            vbt = data.double_gyre(num_timeframes=num_timeframes)
    elif ds is DataSet.FUNC2:
        func_x = lambda t, x, y, z: np.sin(2 * x + 2 * y)
        func_y = lambda t, x, y, z: np.cos(2 * x - 2 * y)
        func_z = lambda t, x, y, z: np.cos(2 * x - 2 * z)
        vbt = data.velocity_by_time_function_3d(func_x, func_y, func_z, (-2, 2), grid_density=100)
    elif ds is DataSet.ARORA2019_5:
        fp = args['data_dir'] / 'arora2019' / 'rank5.pt'
        tf = data.MatrixArora2019(time=0, filepath=fp)
        vbt = data.VelocityByTime(vec_fields=[tf.vec_field])
        # set the mask to the saved mask
        mask = tf.saved_mask('0.8').reshape(vbt.shape_as_completable(interleaved=False))
    elif ds is DataSet.ARORA2019_10:
        fp = args['data_dir'] / 'arora2019' / 'rank10.pt'
        tf = data.MatrixArora2019(time=0, filepath=fp)
        vbt = data.VelocityByTime(vec_fields=[tf.vec_field])
        # set the mask to the saved mask
        mask = tf.saved_mask('0.675').reshape(vbt.shape_as_completable(interleaved=False))
    return vbt, mask


def get_task_by_technique(vbt, vbt_masked, vbt_mask, args):
    ds = args['data_set']
    num_timeframes = args['timeframes']
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
            vbt_rec.save(save_dir / 'reconstructed', plot_time=0)
            return vbt_rec
    return save_dir, task


mask = None


def run_test(**args):
    # Select vector field
    # The mask should be created once so that is the same for all experiments
    global mask
    vbt, mask = choose_dataset(args)

    # Check if test should be skipped
    if (msg := skip_test(vbt, **args)) != '':
        print(f'TEST SKIPPED: {msg}.')
        return

    # Mask vector field
    if mask is None:
        mask = model.get_bit_mask(vbt.shape_as_completable(interleaved=False), args['mask_rate'])
    dim = len(vbt.components)
    vbt_mask = data.VelocityByTime(coords=vbt.coords, vel_by_time_axes=(mask,) * dim, components=('mask',) * dim)
    vbt_masked = vbt.transform(lambda vel: vel * mask, interleaved=False)

    # Select pre-processing technique and algorithm
    save_dir, task = get_task_by_technique(vbt, vbt_masked, vbt_mask, args)

    # Create save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    # Run algorithm
    vbt_rec = task()

    # Record final NMAE results
    nmaes = {n: model.norm_mean_abs_error(rec_a, a, lib=np)
             for n, rec_a, a in zip(vbt.components, vbt_rec.vel_by_time_axes, vbt.vel_by_time_axes)}
    save_json(save_dir / 'nmae', nmaes)


if __name__ == '__main__':
    args = get_argparser().parse_args().__dict__
    if args['run_all'] == 1:
        grid_density = [100]
        # grid_density = [50]
        # for a in [Algorithm.DMF]:
        for a in Algorithm:
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
            args['technique'] = Technique.INTERPOLATED
            for gd in grid_density:
                args['grid_density'] = gd
                for nf in num_factors:
                    args['num_factors'] = nf
                    run_test(**args)
    else:
        run_test(**args)

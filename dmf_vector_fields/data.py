from dataclasses import dataclass
from typing import Tuple
import numpy as np
import pandas as pd
from dmf_vector_fields.settings import torch, device
from dmf_vector_fields import plots
import scipy.interpolate as interp


def auto_component_names(count: int):
    return tuple(f'axis{i}' for i in range(count))


@dataclass
class Coordinates:
    axes: Tuple[np.ndarray]
    components: Tuple[str]

    def __init__(self, axes, components=None):
        if components is None:
            components = auto_component_names(len(axes))
        assert len(axes) == len(components), 'Dimensions do not match.'
        self.axes = tuple(axes)
        self.components = tuple(components)

    @property
    def lib(self):
        """
        The ndarray library used for storing the data fields.

        Returns
        -------
            The torch or numpy module.
        """
        return torch if isinstance(self.axes[0], torch.Tensor) else np

    def transform(self, transform_func):
        """
        Applies a transform to x and y.

        Parameters
        ----------
        transform_func: function(ndarray)
            The transform to apply.

        Returns
        -------
            ``Coordinates``
        """
        return self.__class__(
            tuple(transform_func(a) for a in self.axes),
            self.components
        )

    def ravel(self):
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
        return self.transform(lambda x: x.ravel())

    def bounding_grid(self, grid_density):
        """
        Build the smallest grid such that the area it encloses
        contains the current coordinates.

        Parameters
        ----------
        grid_density: int
            The number of grid points along one edge.

        Returns
        -------
            ``Coordinates``
        """
        lib = self.lib
        ls = lambda c: lib.linspace(lib.min(c), lib.max(c), grid_density)
        mg = lib.meshgrid(*(ls(a) for a in self.axes), indexing='xy')
        return self.__class__(axes=mg, components=self.components)

    def save(self, path):
        save = lambda name, arr: np.savetxt(f'{path}_{name}.csv', arr, delimiter=',')
        for n, a in zip(self.components, self.axes):
            save(n, a)


@dataclass
class VectorField:
    coords: Coordinates
    vel_axes: Tuple[np.ndarray]
    components: Tuple[str]

    def __init__(self, coords, vel_axes, components=None):
        if components is None:
            components = auto_component_names(len(vel_axes))
        assert len(vel_axes) == len(components), 'Dimensions do not match.'
        self.coords = coords
        self.vel_axes = tuple(vel_axes)
        self.components = tuple(components)

    @property
    def lib(self):
        """
        The ndarray library used for storing the data fields.

        Returns
        -------
            The torch or numpy module.
        """
        return self.coords.lib

    def ravel(self):
        """
        Flattens the shape of the VectorField's Coordinates,
        and its velx and vely components.

        Returns
        -------
            A new VectorField instance with components whose
            shape is flattened.
        """
        return self.transform(lambda x: x.ravel(), apply_to_coords=True)

    def to_tuple(self):
        """
        Lists out all the data fields of a VectorField.

        Returns
        -------
            A tuple of all the data fields of a VectorField.
        """
        return *self.coords.axes, *self.vel_axes

    def interp(self, grid_density=None, coords=None, **interp_opts):
        """
        Uses interpolation to change the grid on which a vector field
        is defined.

        Parameters
        ----------
        grid_density: int, default None
            The number of points along one edge of the grid.
            Must not be None if ``coords`` is None.

        coords: Coordinates, default None
            The new grid for the vector field to be defined on.

        **interp_opts: dict
            The keyword arguments to pass to :func:`interp_griddata`.

        Returns
        -------
            A VectorField on defined on new Coordinates.
        """
        if coords is None:
            coords = self.coords.bounding_grid(grid_density)
        new_vel_axes = tuple(
            interp_griddata(self.coords, a, coords, **interp_opts)
            for a in self.vel_axes
        )
        return self.__class__(coords, new_vel_axes, self.components)

    def as_completable(self, grid_density, **kwargs):
        """
        Returns a representation of a VectorField that can be
        given to a matrix completion algorithm.

        Returns
        -------
            ``VectorField``
        """
        return self.interp(grid_density=grid_density, fill_value=0, **kwargs)

    def transform(self, transform_func, apply_to_coords=False):
        """
        Apply ``transform_func`` to velx and vely.

        Parameters
        ----------
        transform_func: function(ndarray)
            Applies a transform to a ndarray.

        apply_to_coords: bool, False
            Apply ``transform_func`` to ``VectorField``'s coordinates.

        Returns
        -------
            A VectorField
        """
        return self.__class__(
            coords=self.coords.transform(transform_func) if apply_to_coords else self.coords,
            vel_axes=tuple(transform_func(a) for a in self.vel_axes),
            components=self.components
        )

    def save(self, path, plot=True):
        save = lambda name, arr: np.savetxt(f'{path}_{name}.csv', arr, delimiter=',')
        self.coords.save(path)
        for n, a in zip(self.components, self.vel_axes):
            save(n, a)
        if plot:
            plots.plot_vec_field(path, self.ravel())


@dataclass
class Timeframe:
    time: int
    filepath: str
    vec_field: VectorField

    def __init__(self, time: int, filepath=None, vec_field=None):
        self.time = time
        self.filepath = filepath
        if vec_field is not None:
            self.vec_field = vec_field
        else:
            assert filepath is not None, 'Need a filepath to load_data'
            self.load_data()

    @property
    def lib(self):
        """
        The ndarray library used for storing the data fields.

        Returns
        -------
            The torch or numpy module.
        """
        return self.vec_field.lib

    def load_data(self):
        raise NotImplementedError('Override this.')

    def as_completable(self, grid_density, **kwargs):
        """
        Returns a tuple of velx and vely.

        Returns
        -------
            An ``Timeframe`` with ``filepath=None``.
        """
        return self.__class__(
            time=self.time,
            filepath=None,
            vec_field=self.vec_field.as_completable(grid_density, **kwargs)
        )

    def transform(self, transform_func, apply_to_coords=False):
        """
        Applies a transform to velx and vely and returns the result.

        transform_func: function(matrix)
            The transform function to apply.

        apply_to_coords: bool, False
            Apply ``transform_func`` to ``Timeframe``'s coordinates.

        Returns
        -------
            An ``Timeframe`` with ``filepath=None``.
        """
        return self.__class__(
            time=self.time,
            filepath=None,
            vec_field=self.vec_field.transform(transform_func, apply_to_coords=apply_to_coords)
        )

    def torch_to_numpy(self):
        """
        Convert all data from torch tensors to numpy ndarrays.

        Returns
        -------
            ``Timeframe`` whose data are numpy ndarrays.
        """
        transform_func = lambda x: x.detach().cpu().numpy()
        return self.transform(transform_func, apply_to_coords=True)

    def numpy_to_torch(self):
        """
        Convert all data from numpy ndarrays to torch tensors.

        Returns
        -------
            ``Timeframe`` whose data are torch tensors.
        """
        transform_func = lambda x: torch.tensor(x).to(device)
        return self.transform(transform_func, apply_to_coords=True)

    def save(self, path, plot=True):
        self.vec_field.save(path, plot=plot)


@dataclass
class VelocityByTime:
    filepath: str
    coords: Coordinates
    vel_by_time_axes: Tuple[np.ndarray]
    components: Tuple[str]

    def __init__(self, coords=None, vel_by_time_axes=None, components=None, filepath=None, vec_fields=None):
        if components is None:
            components = auto_component_names(len(coords.components))
        if vec_fields is not None:
            assert all(len(components) == len(vf.components) for vf in vec_fields), 'Dimensions do not match'
            coords = vec_fields[0].coords.ravel()
            vel_by_time_axes = []
            for i in range(len(components)):
                vel_by_time_axes.append(
                    np.vstack([vf.vel_axes[i].ravel() for vf in vec_fields]).T
                )
            vel_by_time_axes = tuple(vel_by_time_axes)
        elif vel_by_time_axes is None:
            filepath, coords, vel_by_time_axes, components = self.load_data()
        assert len(components) == len(vel_by_time_axes), 'Dimensions do not match'
        self.filepath = filepath
        self.coords = coords
        self.vel_by_time_axes = tuple(vel_by_time_axes)
        self.components = tuple(components)

    @property
    def timeframes(self):
        """
        The number of timeframes stored in velx_by_time and vely_by_time.

        Returns
        -------
            int
        """
        return self.vel_by_time_axes[0].shape[-1]

    @property
    def timeframe_class(self):
        return Timeframe

    @property
    def vec_field_class(self):
        return VectorField

    @property
    def lib(self):
        """
        The ndarray library used for storing the data fields.

        Returns
        -------
            The torch or numpy module.
        """
        return self.coords.lib

    def load_data(self):
        raise NotImplementedError('This should be overriden.')

    def timeframe(self, time):
        """
        Gets a timeframe of the velx_by_time and vely_by_time.

        Parameters
        ----------
        time: int
            An integer in {0, 1,..., VelocityByTime.timeframes - 1}

        Returns
        -------
            ``Timeframe`` with ``filepath = None``.
        """
        return self.timeframe_class(
            time=time,
            filepath=None,
            vec_field=self.vec_field_class(
                self.coords,
                tuple(a[:, time] for a in self.vel_by_time_axes)
            )
        )

    def shape_as_completable(self, interleaved=True):
        axis_shape = self.vel_by_time_axes[0].shape
        if interleaved:
            return axis_shape[0] * len(self.components), axis_shape[1]
        return axis_shape

    def as_completable(self, interleaved=True):
        if not interleaved:
            return self
        dim = len(self.components)
        completable = self.lib.zeros(self.shape_as_completable(interleaved=interleaved))
        if self.lib.__name__ == 'torch':
            completable = completable.to(device)
        for i, a in enumerate(self.vel_by_time_axes):
            completable[i::dim] = a
        return self.__class__(
            coords=self.coords,
            vel_by_time_axes=(completable,),
            components=auto_component_names(1)
        )

    def transform(self, transform_func, interleaved=True, apply_to_coords=False, keep_interleaved=False):
        """
        Apply a transformation to the completable data fields of AnuersymVelocityByTime.

        Parameters
        ----------
        transform_func: function(ndarray)
            The transformation to apply.

        interleaved: bool, default True
            See :func:`VelocityByTime.shape_as_completable`.

        apply_to_coords: bool, False
            Apply ``transform_func`` to ``VelocityByTime``'s coordinates.

        keep_dims: bool, False
            Return the interleaved ``VelocityByTime`` if ``interleaved=True``.

        Returns
        -------
            ``VelocityByTime``
        """
        vbt = self.as_completable(interleaved=interleaved)
        coords = vbt.coords.transform(transform_func) if apply_to_coords else vbt.coords
        transformed = vbt.__class__(
            filepath=None,
            coords=coords,
            vel_by_time_axes=tuple(transform_func(a) for a in vbt.vel_by_time_axes),
            components=vbt.components
        )
        if not interleaved or interleaved and keep_interleaved:
            return transformed
        dims = len(self.components)
        return self.__class__(
            coords=self.coords,
            vel_by_time_axes=tuple(transformed.vel_by_time_axes[0][i::dims] for i in range(dims)),
            components=self.components
        )

    def torch_to_numpy(self):
        """
        Convert all data from torch tensors to numpy ndarrays.

        Returns
        -------
            ``VelocityByTime`` whose data are numpy ndarrays.
        """
        transform_func = lambda x: x.detach().cpu().numpy()
        return self.transform(transform_func, interleaved=False, apply_to_coords=True)

    def numpy_to_torch(self):
        """
        Convert all data from numpy ndarrays to torch tensors.

        Returns
        -------
            ``VelocityByTime`` whose data are torch tensors.
        """
        transform_func = lambda x: torch.tensor(x).to(device)
        return self.transform(transform_func, interleaved=False, apply_to_coords=True)

    def save(self, path, plot_time=None):
        save = lambda name, arr: np.savetxt(f'{path}_{name}.csv', arr, delimiter=',')
        self.coords.save(path)
        for n, a in zip(self.components, self.vel_by_time_axes):
            save(n, a)
        if plot_time is not None:
            self.timeframe(plot_time).vec_field.save(path, plot=True)


def velocity_by_time_function(func_x, func_y, grid_bounds, grid_density, times=None):
    """
    Helper function to create a ``VelocityByTime`` whose components
    are defined by known functions.

    Parameters
    ----------
    func_x: funcition(t, x, y)
        The function defining the x-component of a vector field. It
        must be vectorized with respect to its spatial coordinates (x, y).

    func_y: funcition(t, x, y)
        The function defining the y-component of a vector field. It
        must be vectorized with respect to its spatial coordinates (x, y).

    grid_bounds: list(tuple(float, float))
        The bounds of the grid to evaluate the vector field functions on.
        The grid is square.

    grid_density: int
        The number of points the grid has along any edge.

    times: list(float), default [0]
        A list of times to evaluate the vector field functions at.

    Returns
    -------
        ``VelocityByTime``
    """
    if times is None:
        times = [0]
    linspace_args = lambda gb: (gb[0], gb[1], grid_density)
    grid_line0 = np.linspace(*linspace_args(grid_bounds[0]))
    grid_line1 = np.linspace(*linspace_args(grid_bounds[1]))
    mesh = np.meshgrid(grid_line0, grid_line1)
    coords = Coordinates(axes=mesh).ravel()
    vec_fields = [VectorField(coords=coords, vel_axes=(func_x(t, *mesh), func_y(t, *mesh))) for t in times]
    return VelocityByTime(coords=coords, vec_fields=vec_fields)


def func1():
    func_x = lambda t, x, y: np.sin(2 * x + 2 * y)
    func_y = lambda t, x, y: np.cos(2 * x - 2 * y)
    return velocity_by_time_function(func_x, func_y, [(-2, 2)] * 2, grid_density=100)


def double_gyre(num_timeframes=11):
    # source: https://shaddenlab.berkeley.edu/uploads/LCS-tutorial/examples.html#Sec7.1
    pi = np.pi
    A = 0.25
    omega = 2 * pi / 10
    epsilon = 0.1
    a = lambda t: epsilon * np.sin(omega * t)
    b = lambda t: 1 - 2 * epsilon * np.sin(omega * t)
    f = lambda x, t: a(t) * x**2 + b(t) * x
    dfdx = lambda x, t: a(t) * 2 * x + b(t)
    # psi = lambda t, x, y: A * np.sin(pi * f(x, t)) * np.sin(pi * y)
    u = lambda t, x, y: -pi * A * np.sin(pi * f(x, t)) * np.cos(pi * y)
    v = lambda t, x, y: pi * A * np.cos(pi * f(x, t)) * np.sin(pi * y) * dfdx(x, t)
    return velocity_by_time_function(u, v, [(0, 2), (0, 1)], grid_density=100, times=range(num_timeframes))


class TimeframeAneurysm(Timeframe):
    def load_data(self):
        """
        Load and preprocess one timeframe of the 2d aneurysm data set.
        Any duplicate spatial coordinates are removed.
        """
        data = pd.read_csv(self.filepath)
        col_idxs, col_names = [0, 1, 3, 4], ['velx', 'vely', 'x', 'y']
        data = data.rename(columns={data.columns[i]: n for i, n in zip(col_idxs, col_names)})
        data = data[col_names] # drop extra columns
        data = data.set_index(['x', 'y']) # set index to (x, y)
        data = data.groupby(level=data.index.names).first() # remove duplicate (x, y)
        self.vec_field = VectorField(
            coords = Coordinates(
                axes=(data.index.get_level_values('x').to_numpy(), data.index.get_level_values('y').to_numpy())
            ),
            vel_axes=(data['velx'].to_numpy(), data['vely'].to_numpy())
        )


class VelocityByTimeAneurysm(VelocityByTime):
    @property
    def timeframe_class(self):
        return TimeframeAneurysm

    def load_data(self):
        # duplicate point at row 39, 41
        # From t = 0
        # x   y   velx      vely
        # 1.9 0.0 -0.000152 -8.057502e-07
        data = pd.read_csv(self.filepath, header=None)
        # data = data.drop_duplicates() # remove 2 duplicate rows
        data = data.to_numpy()
        self.velx_by_time = data[0::2]
        self.vely_by_time = data[1::2]

    @classmethod
    def load_from(cls, path):
        load = lambda n: np.loadtxt(f'{path}_{n}.csv', delimiter=',')
        return cls(
            coords=Coordinates(axes=(load('x'), load('y'))),
            vel_by_time_axes=(load('velx_by_time'), load('vely_by_time'))
        )


@dataclass
class MatrixArora2019(Timeframe):
    def load_data(self):
        velx = torch.load(self.filepath).to(dtype=torch.float64).numpy().ravel()
        width, height = range(velx.shape[0]), range(velx.shape[1])
        coords = Coordinates(axes=np.meshgrid(width, height)).ravel()
        self.vec_field = VectorField(coords=coords, velx=velx, vely=None)

    def saved_mask(self, mask_rate):
        fp = self.filepath
        saved_mask_path = fp.parent / f'{fp.stem}maskrate{mask_rate}.pt'
        (idx_u, idx_v), _ = torch.load(saved_mask_path)
        idx_u, idx_v = idx_u.numpy(), idx_v.numpy()
        mask = np.zeros_like(self.vec_field.velx)
        mask[idx_u, idx_v] = 1
        return mask


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
    coords = coords.ravel()
    func_values = func_values.ravel()
    method = kwargs.pop('method', 'cubic')
    return interp.griddata(coords.axes, func_values, new_coords.axes, method=method, **kwargs)

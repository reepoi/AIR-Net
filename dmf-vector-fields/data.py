from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import scipy.interpolate as interp

import plots


device = 'cuda' if torch.cuda.is_available() else 'cpu'


@dataclass(frozen=True)
class Coordinates:
    x: np.ndarray
    y: np.ndarray
    

    @property
    def lib(self):
        """
        The ndarray library used for storing the data fields.
        
        Returns
        -------
            The torch or numpy module.
        """
        return torch if type(self.x) is torch.Tensor else np
    

    @property
    def components(self):
        return 'x', 'y'


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
        return self.__class__(*(transform_func(getattr(self, c)) for c in self.components))


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
    

    def to_tuple(self):
        """
        Lists out all the data fields of Coordinates.

        Returns
        -------
            A tuple of all the data fields of Coordinates.
        """
        return tuple(getattr(self, c) for c in self.components)
    

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
        mg = lib.meshgrid(*(ls(getattr(self, c)) for c in self.components), indexing='xy')
        return self.__class__(*mg)
    

    def save(self, path):
        save = lambda name, arr: np.savetxt(f'{path}_{name}.csv', arr, delimiter=',')
        for c in self.components:
            save(c, getattr(self, c))


@dataclass(frozen=True)
class Coordinates3D(Coordinates):
    z: np.ndarray

    @property
    def components(self):
        return *super().components, 'z'


@dataclass(frozen=True)
class VectorField:
    coords: Coordinates
    velx: np.ndarray
    vely: np.ndarray
    

    @property
    def lib(self):
        """
        The ndarray library used for storing the data fields.
        
        Returns
        -------
            The torch or numpy module.
        """
        return self.coords.lib
    

    @property
    def components(self):
        return 'velx', 'vely'


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
        return *self.coords.to_tuple(), *(getattr(self, c) for c in self.components)
    

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
        new_components = (interp_griddata(self.coords, getattr(self, c), coords, **interp_opts)
                          for c in self.components)
        return self.__class__(coords, *new_components)


    def as_completable(self, grid_density):
        """
        Returns a representation of a VectorField that can be
        given to a matrix completion algorithm.

        Returns
        -------
            ``VectorField``
        """
        return self.interp(grid_density=grid_density, fill_value=0)
    

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
            self.coords.transform(transform_func) if apply_to_coords else self.coords,
            *(transform_func(getattr(self, c)) for c in self.components)
        )
    

    def save(self, path, plot=True):
        save = lambda name, arr: np.savetxt(f'{path}_{name}.csv', arr, delimiter=',')
        self.coords.save(path)
        for c in self.components:
            save(c, getattr(self, c))
        if plot:
            fig, ax = plt.subplots()
            ax.quiver(*self.to_tuple())
            fig.savefig(f'{path}.png')
            plt.close(fig)


@dataclass(frozen=True)
class VectorField3D(VectorField):
    velz: np.ndarray

    @property
    def components(self):
        return *super().components, 'velz'
    

    def save(self, path, plot=True):
        if plot:
            fig, ax = plt.subplots()
            ax = fig.add_subplot(projection='3d')
            ax.quiver(*self.to_tuple())
            fig.savefig(f'{path}.png')
            plt.close(fig)
        

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
        raise NotImplementedError('This should be overriden.')


    def as_completable(self, grid_density):
        """
        Returns a tuple of velx and vely.

        Returns
        -------
            An ``Timeframe`` with ``filepath=None``.
        """
        return self.__class__(
            time=self.time,
            filepath=None,
            vec_field=self.vec_field.as_completable(grid_density)
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
    filepath_vel_by_time: str
    coords: Coordinates
    velx_by_time: np.ndarray
    vely_by_time: np.ndarray

    def __init__(self, coords=None, filepath_vel_by_time=None, vec_fields=None, **vel_by_time_args):
        self.coords = coords
        self.filepath_vel_by_time = filepath_vel_by_time
        if all(c in vel_by_time_args for c in self.components):
            for c in self.components:
                setattr(self, c, vel_by_time_args[c])
        elif vec_fields is not None:
            assert len(self.components) == len(vec_fields[0].components), 'Mixed dimensions.'
            self.coords = vec_fields[0].coords.ravel()
            for c_vbt, c_vf in zip(self.components, vec_fields[0].components):
                setattr(self, c_vbt, np.vstack([getattr(vf, c_vf).ravel() for vf in vec_fields]).T)
        else:
            self.load_data()
    

    @property
    def timeframes(self):
        """
        The number of timeframes stored in velx_by_time and vely_by_time.

        Returns
        -------
            int
        """
        return self.velx_by_time.shape[-1]
    

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
    

    @property
    def components(self):
        return 'velx_by_time', 'vely_by_time'
    

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
                *(getattr(self, c)[:, time] for c in self.components)
            )
        )
    

    def shape_as_completable(self, interleved=True):
        """
        Returns a matrix for completion.

        Parameters
        ----------
        interleved: bool
            If ``interleved`` is ``True``, this returns a single matrix
            with the rows velx and vely for every time step interleved.
            Otherwise, velx and vely for every timestep are returned separately.
        
        Returns
        -------
            ``VelocityByTime``
        """
        shape = self.velx_by_time.shape
        if interleved:
            shape = (shape[0] * len(self.components), shape[1])
        return shape


    def completable_matrices(self, interleved=True):
        """
        Returns the matrix or matrices that are completable.

        Parameters
        ----------
        interleved: bool, default True
            See :func:`VelocityByTime.shape_as_completable`.

        Returns
        -------
            np.ndarray
        """
        if interleved:
            completable = self.lib.zeros(self.shape_as_completable(interleved=interleved))
            if self.lib.__name__ == 'torch':
                completable = completable.to(device)
            num_components = len(self.components)
            for i, c in enumerate(self.components):
                completable[i::num_components] = getattr(self, c)
            return completable
        return tuple(getattr(self, c) for c in self.components)


    def transform(self, transform_func, interleved=True, apply_to_coords=False):
        """
        Apply a transformation to the completable data fields of AnuersymVelocityByTime.

        Parameters
        ----------
        transform_func: function(ndarray)
            The transformation to apply.
        
        interleved: bool, default True
            See :func:`VelocityByTime.shape_as_completable`.

        apply_to_coords: bool, False
            Apply ``transform_func`` to ``VelocityByTime``'s coordinates.
        
        Returns
        -------
            ``VelocityByTime``
        """
        coords = self.coords.transform(transform_func) if apply_to_coords else self.coords
        if interleved:
            completable = self.lib.zeros(self.shape_as_completable(interleved=interleved))
            if self.lib.__name__ == 'torch':
                completable = completable.to(device)
            num_components = len(self.components)
            for i, c in enumerate(self.components):
                completable[i::num_components] = getattr(self, c)
            transformed = transform_func(completable)
            return self.__class__(
                filepath_vel_by_time=self.filepath_vel_by_time,
                coords=coords,
                **{c: transformed[i::num_components] for i, c in enumerate(self.components)}
            )
        return self.__class__(
            filepath_vel_by_time=self.filepath_vel_by_time,
            coords=coords,
            **{c: transform_func(getattr(self, c)) for c in self.components}
        )
    

    def torch_to_numpy(self):
        """
        Convert all data from torch tensors to numpy ndarrays.
        
        Returns
        -------
            ``VelocityByTime`` whose data are numpy ndarrays.
        """
        transform_func = lambda x: x.detach().cpu().numpy()
        return self.transform(transform_func, interleved=False, apply_to_coords=True)
    

    def numpy_to_torch(self):
        """
        Convert all data from numpy ndarrays to torch tensors.
        
        Returns
        -------
            ``VelocityByTime`` whose data are torch tensors.
        """
        transform_func = lambda x: torch.tensor(x).to(device)
        return self.transform(transform_func, interleved=False, apply_to_coords=True)


    def save(self, path, plot_time=None):
        save = lambda name, arr: np.savetxt(f'{path}_{name}.csv', arr, delimiter=',')
        self.coords.save(path)
        for c in self.components:
            save(c, getattr(self, c))
        if plot_time is not None:
            fig, ax = plt.subplots()
            ax.quiver(*self.timeframe(plot_time).vec_field.to_tuple())
            fig.savefig(f'{path}.png')
            plt.close(fig)


class VelocityByTime3D(VelocityByTime):
    velz_by_time: np.ndarray

    @property
    def components(self):
        return *super().components, 'velz_by_time'
    

    @property
    def vec_field_class(self):
        return VectorField3D


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
    
    grid_bounds: tuple(float, float)
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
    b_x, b_y = grid_bounds
    grid_line = np.linspace(b_x, b_y, grid_density)
    mesh = np.meshgrid(grid_line, grid_line)
    coords = Coordinates(*mesh).ravel()
    vec_fields = [VectorField(coords=coords, velx=func_x(t, *mesh), vely=func_y(t, *mesh)) for t in times]
    return VelocityByTime(coords=coords, vec_fields=vec_fields)


def velocity_by_time_function_3d(func_x, func_y, func_z, grid_bounds, grid_density, times=None):
    """
    Helper function to create a ``VelocityByTime3D`` whose components
    are defined by known functions.

    Parameters
    ----------
    func_x: funcition(t, x, y, z)
        The function defining the x-component of a vector field. It
        must be vectorized with respect to its spatial coordinates (x, y, z).

    func_y: funcition(t, x, y, z)
        The function defining the y-component of a vector field. It
        must be vectorized with respect to its spatial coordinates (x, y, z).

    func_z: funcition(t, x, y, z)
        The function defining the z-component of a vector field. It
        must be vectorized with respect to its spatial coordinates (x, y, z).
    
    grid_bounds: tuple(float, float)
        The bounds of the grid to evaluate the vector field functions on.
        The grid is a cube.
    
    grid_density: int
        The number of points the grid has along any edge.
    
    times: list(float), default [0]
        A list of times to evaluate the vector field functions at.
    
    Returns
    -------
        ``VelocityByTime3D``
    """
    if times is None:
        times = [0]
    b_x, b_y = grid_bounds
    grid_line = np.linspace(b_x, b_y, grid_density)
    mesh = np.meshgrid(grid_line, grid_line, grid_line)
    coords = Coordinates3D(*mesh)
    vec_fields = [VectorField3D(coords=coords, velx=func_x(t, *mesh), vely=func_y(t, *mesh), velz=func_z(t, *mesh)) for t in times]
    return VelocityByTime3D(coords=coords, vec_fields=vec_fields)


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
                x=data.index.get_level_values('x').to_numpy(),
                y=data.index.get_level_values('y').to_numpy()
            ),
            velx=data['velx'].to_numpy(),
            vely=data['vely'].to_numpy()
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
        data = pd.read_csv(self.filepath_vel_by_time, header=None)
        # data = data.drop_duplicates() # remove 2 duplicate rows
        data = data.to_numpy()
        self.velx_by_time = data[0::2]
        self.vely_by_time = data[1::2]


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
    xy = coords.x, coords.y
    new_xy = new_coords.x, new_coords.y
    return interp.griddata(xy, func_values, new_xy, method='linear', **kwargs)

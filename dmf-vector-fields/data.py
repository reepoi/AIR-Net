from dataclasses import dataclass
import numpy as np
import pandas as pd
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
        return Coordinates(x=transform_func(self.x), y=transform_func(self.y))


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
        x, y = lib.meshgrid(
            lib.linspace(lib.min(self.x), lib.max(self.x), grid_density),
            lib.linspace(lib.min(self.y), lib.max(self.y), grid_density),
            indexing='xy'
        )
        return Coordinates(x=x, y=y)
    

    def save(self, path):
        save = lambda name, arr: np.savetxt(f'{path}_{name}.csv', arr, delimiter=',')
        save('x', self.x)
        save('y', self.y)


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


    def ravel(self):
        """
        Flattens the shape of the VectorField's Coordinates,
        and its velx and vely components.

        Returns
        -------
            A new VectorField instance with components whose
            shape is flattened.
        """
        return VectorField(
            coords=self.coords.ravel(),
            velx=self.velx.ravel(),
            vely=self.vely.ravel()
        )
    

    def to_tuple(self):
        """
        Lists out all the data fields of a VectorField.

        Returns
        -------
            A tuple of all the data fields of a VectorField.
        """
        return self.coords.x, self.coords.y, self.velx, self.vely
    

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
        new_velx = interp_griddata(self.coords, self.velx, coords, **interp_opts)
        new_vely = interp_griddata(self.coords, self.vely, coords, **interp_opts)
        return VectorField(coords=coords, velx=new_velx, vely=new_vely)


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
        return VectorField(
            coords=self.coords.transform(transform_func) if apply_to_coords else self.coords,
            velx=transform_func(self.velx),
            vely=transform_func(self.vely)
        )
    

    def save(self, path, plot=True):
        save = lambda name, arr: np.savetxt(f'{path}_{name}.csv', arr, delimiter=',')
        self.coords.save(path)
        save('velx', self.velx)
        save('vely', self.vely)
        if plot:
            fig, _ = plots.quiver(*self.to_tuple(), scale=400, save_path=f'{path}.png')
            plots.plt.close(fig)
        


@dataclass
class AneurysmTimeframe:
    time: int
    filepath: str
    vec_field: VectorField

    def __init__(self, time: int, filepath=None, vec_field=None):
        self.time = time
        self.filepath = filepath
        if filepath is None:
            assert vec_field is not None, 'This must not be None if filepath is None.'
            self.vec_field = vec_field
        else:
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
    

    def as_completable(self, grid_density):
        """
        Returns a tuple of velx and vely.

        Returns
        -------
            An ``AneurysmTimeframe`` with ``filepath=None``.
        """
        return AneurysmTimeframe(
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
            Apply ``transform_func`` to ``AneurysmTimeframe``'s coordinates.
        
        Returns
        -------
            An ``AneurysmTimeframe`` with ``filepath=None``.
        """
        return AneurysmTimeframe(
            time=self.time,
            filepath=None,
            vec_field=self.vec_field.transform(transform_func, apply_to_coords=apply_to_coords)
        )
    

    def torch_to_numpy(self):
        """
        Convert all data from torch tensors to numpy ndarrays.
        
        Returns
        -------
            ``AneurysmTimeframe`` whose data are numpy ndarrays.
        """
        transform_func = lambda x: x.detach().cpu().numpy()
        return self.transform(transform_func, apply_to_coords=True)
    

    def numpy_to_torch(self):
        """
        Convert all data from numpy ndarrays to torch tensors.
        
        Returns
        -------
            ``AneurysmTimeframe`` whose data are torch tensors.
        """
        transform_func = lambda x: torch.tensor(x).to(device)
        return self.transform(transform_func, apply_to_coords=True)
    

    def save(self, path, plot=True):
        self.vec_field.save(path, plot=plot)


@dataclass
class AneurysmVelocityByTime:
    filepath_vel_by_time: str
    coords: Coordinates
    velx_by_time: np.ndarray
    vely_by_time: np.ndarray

    def __init__(self, coords: Coordinates, filepath_vel_by_time=None, velx_by_time=None, vely_by_time=None):
        self.coords = coords
        self.filepath_vel_by_time = filepath_vel_by_time
        if velx_by_time is not None:
            assert vely_by_time is not None, 'Both vel[x,y]_by_time must not be None'
            self.velx_by_time = velx_by_time
            self.vely_by_time = vely_by_time
        else:
            self.load_data()
    

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
    def lib(self):
        """
        The ndarray library used for storing the data fields.
        
        Returns
        -------
            The torch or numpy module.
        """
        return self.coords.lib
    

    def timeframe(self, time):
        """
        Gets a timeframe of the velx_by_time and vely_by_time.

        Parameters
        ----------
        time: int
            An integer in {0, 1,..., AneurysmVelocityByTime.timeframes - 1}
        
        Returns
        -------
            AneurysmTimeframe with ``filepath = None``.
        """
        return AneurysmTimeframe(
            time=time,
            filepath=None,
            vec_field=VectorField(
                coords=self.coords,
                velx=self.velx_by_time[:, time],
                vely=self.vely_by_time[:, time]
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
            ``AneurysmVelocityByTime``
        """
        shape = self.velx_by_time.shape
        if interleved:
            shape = (shape[0] * 2, shape[1])
        return shape
        # if interleved:
        #     num_points = self.velx_by_time.shape[0]
        #     matrix = self.lib.zeros((num_points * 2, self.timeframes))
        #     matrix[0::2] = self.velx_by_time
        #     matrix[1::2] = self.vely_by_time
        #     return matrix
        # return self


    def transform(self, transform_func, interleved=True, apply_to_coords=False):
        """
        Apply a transformation to the completable data fields of AnuersymVelocityByTime.

        Parameters
        ----------
        transform_func: function(ndarray)
            The transformation to apply.
        
        interleved: bool, default True
            See :func:`AneurysmVelocityByTime.as_completable`.

        apply_to_coords: bool, False
            Apply ``transform_func`` to ``AneurysmVelocityByTime``'s coordinates.
        
        Returns
        -------
            ``AneursymVelocityByTime``
        """
        coords = self.coords.transform(transform_func) if apply_to_coords else self.coords
        if interleved:
            num_points = self.velx_by_time.shape[0]
            completable = self.lib.zeros(self.shape_as_completable(interleved=interleved))
            if self.lib.__name__ == 'torch':
                completable = completable.to(device)
            completable[0::2] = self.velx_by_time
            completable[1::2] = self.vely_by_time
            transformed = transform_func(completable)
            return AneurysmVelocityByTime(
                filepath_vel_by_time=self.filepath_vel_by_time,
                coords=coords,
                velx_by_time=transformed[0::2],
                vely_by_time=transformed[1::2]
            )
        return AneurysmVelocityByTime(
            filepath_vel_by_time=self.filepath_vel_by_time,
            coords=coords,
            velx_by_time=transform_func(self.velx_by_time),
            vely_by_time=transform_func(self.vely_by_time)
        )
    

    def torch_to_numpy(self):
        """
        Convert all data from torch tensors to numpy ndarrays.
        
        Returns
        -------
            ``AneurysmVelocityByTime`` whose data are numpy ndarrays.
        """
        transform_func = lambda x: np.ndarray(x.detach().cpu())
        return self.transform(transform_func, interleved=False, apply_to_coords=True)
    

    def numpy_to_torch(self):
        """
        Convert all data from numpy ndarrays to torch tensors.
        
        Returns
        -------
            ``AneurysmVelocityByTime`` whose data are torch tensors.
        """
        transform_func = lambda x: torch.tensor(x).to(device)
        return self.transform(transform_func, interleved=False, apply_to_coords=True)


    def save(self, path, plot_time=None):
        save = lambda name, arr: np.savetxt(f'{path}_{name}.csv', arr, delimiter=',')
        self.coords.save(path)
        save('velx_by_time', self.velx_by_time)
        save('vely_by_time', self.vely_by_time)
        if plot_time:
            fig, _ = plots.quiver(*self.timeframe(plot_time).to_tuple(), scale=400, save_path=f'{path}.png')
            plots.plt.close(fig)



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

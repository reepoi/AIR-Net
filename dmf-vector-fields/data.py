from dataclasses import dataclass
import numpy as np
import pandas as pd
import scipy.interpolate as interp


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


@dataclass(frozen=True)
class Coordinates:
    x: np.ndarray
    y: np.ndarray

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
        return Coordinates(x=self.x.ravel(), y=self.y.ravel())


@dataclass(frozen=True)
class VectorField:
    coords: Coordinates
    velx: np.ndarray
    vely: np.ndarray


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
    

    def interp(self, coords: Coordinates, **interp_opts):
        """
        Uses interpolation to change the grid on which a vector field
        is defined.

        Parameters
        ----------
        coords: Coordinates
            The new grid for the vector field to be defined on.
        
        Returns
        -------
            A VectorField on defined on new Coordinates.
        """
        vec_field = self.vec_field
        new_velx = interp_griddata(vec_field.coords, vec_field.velx, coords, **interp_opts)
        new_vely = interp_griddata(vec_field.coords, vec_field.vely, coords, **interp_opts)
        return VectorField(coords=coords, velx=new_velx, vely=new_vely)


    def as_completable(self):
        """
        Returns a representation of a VectorField that can be
        given to a matrix completion algorithm.

        Returns
        -------
            tuple(np.ndarray, np.ndarray)
        """
        return self.interp()
    

    def transform(self, transform_func, as_completable=True):
        """
        Apply ``transform_func`` to velx and vely.

        Parameters
        ----------
        transform_func: function(matrix)
            Applies a transform to a matrix.
        
        as_completable: bool, default True
            Applies the transform after calling as_completable.
        
        Returns
        -------
            A VectorField
        """
        if as_completable:
            return self.as_completable().transform(transform_func)
        return VectorField(
            coords=self.coords,
            velx=transform_func(self.velx),
            vely=transform_func(self.vely)
        )


@dataclass
class AneurysmTimeframe:
    time: int
    filepath: str
    vec_field: VectorField

    def __init__(self, time: int, filepath=None, vec_field=None):
        self.time = time
        self.filepath = filepath
        if filepath is None:
            assert vec_field is not None, 'This must not be none if filepath is none.'
            self.vec_field = vec_field
        else:
            self.load_data()


    def load_data(self):
        """
        Load and preprocess one timeframe of the 2d aneurysm data set.
        Any duplicate spatial coordinates are removed.
        """
        self.filepath = filepath
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
            )
            velx=velx=data['velx'].to_numpy()
            vely=vely=data['vely'].to_numpy()
        )
    

    def as_completable(self):
        """
        Returns a tuple of velx and vely.

        Returns
        -------
            tuple(np.ndarray, np.ndarray)
        """
        return self.vec_field.as_completable()
    

    def transform(self, transform_func, as_completable=True):
        """
        Applys a transform to velx and vely and returns the result.
        
        transform_func: function(matrix)
            The transform function to apply.

        as_completable: bool, default True
            Applies the transform after calling as_completable.
        
        Returns
        -------
            An AneurysmTimeframe with ``filepath=None``.
        """
        return AneurysmTimeframe(
            time=self.time,
            filepath=None,
            vec_field=self.vec_field.transform(transform_func, as_completable=as_completable)
        )


@dataclass
class AneurysmVelocityByTime(VectorField):
    filepath_vel_by_time: str
    aneurysm_timeframe: AneurysmTimeframe

    def __init__(self, filepath_vel_by_time: str, aneurysm_timeframe: AneurysmTimeframe):
        self.filepath_vel_by_time = filepath_vel_by_time
        self.aneurysm_timeframe = aneurysm_timeframe
        self.load_data()
    

    def load_data(self):
        # duplicate point at row 39, 41
        # From t = 0
        # x   y   velx      vely
        # 1.9 0.0 -0.000152 -8.057502e-07
        data = pd.read_csv(self.filepath_vel_by_time, header=None)
        data = data.drop_duplicates() # remove 2 duplicate rows
        timeframe = self.aneurysm_timeframe.ravel()
        self.coords = self.aneurysm_timeframe.coords,
        self.velx = data[0::2]
        self.vely = data[1::2]

    
    def as_completable1(self):
        """
        Returns a single matrix with velx and vely rows interleved.

        Returns
        -------
            np.ndarray
        """
        num_points, timeframes = self.velx.shape
        matrix = np.zeros((num_points * 2, timeframe))
        matrix[0::2] = velx
        matrix[1::2] = vely
        return matrix
    

    def as_completable(self, interleved=True):
        """
        Returns a matrix for completion. If ``interleved`` is ``True``,
        this returns a single matrix with the rows velx and vely for
        every time step interleved. Otherwise, velx and vely for every
        timestep are returned separately.

        Parameters
        ----------
        interleved: bool
        
        Returns
        -------
            np.ndarray
        """
        if interleved:
            num_points, timeframes = self.velx.shape
            matrix = np.zeros((num_points * 2, timeframe))
            matrix[0::2] = velx
            matrix[1::2] = vely
            return matrix
        return self.velx, self.vely


    def transform(self, transform_func, interleved=True):
    

    def accuracy_report1(self, reconstructed_matrix, epoch, loss, last_report: bool, component_name, ground_truth_matrix=None):
        if ground_truth_matrix is None:
            ground_truth_matrix = self.as_completable1()
        nmae = utils.nmae_against_ground_truth(reconstructed_matrix, ground_truth_matrix)
        print(f'Component: interleved, Epoch: {epoch}, Loss: {loss:.5e}, NMAE (Original): {nmae:.5e}')
        if last_report:
            print(f'\n*** END interleved ***\n')

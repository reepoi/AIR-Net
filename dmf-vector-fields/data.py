import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
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


@dataclass
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


    def as_completable(self):
        """
        Returns a representation of a VectorField that can be
        given to a matrix completion algorithm.

        Returns
        -------
            any
        """
        raise NotImplemented('as_completable must be overriden.')
    


@dataclass
class AneurysmTimeframe(VectorField):
    time: int
    filepath: str

    def __init__(self, time: int, filepath: str):
        self.time = time
        self.filepath = filepath
        self.load_data()

    def load_data(self):
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
        data = pd.read_csv(self.filepath)
        col_idxs, col_names = [0, 1, 3, 4], ['velx', 'vely', 'x', 'y']
        data = data.rename(columns={data.columns[i]: n for i, n in zip(col_idxs, col_names)})
        data = data[col_names] # drop extra columns
        data = data.set_index(['x', 'y']) # set index to (x, y)
        data = data.groupby(level=data.index.names).first() # remove duplicate (x, y)
        self.coords = Coordinates(
            x=data.index.get_level_values('x').to_numpy(),
            y=data.index.get_level_values('y').to_numpy()
        )
        self.velx = velx=data['velx'].to_numpy()
        self.vely = vely=data['vely'].to_numpy()
    

    def as_completable(self):
        """
        Returns a tuple of velx and vely.

        Returns
        -------
            tuple(np.ndarray, np.ndarray)
        """
        return self.velx, self.vely
        


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
        self.velx = data[0::2],
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
    

    def as_completable2(self):
        """
        Returns a tuple of velx and vely.

        Returns
        -------
            tuple(np.ndarray, np.ndarray)
        """
        return self.velx, self.vely
    

    def accuracy_report1(self, reconstructed_matrix, epoch, loss, last_report: bool, component_name, ground_truth_matrix=None):
        if ground_truth_matrix is None:
            ground_truth_matrix = self.as_completable1()
        nmae = utils.nmae_against_ground_truth(reconstructed_matrix, ground_truth_matrix)
        print(f'Component: interleved, Epoch: {epoch}, Loss: {loss:.5e}, NMAE (Original): {nmae:.5e}')
        if last_report:
            print(f'\n*** END interleved ***\n')

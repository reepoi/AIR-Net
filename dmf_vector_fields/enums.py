import enum


class DataSet(enum.Enum):
    ANEURYSM = 'aneurysm'
    FUNC1 = 'func1'
    FUNC2 = 'func2'
    DOUBLE_GYRE = 'double-gyre'
    ARORA2019_5 = 'arora2019-rank5'
    ARORA2019_10 = 'arora2019-rank10'


class Algorithm(enum.Enum):
    IST = 'ist'
    DMF = 'dmf'


class Technique(enum.Enum):
    IDENTITY = 'identity'
    INTERLEAVED = 'interleaved'
    INTERPOLATED = 'interpolated'

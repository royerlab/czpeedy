from __future__ import annotations
from itertools import product
from typing import Iterable, Iterator
from pathlib import Path
import numpy as np
from numpy.typing import ArrayLike
import tensorstore as ts

# A more user-friendly API around tensorstore's `spec` concept (which is usually just json - see `TrialParameters#to_spec`.)
# Contains class methods for utilities related to creating large numbers of trial parameter sets.
class TrialParameters:
    shape: ArrayLike[int]
    chunk_size: list[int]
    dtype: np.dtype
    clevel: int
    compressor: str
    shuffle: int
    endianness: int

    # All parameters are either obvious or can be referenced in tensorstore's spec documentation, with the exception of `endianness`.
    # `endianness` is -1 for little endian, 0 for indeterminate endianness (only applies for 1 byte values), and +1 for big endian.
    def __init__(self, shape: ArrayLike[int], chunk_size: ArrayLike, dtype: np.dtype, *, clevel: int, compressor: str, shuffle: int, endianness: int):
        self.shape = shape
        self.chunk_size = list(chunk_size)
        self.dtype = dtype
        self.clevel = clevel
        self.compressor = compressor
        self.shuffle = shuffle
        self.endianness = endianness
    
    # The name actually varies
    # depending on the driver (i.e. zarr vs zarr3 vs N5...) and endianneess. Currently
    # only supports zarr, but in the future we should support more.

    # Returns a zarr v2 dtype string based on the numpy data type of this `TrialParameters`. 
    # Refernce: https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html#data-type-encoding
    # TODO: This only supports uint16 right now!
    def dtype_json(self) -> str:
        endianness_char = ('|', '>', '<')[self.endianness]
        return endianness_char + "u2"

    # Produces a jsonable dict that communicates all the trial parameters to tensorstore.
    # Usage: `ts.open(trial_parameters.to_spec()).result()`
    def to_spec(self, output_path: Path) -> dict:
        return {
            'driver': 'zarr',
            'kvstore': {
                'driver': 'file',
                'path': str(output_path.absolute()),
            },

            'metadata': {
                'compressor': {
                    'id': 'blosc',
                    'cname': self.compressor,
                    'shuffle': self.shuffle,
                    'clevel': self.clevel
                },
                'dtype': self.dtype_json(),
                'shape': self.shape,
                "chunks": self.chunk_size,
            },
            'create': True,
            'delete_existing': True,
        }
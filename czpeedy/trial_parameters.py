from __future__ import annotations
from itertools import product
from typing import Iterable, Iterator
from pathlib import Path
import numpy as np
from numpy.typing import ArrayLike
import tensorstore as ts

class TrialParameters:
    shape: ArrayLike[int]
    dtype: np.dtype
    clevel: int
    compressor: str

    def __init__(self, shape: ArrayLike[int], dtype: np.dtype, *, clevel: int, compressor: str):
        self.shape = shape
        self.dtype = dtype
        self.clevel = clevel
        self.compressor = compressor
    
    # The name actually varies
    # depending on the driver (i.e. zarr vs zarr3 vs N5...) and endianneess. Currently
    # only supports zarr, but in the future we should support more.
    def dtype_json(self) -> str:
        return "<u2"

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
                    'shuffle': 1,
                    'clevel': self.clevel
                },
                'dtype': self.dtype_json(),
                'shape': self.shape,
                "chunks": [482, 480, 2048],
            },
            'create': True,
            'delete_existing': True,
        }

    # def all_combinations(shape: ArrayLike[int], dtype: np.dtype, clevels: Iterable[int] = [2, 4, 6, 8], compressors: Iterable[str] = ["blosclz", "lz4", "lz4hc", "snappy", "zlib", "zstd"]) -> Iterator[TrialParameters]:
    def all_combinations(shape: ArrayLike[int], dtype: np.dtype, clevels: Iterable[int] = [2], compressors: Iterable[str] = ["blosclz", "lz4", "lz4hc", "snappy", "zlib", "zstd"]) -> Iterator[TrialParameters]:
        for clevel in clevels:
            if clevel < 0 or clevel > 9:
                raise ValueError("clevel must range from 0 to 9")
        
        def to_trial_parameters(clevel: int, compressor: str) -> TrialParameters:
            return TrialParameters(shape, dtype, clevel=clevel, compressor=compressor)
        
        return map(
            lambda args: to_trial_parameters(*args),
            product(iter(clevels), iter(compressors)))
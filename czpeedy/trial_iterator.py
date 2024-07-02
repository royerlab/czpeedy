from __future__ import annotations
from itertools import product
from typing import Iterable, Iterator
from pathlib import Path
import numpy as np
import tensorstore as ts

class TrialParameters:
    clevel: int
    compressor: str

    def __init__(self, *, clevel: int, compressor: str):
        self.clevel = clevel
        self.compressor = compressor
    
    # This is an instance method and not a class method because the name actually varies
    # depending on the driver (i.e. zarr vs zarr3 vs N5...) and endianneess. Currently
    # only supports zarr, but in the future we should support more.
    def name_for_data_type(self, type: np.uint8 | np.uint16) -> str:
        return "<u2"

    def to_spec(self, output_path: Path, data: np.ndarray) -> dict:
        return {
            'driver': 'zarr',
            'kvstore': {
                'driver': 'file',
                'path': output_path,
            },

            'metadata': {
                'compressor': {
                    'id': 'blosc',
                    'cname': self.compressor,
                    'shuffle': 1,
                    'clevel': self.clevel
                },
                'dtype': '<u2',
                'shape': data.shape,
                "chunks": [482, 480, 2048],
            },
            'create': True,
            'delete_existing': True,
            'dtype': self.name_for_data_type(data)
        }

    def all_combinations(clevels: Iterable[int] = [2, 4, 6, 8], compressors: Iterable[str] = ["blosclz", "lz4", "lz4hc", "snappy", "zlib", "zstd"]) -> Iterator[TrialParameters]:
        for clevel in clevels:
            if clevel < 0 or clevel > 9:
                raise ValueError("clevel must range from 0 to 9")
        
        def to_trial_parameters(clevel: int, compressor: str) -> TrialParameters:
            return TrialParameters(clevel=clevel, compressor=compressor)
        
        return map(
            lambda args: to_trial_parameters(*args),
            product(iter(clevels), iter(compressors)))
from __future__ import annotations
from itertools import product
from typing import Iterable, Iterator
from pathlib import Path
import numpy as np
from numpy.typing import ArrayLike
import tensorstore as ts

ALL_COMPRESSORS = ["blosclz", "lz4", "lz4hc", "snappy", "zlib", "zstd"]

class TrialParameters:

    shape: ArrayLike[int]
    dtype: np.dtype
    clevel: int
    compressor: str
    shuffle: int
    chunk_size: list[int]
    endianness: int

    def __init__(self, shape: ArrayLike[int], dtype: np.dtype, *, clevel: int, compressor: str, shuffle: int, chunk_size: ArrayLike[int], endianness: int):
        self.shape = shape
        self.dtype = dtype
        self.clevel = clevel
        self.compressor = compressor
        self.shuffle = shuffle
        self.chunk_size = list(chunk_size)
        self.endianness = endianness
    
    # The name actually varies
    # depending on the driver (i.e. zarr vs zarr3 vs N5...) and endianneess. Currently
    # only supports zarr, but in the future we should support more.
    def dtype_json(self) -> str:
        endianness_char = ('|', '>', '<')[self.endianness]
        return endianness_char + "u2"

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
                "chunks": self.chunk_size,
            },
            'create': True,
            'delete_existing': True,
        }

    # def all_combinations(shape: ArrayLike[int], dtype: np.dtype, clevels: Iterable[int] = [2, 4, 6, 8], compressors: Iterable[str] = ["blosclz", "lz4", "lz4hc", "snappy", "zlib", "zstd"]) -> Iterator[TrialParameters]:
    def all_combinations(
            shape: ArrayLike[int],
            dtype: np.dtype,
            clevels: Iterable[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            compressors: Iterable[str] = ALL_COMPRESSORS,
            shuffles: Iterable[int] = [0, 1, 2],
            chunk_sizes: Iterable[ArrayLike[int]] = None,
            endiannesses: Iterable[int] = [-1, 1]) -> Iterator[TrialParameters]:
        
        if chunk_sizes is None:
            chunk_sizes = [[100] * len(shape)]

        for clevel in clevels:
            if clevel < 0 or clevel > 9:
                raise ValueError("clevel must range from 0 to 9")
        
        for chunk_size in chunk_sizes:
            for ax in chunk_size:
                if ax < 1:
                    raise ValueError("chunk size must be positive in each axis")
        
        for shuffle in shuffles:
            if shuffle < -1 or shuffle > 2:
                raise ValueError("Shuffle must be -1 (automatic), 0 (none), 1 (byte shuffle) or 2 (bit shuffle)")
        
        for compressor in compressors:
            if compressor not in ALL_COMPRESSORS:
                raise ValueError(f"\"{compressor}\" is not a known compressor id")
        
        for endianness in endiannesses:
            if endianness < -1 or endianness > 1:
                raise ValueError(f"Endianness must be -1 (little endian), 0 (neither, i.e. for uint8), or 1 (big endian)")
        
        def to_trial_parameters(clevel: int, compressor: str, shuffle: int, chunk_size: ArrayLike[int], endianess: int) -> TrialParameters:
            return TrialParameters(shape, dtype, clevel=clevel, compressor=compressor, shuffle=shuffle, chunk_size=chunk_size, endianness=endianness)
        
        return map(
            lambda args: to_trial_parameters(*args),
            product(iter(clevels), iter(compressors), iter(shuffles), iter(chunk_sizes), iter(endiannesses)))
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
            chunk_sizes = TrialParameters.suggest_chunk_sizes(shape)

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
        
        count = len(clevels) * len(compressors) * len(shuffles) * len(chunk_sizes) * len(endiannesses)

        def to_trial_parameters(clevel: int, compressor: str, shuffle: int, chunk_size: ArrayLike[int], endianness: int) -> TrialParameters:
            return TrialParameters(shape, dtype, clevel=clevel, compressor=compressor, shuffle=shuffle, chunk_size=chunk_size, endianness=endianness)
        
        return map(
            lambda args: to_trial_parameters(*args),
            product(iter(clevels), iter(compressors), iter(shuffles), iter(chunk_sizes), iter(endiannesses))), count
    
    def suggest_chunk_sizes(shape: ArrayLike[int]) -> list[list[int]]:
        # Concept: The smallest size we reasonably want along an axis is min(axis_size, 100) - 100 is small,
        # so we use 100 as minimum unless axis_size is even smaller.
        # Figure out an integer n such that 100 ~= axis_size / n. Then compute the sequence
        # axis_size / x for x in range (1, n). This forms a sequence of not-absurd chunk sizes along
        # this axis - the smallest will be around 100, the largest will be the full shape of the array,
        # and the spacing ensures that there won't be any crazy wasted space (i.e. if the axis is size 100
        # and the chunk is size 99, you need two chunks of size 99 to cover it. huge waste. But this 
        # method ensures the axis size is always quite close to (i.e. just beneath) a multiple of the chunk).
        #
        # Repeating this for each axis, you can sensibly combine any set of chunk lengths and produce
        # a usable chunk. But the set of all possible combinations is large and not very enlightening.
        # We thus compute each possible chunk size and its volume. Then, we select the smallest chunk
        # and add it to our list of suggestions. From there, we iterate over a sorted list of chunk volumes.
        # If we find a chunk with volume >= 1.5x the last chunk on our suggestion list, we add it to the
        # suggestion list. Finally, we ensure to suggest the largest possible chunk (as it is semantically
        # significant). Because the main read/write speed change caused by varying chunk sizes is the
        # sequential read/write length, using a geometric series like this allows us to test what
        # size is the best without testing thousands of chunk sizes that have very similar volumes.
        def break_axis(axis: int) -> list[int]:
            if axis < 100:
                return [axis]            
            else:
                largest_divisor = int(axis / 100) # => axis ~= 100 * largest_divisor
                chunk_lengths = []
                n = largest_divisor
                while n >= 1:
                    # We either want to exactly divide the axis or divide it into chunks that are more than half full.
                    if axis % n == 0:
                        # chunks |---|---|---|---|
                        #   data |---------------|
                        chunk_lengths.append(int(axis / n))
                    else:
                        # If we use int(axis / n), which rounds down, n * chunk_size will just barely undershoot the needed size.
                        # this forces us to use a chunk with its center of mass outside of the array:
                        #   data --------- (size = 9, int(9 / 2) = 4)
                        # chunks ....oooo.... (we need 3 chunks to cover all 9 data points - so 3 spots are wasted)
                        #
                        # Instead, we round up to int(axis / n). This will reduce the number of chunks we get by one, but make
                        # them slightly bigger. This is therefore always less wasteful:
                        #   data --------- (size = 9, int(9 / 2) + 1 = 5)
                        # chunks .....ooooo (we need 2 chunks to cover all 9 data points - so only 1 spot is wasted)
                        chunk_lengths.append(1 + int(axis / n))

                    # We compute a geometric series in n so that the chunk lengths are well spaced.
                    n = int(n / 1.5)

                return chunk_lengths
        
        chunks = list(product(*[break_axis(axis) for axis in shape]))
        chunks_with_volumes = map(lambda chunk: (chunk, np.prod(chunk)), chunks)
        chunks_with_volumes = sorted(chunks_with_volumes, key=lambda item: item[1])
        
        factor = 1.5
        smallest_chunk, volume_to_beat = chunks_with_volumes[0]
        suggested_chunks = [smallest_chunk]
        volume_to_beat *= factor
        
        for chunk, volume in chunks_with_volumes:
            if volume > volume_to_beat:
                suggested_chunks.append(chunk)
                volume_to_beat = volume * factor
        
        # Test to see the % of wasted space that the suggested chunks create. In my testing
        # this is always less than 2% of the size of the data
        # def waste(shape, chunk):
        #     grid_size = np.uint32(np.array(shape) / np.array(chunk)) * np.array(chunk)
        #     grid_size += (np.array(shape) > grid_size) * np.array(chunk)
        #     return np.prod(grid_size) - np.prod(shape)
        
        # for chunk in suggested_chunks:
        #     print(100 % waste(shape, chunk) / np.prod(shape))

        return suggested_chunks

from typing import Iterable, Iterator
from itertools import product

import numpy as np
from numpy.typing import ArrayLike
from termcolor import colored

from .trial_parameters import TrialParameters

class ParameterSpace:
    ALL_COMPRESSORS = ["blosclz", "lz4", "lz4hc", "snappy", "zlib", "zstd"]
    SHUFFLE_TYPES = {"auto": -1, "none": 0, "byte": 1, "bit": 2}
    ENDIANNESSES = {"big": 1, "auto": 0, "little": -1}

    shape: tuple[int, ...]
    dtype: np.dtype
    clevels: list[int]
    compressors: list[str]
    shuffles: list[str]
    chunk_sizes: list[tuple[int, ...]]
    endiannesses: list[str]
    num_combinations: int

    def __init__(
        self,
        shape: ArrayLike,
        chunk_sizes: Iterable[ArrayLike],
        dtype: np.dtype,
        clevels: Iterable[int] = [1, 2, 3, 5],
        compressors: Iterable[str] = ["blosclz", "lz4", "snappy", "zlib", "zstd"],
        shuffles: Iterable[str] = ["none", "bit", "byte"],
        endiannesses: Iterable[str] = ["big", "little"]):

        for clevel in clevels:
            if clevel < 0 or clevel > 9:
                raise ValueError("clevel must range from 0 to 9")
        
        for chunk_size in chunk_sizes:
            for ax in chunk_size:
                if ax < 1:
                    raise ValueError("chunk size must be positive in each axis")
        
        for shuffle in shuffles:
            if shuffle not in ParameterSpace.SHUFFLE_TYPES:
                raise ValueError(f"Shuffle \"{shuffle}\" is not recognized. Shuffle must be one of {", ".join(ParameterSpace.SHUFFLE_TYPES)}")
        
        for compressor in compressors:
            if compressor not in ParameterSpace.ALL_COMPRESSORS:
                raise ValueError(f"\"{compressor}\" is not a known compressor id")
        
        if dtype.itemsize == 1:
            endiannesses = ["auto"]
        for endianness in endiannesses:
            if endianness not in ParameterSpace.ENDIANNESSES:
                raise ValueError(f"Endianness \"{endianness}\" is not recognized. Endianness must be one of {", ".join(ParameterSpace.ENDIANNESSES)}")
        
        self.shape = tuple(shape)
        self.chunk_sizes = [tuple(chunk_size) for chunk_size in chunk_sizes]
        self.dtype = dtype
        self.clevels = list(clevels)
        self.compressors = list(compressors)
        self.shuffles = list(shuffles)
        self.endiannesses = list(endiannesses)

        self.num_combinations = len(chunk_sizes) * len(clevels) * len(compressors) * len(shuffles) * len(endiannesses)
    
    def summarize(self):
        # print(colored(title, "green") +)
        print(colored("Parameter space", "green") + f" ({self.num_combinations} total tests)")

        fmt_title = lambda title: colored(f"{title:>15} ", "light_grey")
        shape_formatter = lambda shape: "x".join(map(str, shape))

        print(fmt_title("shape") + shape_formatter(self.shape))
        print(fmt_title("chunk sizes") + ", ".join(map(shape_formatter, self.chunk_sizes)))
        print(fmt_title("dtype") + str(self.dtype))
        print(fmt_title("clevels") + ", ".join(map(str, self.clevels)))
        print(fmt_title("compressors") + ", ".join(self.compressors))
        print(fmt_title("shuffles") + ", ".join(map(str, self.shuffles)))
        print(fmt_title("endiannesses") + ", ".join(map(str, self.endiannesses)))
        print(fmt_title("test count") + str(self.num_combinations))


    # Returns:
    # 1. An iterator that steps over every possible set of trial parameters given allowable values for
    #    each parameter (i.e. the cartesian product of all parameters)
    # 2. The number of `TrialParameters` instances that the iterator will produce (i.e. len(list(iterator))).
    #    This is useful because the iterator can in theory be massive and there is no need to convert it to
    #    a list if you're just iterating over it.
    # Parameters that are optional have non-exhaustive domains that try to restrict test count. For example,
    # compression level 9 is not tested as it very likely bottlenecks data at the cpu.
    # Raises ValueError if any parameters are out of bounds.
    def all_combinations(self) -> tuple[Iterator[TrialParameters], int]:
        def to_trial_parameters(clevel: int, compressor: str, shuffle: int, chunk_size: ArrayLike, endianness: int) -> TrialParameters:
            return TrialParameters(
                self.shape,
                chunk_size,
                self.dtype,
                clevel=clevel,
                compressor=compressor,
                shuffle=shuffle,
                endianness=endianness)
        
        return map(
            lambda args: to_trial_parameters(*args),
            product(
                self.clevels,
                self.compressors,
                [ParameterSpace.SHUFFLE_TYPES[shuffle] for shuffle in self.shuffles],
                self.chunk_sizes,
                [ParameterSpace.ENDIANNESSES[endianness] for endianness in self.endiannesses]))
    
    def suggest_chunk_sizes(shape: ArrayLike, itemsize: int, max_bytes = 2**31 - 17, size_ratio = 1.5, volume_ratio = 1.5) -> list[list[int]]:
        # Concept: The smallest size we reasonably want along an axis is min(axis_size, 100) - 100 is small,
        # so we use 100 as minimum unless axis_size is even smaller.
        # Figure out an integer n such that 100 ~= axis_size / n. Then compute the sequence
        # axis_size / x for x in range (1, n). This forms a sequence of not-absurd chunk sizes along
        # this axis - the smallest will be around 100, the largest will be the full shape of the array,
        # and the spacing ensures that there won't be any crazy wasted space (i.e. if the axis is size 100
        # and the chunk is size 99, you need two chunks of size 99 to cover it. huge waste. But this 
        # method ensures the axis size is always quite close to (i.e. just beneath) a multiple of the chunk).
        # (Note: in the implementation below we don't use all of the quotients - just ones that are sufficiently
        # spaced. i.e. not all x in [1, n] are used if their recipocals are close)
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
                    n = int(n / size_ratio)

                return chunk_lengths
        
        chunks = list(product(*[break_axis(axis) for axis in shape]))
        chunks_with_volumes = map(lambda chunk: (chunk, np.prod(chunk)), chunks)
        chunks_with_volumes = sorted(chunks_with_volumes, key=lambda item: item[1])
        
        smallest_chunk, volume_to_beat = chunks_with_volumes[0]
        suggested_chunks = [smallest_chunk]
        volume_to_beat *= volume_ratio
        
        for chunk, volume in chunks_with_volumes:
            if volume > volume_to_beat:
                suggested_chunks.append(chunk)
                volume_to_beat = volume * volume_ratio
        
        # We always want to suggest using the minimum and maximum chunk sizes, as those
        # are the extremes of the sequential write performance spectrum.
        shape = list(shape)
        if shape not in suggested_chunks:
            suggested_chunks.append(shape)

        max_volume = int(max_bytes / itemsize)
        suggested_chunks = list(filter(lambda chunk: np.prod(chunk) < max_volume, suggested_chunks))
        
        # Test to see the % of wasted space that the suggested chunks create. In my testing
        # this is always less than 2% of the size of the data
        # def waste(shape, chunk):
        #     grid_size = np.uint32(np.array(shape) / np.array(chunk)) * np.array(chunk)
        #     grid_size += (np.array(shape) > grid_size) * np.array(chunk)
        #     return np.prod(grid_size) - np.prod(shape)
        
        # for chunk in suggested_chunks:
        #     print(100 % waste(shape, chunk) / np.prod(shape))

        return suggested_chunks
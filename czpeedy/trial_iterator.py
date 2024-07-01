from __future__ import annotations
from itertools import product
from typing import Iterable, Iterator
import tensorstore as ts

class TrialParameters:
    clevel: int

    def __init__(self, *, clevel: int, compressor: str):
        self.clevel = clevel
        self.compressor = compressor
    
    def to_spec() -> TrialParameters:
        return {
            'metadata'
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
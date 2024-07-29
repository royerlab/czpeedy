from __future__ import annotations
from pathlib import Path
import numpy as np
from numpy.typing import ArrayLike


# A more user-friendly API around tensorstore's `spec` concept (which is usually just json - see `TrialParameters#to_spec`.)
# Contains class methods for utilities related to creating large numbers of trial parameter sets.
class TrialParameters:
    shape: ArrayLike[int]
    chunk_size: list[int]
    dtype: np.dtype
    zarr_version: int
    clevel: int
    compressor: str
    shuffle: int
    endianness: int
    output_path: Path

    # All parameters are either obvious or can be referenced in tensorstore's spec documentation, with the exception of `endianness`.
    # `endianness` is -1 for little endian, 0 for indeterminate endianness (only applies for 1 byte values), and +1 for big endian.
    def __init__(
        self,
        shape: ArrayLike[int],
        chunk_size: ArrayLike,
        output_path: Path,
        dtype: np.dtype,
        zarr_version: int,
        clevel: int,
        compressor: str,
        shuffle: int,
        endianness: int,
    ):
        self.shape = shape
        self.chunk_size = list(chunk_size)
        self.output_path = output_path
        self.dtype = dtype
        self.zarr_version = zarr_version
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
    def dtype_json_v2(self) -> str:
        endianness_char = ("|", ">", "<")[self.endianness]
        return endianness_char + "u2"

    # TODO: This only supports uint16 right now!
    def dtype_json_v3(self) -> str:
        return "uint16"

    # Produces a jsonable dict that communicates all the trial parameters to tensorstore.
    # Usage: `ts.open(trial_parameters.to_spec()).result()`
    def to_spec(self) -> dict:
        if self.zarr_version == 2:
            return {
                "driver": "zarr",
                "kvstore": {
                    "driver": "file",
                    "path": str(self.output_path.absolute()),
                },
                "metadata": {
                    "compressor": {
                        "id": "blosc",
                        "cname": self.compressor,
                        "shuffle": self.shuffle,
                        "clevel": self.clevel,
                    },
                    "dtype": self.dtype_json_v2(),
                    "shape": self.shape,
                    "chunks": self.chunk_size,
                },
                "create": True,
                "delete_existing": True,
            }
        elif self.zarr_version == 3:
            #     'driver': 'zarr3',
            #     'kvstore': {
            #         'driver': 'file',
            #         'path': 'D:\\Seth\\tmp/',
            #     },
            #     'metadata': {
            #         "shape": shape,
            #         "chunk_grid": {
            #             "name": "regular",
            #             "configuration": {"chunk_shape": [400, 400, 1024]}
            #             # "configuration": {"chunk_shape": [482, 480, 512]} # doesn't write at all?????
            #         },
            #         "chunk_key_encoding": {"name": "default"},
            #         # "codecs": [{"name": "blosc", "configuration": {"cname": "lz4", "shuffle": "bitshuffle"}}],
            #         "data_type": "uint16",
            #     },
            #
            #     # 'metadata': {
            #     #     'compressor': {
            #     #         'id': 'blosc',
            #     #         'cname': 'lz4',
            #     #         'shuffle': 2
            #     #     },
            #     #     'dtype': '>u2',
            #     #     'shape': shape,
            #     #     # 'blockSize': [100, 100, 100],
            #     # },
            #     'create': True,
            #     'delete_existing': True,
            #     # 'dtype': 'uint16'
            #     },
            #     # Due to a (i think) bug in tensorstore, you have to put the codec separate from the other metadata
            #     # or it fails to merge the expected codecs ([]) and the given codecs (not empty array)
            #     codec=ts.CodecSpec({
            #       "codecs": [{"name": "blosc", "configuration": {"cname": "lz4", "shuffle": "bitshuffle"}}],
            #       'driver': 'zarr3',
            #     })).result()
            return {
                "driver": "zarr3",
                "kvstore": {
                    "driver": "file",
                    "path": str(self.output_path.absolute()),
                },
                "metadata": {
                    "shape": self.shape,
                    "chunk_grid": {
                        "name": "regular",
                        "configuration": {"chunk_shape": self.chunk_size},
                    },
                    "chunk_key_encoding": {"name": "default"},
                    "data_type": self.dtype_json_v3(),
                },
                "create": True,
                "delete_existing": True,
            }

        else:
            raise ValueError(f"Unsupported zarr version: {self.zarr_version}")

    def codecs(self) -> dict:
        if self.zarr_version == 2:
            return {
                "driver": "zarr",
            }
        elif self.zarr_version == 3:
            return {
                "codecs": [
                    {
                        "name": "bytes",
                        "configuration": {
                            "endian": ("little", "big", "little")[self.endianness]
                        },
                    },
                    {
                        "name": "blosc",
                        "configuration": {
                            "cname": self.compressor,
                            "shuffle": ("noshuffle", "shuffle", "bitshuffle")[
                                self.shuffle
                            ],
                            "clevel": self.clevel,
                        },
                    },
                ],
                "driver": "zarr3",
            }

    # A human-readable summary of the trial parameters.
    def summarize(self) -> str:
        endianness = ("auto", "big", "little")[self.endianness]
        shuffle = ("none", "byte", "bit")[self.shuffle]
        shape = "x".join(map(str, self.shape))
        chunk_size = "x".join(map(str, self.chunk_size))
        return f"shape: {shape}, chunk size: {chunk_size}, dtype: {self.dtype}, zarr version: {self.zarr_version}, clevel: {self.clevel}, compressor: {self.compressor}, shuffle: {shuffle}, endianness: {endianness}"

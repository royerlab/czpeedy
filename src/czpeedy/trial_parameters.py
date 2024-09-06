from __future__ import annotations
from typing import Union
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
    def dtype_json_v2(self) -> Union[list, str]:
        # Helper function to parse a regular (not structured) dtype into a zarr v2 dtype string.
        def field_dtype(dtype: np.dtype) -> str:
            kind = dtype.kind
            itemsize = dtype.itemsize

            # If the user didn't specify a byteorder (i.e. if it's still on native byteorder,
            # then we allow the user to specify the endianness. Otherwise, we use the byteorder
            # provided in this specific field - if this is a structured dtype, it might very well
            # be important)
            if dtype.byteorder == "=":
                endianness_char = ("|", ">", "<")[self.endianness]
            else:
                endianness_char = dtype.byteorder
            
            if kind in 'biufc':
                return f"{endianness_char}{kind}{itemsize}"
            elif kind == 'M':
                return f"{endianness_char}M8[{dtype.str[4:-1]}]"
            elif kind == 'm':
                return f"{endianness_char}m8[{dtype.str[4:-1]}]"
            elif kind in 'SU':
                return f"{endianness_char}{kind}{itemsize}"
            elif kind == 'V':
                return f"{endianness_char}V{itemsize}"
            else:
                raise ValueError(f"Unsupported dtype: {self.dtype}")
        
        # Helper function to parse a structured dtype into a zarr v2 compatible dtype object.
        def structured_dtype(self) -> list:
            result = []
            # TODO: Figure out how to pass byte offsets to tensorstore (if at all)?
            for name, (dtype, offset) in self.dtype.fields.items():
                field = [name, self._field_dtype(dtype)]
                if dtype.shape:
                    field.append(list(dtype.shape))
                result.append(field)
            return result
        
        def dtype_str(dtype: np.dtype) -> Union[str, list]:
            if dtype.fields is not None:
                # For the time being, we only support regular dtypes. There isn't really a way to
                # pass a structured dtype to this via the CLI, but we should support it in the future.
                raise ValueError("Structured dtypes are not supported yet.")
                return structured_dtype(dtype)
            else:
                return field_dtype(dtype)

        return dtype_str(self.dtype)

    # TODO: This only supports uint16 right now!
    def dtype_json_v3(self) -> str:
        if self.dtype.kind == 'V':
            return f'r{self.dtype.itemsize * 8}'
        name = self.dtype.name
        if name in ('bool', 'int8', 'int16', 'int32', 'int64', 
                    'uint8', 'uint16', 'uint32', 'uint64',
                    'float16', 'float32', 'float64', 
                    'complex64', 'complex128'):
            return name
        raise ValueError(f"Unsupported dtype: {self.dtype}")

    # Produces a jsonable dict that communicates all the trial parameters to tensorstore.
    # Usage: `ts.open(trial_parameters.to_spec()).result()`
    def to_spec(self) -> dict:
        if self.zarr_version == 2:
            return {
                "driver": "zarr",
                "kvstore": {
                    "driver": "file",
                    "path": str(self.output_path.absolute()),
                    # "file_io_sync": False
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

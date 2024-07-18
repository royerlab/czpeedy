import argparse
import os
from pathlib import Path
from typing import Any, Callable, Tuple

import numpy as np
from termcolor import colored

from czpeedy.runner import Runner
from czpeedy.parameter_space import ParameterSpace


# An argument type for argparse that ensures the argument is a valid directory or a directory that could be created.
# Used to specify the output folder.
def dir_or_nonexistent(path: str) -> Path:
    if os.path.isfile(path):
        raise argparse.ArgumentTypeError(f"{path} is a file, not a directory.")

    return Path(path)


# An argument type for argparse that ensures the argument is an existent directory or file.
# Used to ensure that the input source exists.
def dir_or_file(path: str) -> Path:
    if os.path.exists(path):
        return Path(path)
    else:
        raise argparse.ArgumentTypeError(f"{path} does not exist in the filesystem.")


# An argument type for argparse that ensures the argument is a valid shape for a numpy array (i.e. of form "a,b,c,d,..." where abcd are integers)
# Used to specify the input shape if needed.
def numpy_shape(text: str) -> Tuple[int, ...]:
    axes = text.split("x")

    try:
        return tuple(int(ax) for ax in axes)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"`{text}` is not an x-delimited list of integers."
        )


# An argument type for argparse that ensures the argument is a valid numpy array data type.
def numpy_dtype(text: str) -> np.dtype:
    try:
        return np.dtype(text)
    except TypeError:
        raise argparse.ArgumentTypeError(
            f'Provided dtype string "{text}" is not a valid numpy data type.'
        )


def filepath(text: str) -> Path:
    try:
        return Path(text)
    except Exception:
        raise argparse.ArgumentTypeError(f'Failed to convert "{text}" to a filepath.')


def endianness(text: str) -> int:
    endiannesses = ParameterSpace.ENDIANNESSES
    if text in endiannesses:
        return text
    else:
        raise argparse.ArgumentTypeError(
            f"Failed to convert \"{text}\" to an endianness. Valid endiannesses: {", ".join(endiannesses.keys())}"
        )


def clevel(text: str) -> int:
    try:
        level = int(text)
        if level < 0 or level > 9:
            raise ValueError()
        return level
    except Exception:
        raise argparse.ArgumentTypeError(
            f'Failed to convert "{text}" to a compression level. A valid compression level is an integer from 0 to 9 inclusive.'
        )


def compressor(text: str) -> str:
    compressors = ParameterSpace.ALL_COMPRESSORS
    if text in compressors:
        return text
    raise argparse.ArgumentTypeError(
        f"\"{text}\" is not a valid compressor. Valid compressors: {", ".join(compressors)}."
    )


def shuffle_type(text: str) -> str:
    shuffles = ParameterSpace.SHUFFLE_TYPES
    if text in shuffles:
        return text
    raise argparse.ArgumentTypeError(
        f"\"{text}\" is not a valid shuffle type. Valid shuffle types: {", ".join(shuffles.keys())}"
    )


# Takes all the information that the user provided about the input source and attempts to load it into a numpy array.
# Currently, only raw numpy data files are supported.
def load_input(
    source: Path, shape: list[int] | None = None, dtype: np.dtype | None = None
) -> np.ndarray:
    if source.is_file:
        # Raw numpy data dump (or known type):
        print(f"{colored("Reading input file", "green")} as raw numpy dump")
        if shape is None:
            raise ValueError(
                "Cannot read from a raw numpy data file without knowing the intended data shape. To resolve this, use the --shape flag (--shape x,y,z)"
            )
        if dtype is None:
            raise ValueError(
                "Cannot read from a raw numpy data file without knowing the intended data type. To resolve this, use the --dtype flag (--dtype uint32)"
            )
        with open(source, "rb") as f:
            return np.fromfile(f, dtype=dtype).reshape(shape)
    else:
        raise NotImplementedError("Loading from zarr is not yet supported.")


# Given a callable that can be used as a type in argparse (i.e. it can convert a string to a more specific type),
# produces a wrapper that accepts a *list* of that type. i.e. if the type 'endianness' works as follows:
# --endianness big => args.endianness = +1
# then using the type `list_type(endianness)` produces a type with these semantics:
# --endianness big => args.endianness = [+1]
# --endianness big,little => args.endianness = [+1, -1]
# This is useful for specifying a set of parameters for each given property.
def list_type(element_type: Callable[[str], Any]) -> set[Any]:
    def parser(text: str) -> list[Any]:
        return {element_type(part) for part in text.split(",")}

    return parser


def zarr_version(text: str) -> int:
    try:
        version = int(text)
        if version < 2 or version > 3:
            raise ValueError()
        return version
    except ValueError:
        raise argparse.ArgumentTypeError(
            f'Failed to convert "{text}" to a zarr version. Valid zarr versions: 2, 3.'
        )


# Runs the main CLI. Currently, only write testing is implemented.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source",
        type=dir_or_file,
        help="The input dataset used in benchmarking. If write benchmarking, this is the data that will be written to disk.",
    )
    parser.add_argument(
        "--dest",
        type=dir_or_nonexistent,
        help="The destination where write testing will occur. A directory will be created inside, called 'czpeedy'. Each write test will delete and recreate the `czpeedy` folder.",
    )
    parser.add_argument(
        "--savecsv",
        type=filepath,
        help="The destination to save test results to in csv format. Will overwrite the named file if it exists already.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="The number of times to test each configuration. This increases confidence that speeds are repeatable, but takes a while. (default: 3)",
    )
    parser.add_argument(
        "--dtype",
        type=numpy_dtype,
        help="If your data source is a raw numpy array dump, you must provide its dtype (i.e. --dtype uint32)",
    )
    parser.add_argument(
        "--shape",
        type=numpy_shape,
        help="If your data source is a raw numpy array dump, you must provide the shape (i.e. --shape 1920x1080x1024). Ignored if the data source has a shape in its metadata.",
    )
    parser.add_argument(
        "--clevel",
        type=list_type(clevel),
        help='The endianness you want to write your data as (can be big, little, or none). "none" is only an acceptable endianness if the dtype is 1 byte.',
    )
    parser.add_argument(
        "--compressor",
        type=list_type(compressor),
        help=f"The compressor id you want to use with blosc. Valid compressors: {", ".join(ParameterSpace.ALL_COMPRESSORS)}.",
    )
    parser.add_argument(
        "--shuffle",
        type=list_type(shuffle_type),
        help=f"The shuffle mode you want to use with blosc compression. Valid shuffle types: {", ".join(ParameterSpace.SHUFFLE_TYPES.keys())}",
    )
    parser.add_argument(
        "--chunk-size",
        type=list_type(numpy_shape),
        help="The chunk size that tensorstore should use when writing data. i.e. --chunk-size 100x100x100. Must have the same number of dimensions as the source data.",
    )
    parser.add_argument(
        "--endianness",
        type=list_type(endianness),
        help='The endianness you want to write your data as (can be big, little, or none). "none" is only an acceptable endianness if the dtype is 1 byte.',
    )
    parser.add_argument(
        "--zarr-version",
        type=list_type(zarr_version),
        help="The version of zarr to use. (Supported: 2, 3.)",
    )
    args = parser.parse_args()

    if args.dest:
        print(
            f"{colored("Beginning write testing", "green")} (from {args.source} to {args.dest})"
        )
        data = load_input(args.source, args.shape, args.dtype)
        if args.chunk_size is None:
            args.chunk_size = ParameterSpace.suggest_chunk_sizes(
                data.shape, data.itemsize
            )

        parameter_space = ParameterSpace(
            data.shape,
            args.chunk_size,
            data.dtype,
            args.zarr_version,
            args.clevel,
            args.compressor,
            args.shuffle,
            args.endianness,
        )
        parameter_space.summarize()

        args.dest.mkdir(parents=True, exist_ok=True)
        runner = Runner(
            parameter_space.all_combinations(),
            data,
            args.dest,
            args.repetitions,
            parameter_space.num_combinations,
        )
        try:
            runner.run_all()
        except KeyboardInterrupt:
            print(
                colored(
                    "\nCtrl-C detected mid-test - Printing partial results and terminating.",
                    "black",
                    "on_red",
                )
            )

        print()
        print(colored("Fastest Specs:", "green"))
        runner.print_results()
        if args.savecsv:
            print(f"{colored("Saving results", "green")} as {args.savecsv}")
            runner.save_results_csv(args.savecsv)
    else:
        print("Read testing is not yet supported")


if __name__ == "__main__":
    main()

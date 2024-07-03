import argparse
import os
from pathlib import Path

import numpy as np
import colorama
from termcolor import colored

from czpeedy.trial_parameters import TrialParameters
from czpeedy.runner import Runner

def dir_or_nonexistent(path: str) -> Path:
    if os.path.isfile(path):
        raise argparse.ArgumentTypeError(f"{path} is a file, not a directory.")
    
    return Path(path)
    
def dir_or_file(path: str) -> Path:
    if os.path.exists(path):
        return Path(path)
    else:
        raise argparse.ArgumentTypeError(f"{path} does not exist in the filesystem.")

def numpy_shape(text: str) -> list[int]:
    axes = text.split(',')
    out = []
    try:
        for ax in axes:
            out.append(int(ax))
        return out
    except ValueError:
        raise argparse.ArgumentTypeError(f"`{text}` is not a comma-delimited list of integers.")

def load_input(source: Path, shape: list[int] | None = None, dtype: np.dtype | None = None) -> np.ndarray:
    if source.is_file:
        # Raw numpy data dump (or known type):
        print(f"{colored("Reading input file", "green")} as raw numpy dump")
        if shape is None:
            raise ValueError("Cannot read from a raw numpy data file without knowing the intended data shape. To resolve this, use the --shape flag (--shape x,y,z)")
        if dtype is None:
            raise ValueError("Cannot read from a raw numpy data file without knowing the intended data type. To resolve this, use the --dtype flag (--dtype uint32)")
        with open(source, 'rb') as f:
            return np.fromfile(f, dtype=dtype).reshape(shape)
    else:
        raise NotImplementedError("Loading from zarr is not yet supported.")

def numpy_dtype(text: str) -> np.dtype:
    try:
        return np.dtype(text)
    except TypeError:
        raise argparse.ArgumentTypeError(f"Provided dtype string \"{text}\" is not a valid numpy data type.")

def filepath(text: str) -> Path:
    try:
        return Path(text)
    except:
        raise argparse.ArgumentTypeError(f"Failed to convert \"{text}\" to a filepath.")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=dir_or_file, help="The input dataset used in benchmarking. If write benchmarking, this is the data that will be written to disk.")
    parser.add_argument("--dest", type=dir_or_nonexistent, help="The destination where write testing will occur. A directory will be created inside, called 'czpeedy'. Each write test will delete and recreate the `czpeedy` folder.")
    parser.add_argument("--shape", type=numpy_shape, help="If your data source is a raw numpy array dump, you must provide the shape (i.e. --shape 1920,1080,1024). Ignored if the data source has a shape in its metadata.")
    parser.add_argument("--dtype", type=numpy_dtype, help="If your data source is a raw numpy array dump, you must provide its dtype (i.e. --dtype uint32)")
    parser.add_argument("--repetitions", type=int, default=3, help="The number of times to test each configuration. This increases confidence that speeds are repeatable, but takes a while. (default: 3)")
    parser.add_argument("--savecsv", type=filepath, help="The destination to save test results to in csv format. Will overwrite the named file if it exists already.")
    args = parser.parse_args()

    if args.dest:
        print(f"{colored("Beginning write testing", "green")} (from {args.source} to {args.dest})")
        data = load_input(args.source, args.shape, args.dtype)
        trial_parameters = TrialParameters.all_combinations(data.shape, data.dtype)
        args.dest.mkdir(parents=True, exist_ok=True)
        runner = Runner(trial_parameters, data, args.dest, args.repetitions)
        try:
            runner.run_all()
        except KeyboardInterrupt:
            print(colored("Ctrl-C detected - Printing partial results and terminating.", "black", "on_red"))
        print(colored("Fastest Specs:", "green"))

        runner.print_results()
        if args.savecsv:
            print(f"{colored("Saving results", "green")} as {args.savecsv}")
            runner.save_results_csv(args.savecsv)
    else:
        print("Read testing is not yet supported")

if __name__ == "__main__":
    main()
else:
    print("main.py is intended to be used as a command line tool. Import czpeedy/czpeedy.py to use this as a library.")
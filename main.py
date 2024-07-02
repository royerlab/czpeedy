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

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=dir_or_file, help="The input dataset used in benchmarking. If write benchmarking, this is the data that will be written to disk.")
    parser.add_argument("--dest", type=dir_or_nonexistent, help="The destination where write testing will occur. A directory will be created inside, called 'czpeedy'. Each write test will delete and recreate the `czpeedy` folder.")
    args = parser.parse_args()

    if args.dest:
        print(f"{colored("Beginning write testing", "green")} (from {args.source} to {args.dest})")
        args.dest.mkdir(parents=True, exist_ok=True)
        trial_parameters = TrialParameters.all_combinations()
        data = np.zeros((1920, 1440, 2048), np.uint16)
        runner = Runner(trial_parameters, data, args.dest)
        runner.run_all()
    else:
        print("Read testing is not yet supported")

if __name__ == "__main__":
    main()
else:
    print("main.py is intended to be used as a command line tool. Import czpeedy/czpeedy.py to use this as a library.")
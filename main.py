import argparse
import os
from pathlib import Path

import colorama
from termcolor import colored

from czpeedy.trial_iterator import TrialParameters

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
        for trial_parameters in TrialParameters.all_combinations():
            print(trial_parameters)
    else:
        print("Read testing is not yet supported")

if __name__ == "__main__":
    main()
else:
    print("main.py is intended to be used as a command line tool. Import czpeedy/czpeedy.py to use this as a library.")
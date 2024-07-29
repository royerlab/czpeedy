from typing import Iterable
from pathlib import Path
import time
import csv

import numpy as np
import tensorstore as ts
from termcolor import colored

from .trial_parameters import TrialParameters


# Test runner class that performs a write speed measurement of some number of `TrialParameters` (provided by an iterable).
# Pretty-prints output while running, reports the best parameters, and can save data to file.
class Runner:
    trial_params: Iterable[TrialParameters]
    data: np.ndarray
    repetitions: int
    batch_count: int
    results: dict[TrialParameters, list[float]]

    def __init__(
        self,
        trial_params: Iterable[TrialParameters],
        data: np.ndarray,
        repetitions: int,
        batch_count: int | None = None,
    ):
        self.trial_params = trial_params
        self.data = data
        self.repetitions = repetitions
        self.batch_count = batch_count
        self.results = {}

    # Given a tensorstore dataset object (i.e. the result of ts.open(spec).result()),
    # performs a write test and returns the duration in seconds.
    def time_execution(self, dataset) -> float:
        now = time.time()
        dataset.write(self.data).result()
        elapsed = time.time() - now
        return elapsed

    # Runs a number of repeat tests for every instance of trial_parameters (referred to as a "batch"),
    # pretty-prints results while doing so, and collects all the data in the results dict.
    def run_all(self):
        is_first_loop = True
        best_time = None
        # Ensure there is always enough justification to fit "Test x/x" when x = self.repetitions
        rjust_level = 10 + 2 * (int(np.log10(self.repetitions)) + 1)

        for batch_id, trial_param in enumerate(self.trial_params):
            result = []
            spec = trial_param.to_spec()
            codecs = ts.CodecSpec(trial_param.codecs())

            dataset = ts.open(spec, codec=codecs).result()

            if is_first_loop:
                print(
                    f"{colored("Warming up...", "green")} (The very first write is often 2x slower than expected, so czpeedy discards it)"
                )
                self.time_execution(dataset)
                is_first_loop = False

            print()
            print(
                f"{colored(f"Starting Test Batch {batch_id + 1}" + (f"/{self.batch_count}" if self.batch_count else ""), "green")}{f" ({100*batch_id/self.batch_count:.1f}%)" if self.batch_count else ""}"
            )
            print(colored(trial_param.summarize(), "light_grey"))
            for n in range(self.repetitions):
                print(f"Test {n + 1}/{self.repetitions}: ".rjust(rjust_level), end="")
                elapsed = self.time_execution(dataset)
                print(f"{elapsed:.2f}s")
                result.append(elapsed)

            mean = np.mean(result)
            print("Mean: ".rjust(rjust_level) + f"{np.mean(result):.2f}s", end="")

            if best_time is not None and mean < best_time:
                print(colored(" (Fastest yet 🏆)", "yellow"))
                best_time = mean
            else:
                print()

            if best_time is None:
                best_time = mean

            self.results[trial_param] = result

    def print_results(self, topn=3):
        sorted_results = sorted(self.results.items(), key=lambda item: np.mean(item[1]))

        # Print the topn results in ascending order so that the user sees the #1 spec
        # at the bottom of the terminal output
        for i, (trial_param, timings) in reversed(
            list(enumerate(sorted_results[:topn]))
        ):
            mean_time = np.mean(timings)
            stddev = np.std(timings)

            print(colored("🥇" if i == 0 else f"#{i + 1}", "green"), end="")
            print(
                f": Mean Runtime: {mean_time:.2f}s "
                + (f"(σ={int(1000*stddev)}ms)" if self.repetitions > 1 else "")
            )
            print(colored(trial_param.summarize(), attrs=["bold"]))
            print()

    def save_results_csv(self, path: Path):
        with open(path, "wt") as fp:
            writer = csv.writer(fp, delimiter=",")

            # Write the header
            writer.writerow(
                [
                    "mean write time (s)",
                    "write std.dev (s)",
                    "dtype",
                    "zarr version",
                    "clevel",
                    "compressor",
                    "shuffle",
                    "chunk size",
                    "endianness",
                    "spec json",
                    "codec json",
                ]
            )
            for trial_param, timings in self.results.items():
                writer.writerow(
                    [
                        f"{np.mean(timings):.2f}",
                        f"{np.std(timings):.2f}",
                        trial_param.dtype_json_v2() if trial_param.zarr_version == 2 else trial_param.dtype_json_v3(),
                        trial_param.zarr_version,
                        trial_param.clevel,
                        trial_param.compressor,
                        trial_param.shuffle,
                        ",".join(str(ax) for ax in trial_param.chunk_size),
                        trial_param.endianness,
                        trial_param.to_spec(self.dest),
                        trial_param.codecs(),
                    ]
                )

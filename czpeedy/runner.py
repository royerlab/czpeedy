from typing import Iterable
from pathlib import Path
import time
import csv

import numpy as np
import tensorstore as ts
from termcolor import colored

from .trial_parameters import TrialParameters

class Runner:
    trial_params: Iterable[TrialParameters]
    data: np.ndarray
    dest: Path
    repetitions: int
    batch_count: int
    results: dict[TrialParameters, list[float]]

    def __init__(self, trial_params: Iterable[TrialParameters], data: np.ndarray, dest: Path, repetitions: int, batch_count: int | None = None):
        self.trial_params = trial_params
        self.data = data
        self.dest = dest
        self.repetitions = repetitions
        self.batch_count = batch_count
        self.results = {}
    
    def time_execution(self, dataset) -> float:
        now = time.time()
        dataset.write(self.data).result()
        elapsed = time.time() - now
        return elapsed

    def run_all(self):
        is_first_loop = True
        best_time = None
        # Ensure there is always enough justification to fit "Test x/x" when x = self.repetitions
        rjust_level = 10 + 2 * (int(np.log10(self.repetitions)) + 1)

        for (batch_id, trial_param) in enumerate(self.trial_params):
            result = []
            spec = trial_param.to_spec(self.dest)
            dataset = ts.open(spec).result()

            if is_first_loop:
                print(f"{colored("Warming up...", "green")} (The very first write is often 2x slower than expected, so czpeedy discards it)")
                self.time_execution(dataset)
                is_first_loop = False

            print()
            print(f"{colored(f"Starting Test Batch {batch_id + 1}" + (f"/{self.batch_count}" if self.batch_count else ""), "green")}{f" ({100*batch_id/self.batch_count:.1f}%)" if self.batch_count else ""}")
            print(colored(spec, "light_grey"))
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

        for i, (trial_param, timings) in reversed(list(enumerate(sorted_results[:topn]))):
            mean_time = np.mean(timings)
            stddev = np.std(timings)
            spec = trial_param.to_spec(self.dest)
            
            print()
            print(colored("🥇" if i == 0 else f"#{i + 1}", "green"), end="")
            print(f": Mean Runtime: {mean_time:.2f}s " + (f"(σ={int(1000*stddev)}ms)" if self.repetitions > 1 else ""))
            print(spec)
    
    def save_results_csv(self, path: Path):
        with open(path, "wt") as fp:
            writer = csv.writer(fp, delimiter=",")

            # Write the header
            writer.writerow(["mean write time (s)", "write std.dev (s)", "dtype", "clevel", "compressor", "spec json"])
            for trial_param, timings in self.results.items():
                writer.writerow([
                    f"{np.mean(timings):.2f}",
                    f"{np.std(timings):.2f}",
                    trial_param.dtype_json(),
                    trial_param.clevel,
                    trial_param.compressor,
                    trial_param.to_spec(self.dest)
                ])
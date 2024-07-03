from typing import Iterable
from pathlib import Path
import time

import numpy as np
import tensorstore as ts
from termcolor import colored

from .trial_parameters import TrialParameters

class Runner:
    results: dict[TrialParameters, list[float]]
    data: np.ndarray
    dest: Path
    repetitions: int

    def __init__(self, trial_params: Iterable[TrialParameters], data: np.ndarray, dest: Path, repetitions: int):
        self.trial_params = trial_params
        self.data = data
        self.dest = dest
        self.repetitions = repetitions
        self.results = {}
    
    def time_execution(self, dataset) -> float:
        now = time.time()
        dataset.write(self.data).result()
        elapsed = time.time() - now
        return elapsed

    def run_all(self):
        is_first_loop = True
        best_time = None

        for trial_param in self.trial_params:
            result = []
            spec = trial_param.to_spec(self.dest)
            dataset = ts.open(spec).result()

            if is_first_loop:
                print(f"{colored("Warming up...", "green")} (The very first write is often 2x slower than expected, so czpeedy discards it)")
                self.time_execution(dataset)
                is_first_loop = False

            print(f"{colored("Starting Test Batch", "green")} {str(spec)}")
            for n in range(self.repetitions):
                print(f"\tTest {n + 1}/{self.repetitions}: ", end="")
                elapsed = self.time_execution(dataset)
                print(f"{elapsed:.2f}s")
                result.append(elapsed)
            
            mean = np.mean(result)
            print(f"\t Mean: {np.mean(result):.2f}s", end="")

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
            print(f"\tMean Runtime: {mean_time:.2f}s " + (f"(σ={int(1000*stddev)}ms)" if self.repetitions > 1 else ""))
            print(spec)
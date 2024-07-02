from typing import Iterable
from pathlib import Path
import time

import numpy as np
import tensorstore as ts

from .trial_parameters import TrialParameters

class Runner:
    def __init__(self, trial_params: Iterable[TrialParameters], data: np.ndarray, dest: Path | None):
        self.trial_params = trial_params
        self.data = data
        self.dest = dest
    
    def run_all(self):
        for trial_param in self.trial_params:
            spec = trial_param.to_spec(self.dest, self.data)
            dataset = ts.open(spec).result()
            now = time.time()
            dataset.write(self.data).result()
            elapsed = time.time() - now

            print(spec)
            print(elapsed)
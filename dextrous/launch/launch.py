import random

import ezpyzy as ez
import dataclasses as dc
import textwrap as tw
import random as rng
import pathlib as pl
import subprocess as sp

from dextrous.experiment import LlamaTrackerExperimentTable, T5TrackerExperimentTable

experiment_path = pl.Path('slurm')
experiment_path.mkdir(exist_ok=True)
existing_experiment_names = {
    file.stem for file in experiment_path.iterdir() if file.suffix in {'.csv', '.json'}
}

@ez.settings
class T5Launch(T5TrackerExperimentTable):
    def __post_init__(self):
        super().__post_init__()
        if self.experiment() is None:
            self.experiment(ez.denominate(existing_names=existing_experiment_names))
        ex = self.experiment()
        self.settings['experiment'] = self.experiment()
        ez.File(f'slurm/{ex}.json').save(self.settings)
        print(f'Launching experiment: {ex}')
        sp.Popen(
            f'sbatch --job-name={ex} --output=out/{ex}.out launch.sh {ex}',
            shell=True
        )

@ez.settings
class LlamaLaunch(LlamaTrackerExperimentTable):
    def __post_init__(self):
        super().__post_init__()
        if self.experiment() is None:
            self.experiment(ez.denominate(existing_names=existing_experiment_names))
        ex = self.experiment()
        self.settings['experiment'] = self.experiment()
        ez.File(f'slurm/{ex}.json').save(self.settings)
        print(f'Launching experiment: {ex}')
        sp.Popen(
            f'sbatch --job-name={ex} --output=out/{ex}.out launch.sh {ex}',
            shell=True
        )
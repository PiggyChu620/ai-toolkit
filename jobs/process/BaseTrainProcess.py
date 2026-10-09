import random
from datetime import datetime
import os
from collections import OrderedDict
from typing import TYPE_CHECKING, Union

import torch
import yaml

from jobs.process.BaseProcess import BaseProcess

if TYPE_CHECKING:
    from jobs import TrainJob, BaseJob, ExtensionJob
    from torch.utils.tensorboard import SummaryWriter
    from tqdm import tqdm


class BaseTrainProcess(BaseProcess):

    def __init__(
            self,
            process_id: int,
            job,
            config: OrderedDict
    ):
        super().__init__(process_id, job, config)
        self.process_id: int
        self.config: OrderedDict
        self.writer: 'SummaryWriter'
        self.job: Union['TrainJob', 'BaseJob', 'ExtensionJob']
        self.progress_bar: 'tqdm' = None

        self.training_seed = self.get_conf('training_seed', self.job.training_seed if hasattr(self.job, 'training_seed') else None)
        # if training seed is set, use it
        if self.training_seed is not None:
            torch.manual_seed(self.training_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.training_seed)
            random.seed(self.training_seed)

        self.progress_bar = None
        self.writer = None
        self.training_folder = self.get_conf('training_folder',
                                             self.job.training_folder if hasattr(self.job, 'training_folder') else None)
        self.save_root = os.path.join(self.training_folder, self.name)
        self.step = 0
        self.first_step = 0
        self.log_dir = self.get_conf('log_dir', self.job.log_dir if hasattr(self.job, 'log_dir') else None)
        self.setup_tensorboard()
        self.save_training_config()

    def run(self):
        super().run()
        # implement in child class
        # be sure to call super().run() first
        pass

    # def print(self, message, **kwargs):
    def print(self, *args):
        if self.progress_bar is not None:
            self.progress_bar.write(' '.join(map(str, args)))
            self.progress_bar.update()
        else:
            print(*args)

    def setup_tensorboard(self):
        if self.log_dir:
            from torch.utils.tensorboard import SummaryWriter
            now = datetime.now()
            time_str = now.strftime('%Y%m%d-%H%M%S')
            summary_name = f"{self.name}_{time_str}"
            summary_dir = os.path.join(self.log_dir, summary_name)
            self.writer = SummaryWriter(summary_dir)

    def save_training_config(self):
        os.makedirs(self.save_root, exist_ok=True)

        # This file is an OUTPUT, never an input. The UI launches run.py with
        # .job_config.json, regenerated from aitk_db.db on every start; nothing in
        # the tree ever reads this dump back. Hand-editing it therefore changes
        # nothing, silently -- and the next run overwrites the edit. Warn loudly
        # instead of clobbering, because a silent discard is exactly what makes it
        # look like the trainer is ignoring a setting.
        save_dif = os.path.join(self.save_root, 'config.used.yaml')
        legacy = os.path.join(self.save_root, 'config.yaml')
        if os.path.exists(legacy):
            try:
                with open(legacy, 'r') as f:
                    previous = yaml.safe_load(f)
            except Exception:
                previous = None
            if previous is not None and previous != self.job.raw_config:
                print(
                    f"\nWARNING: {legacy} differs from the config this run was "
                    f"launched with, and it is NOT an input -- it is a dump of the "
                    f"previous run. Any edits to it were ignored. Change the setting "
                    f"in the UI (it lives in aitk_db.db and is written to "
                    f".job_config.json at launch).\n"
                )

        with open(save_dif, 'w') as f:
            f.write("# AUTO-GENERATED record of the config this run was launched with.\n")
            f.write("# Editing this file does NOTHING: it is an output, never an input.\n")
            f.write("# The live config lives in the UI / aitk_db.db and reaches the\n")
            f.write("# trainer via .job_config.json, regenerated on every start.\n")
            yaml.dump(self.job.raw_config, f)

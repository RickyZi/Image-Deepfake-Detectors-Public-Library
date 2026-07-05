'''
Copyright 2024 Image Processing Research Group of University Federico
II of Naples ('GRIP-UNINA'). All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

from .dataset import create_dataloader        # standard dataset loader
from .training import TrainingModel            # frozen-backbone fc-only trainer
from .finetuning import FTModel                # LoRA / block-unfreeze trainer


class EarlyStopping:
    """
    Tracks the best validation score seen so far.

    Returns True from __call__ when the new score is strictly better than
    best + delta, i.e. when the caller should save a new best checkpoint.
    Sets self.early_stop = True after `patience` consecutive non-improving
    calls, signalling the training loop to drop the LR or stop.
    """

    def __init__(self, init_score=None, patience=1, verbose=False, delta=0, logger = None):
        self.best_score = init_score
        self.patience   = patience
        self.delta      = delta
        self.verbose    = verbose
        self.count_down = patience
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            if self.verbose:
                print(f"EarlyStopping: initial score set to {score:.6f}")
            self.best_score = score
            self.count_down = self.patience
            return True

        if score > self.best_score + self.delta:
            if self.verbose:
                print(f"EarlyStopping: score improved {self.best_score:.6f} → {score:.6f}")
                if self.logger:
                    self.logger.info(f"EarlyStopping: score improved {self.best_score:.6f} → {score:.6f}")
            self.best_score = score
            self.count_down = self.patience
            return True
        else:
            self.count_down -= 1
            if self.verbose:
                print(f"EarlyStopping: no improvement "
                      f"(count_down={self.count_down}/{self.patience})")
                if self.logger:
                    self.logger.info(f"EarlyStopping: no improvement "
                      f"(count_down={self.count_down}/{self.patience})")
            if self.count_down <= 0:
                self.early_stop = True
            return False

    def reset_counter(self):
        self.count_down = self.patience
        self.early_stop = False
# -*- coding: utf-8 -*-
"""
 Recipe for training the FastSpeech2 Text-To-Speech model 
 
 Authors
 * Ayush Agarwal 2021
"""

from os import name
from recipes.TTS.common.utils import PretrainedModelMixin, ProgressSampleImageMixin
import torch
import speechbrain as sb
import sys
import logging
from hyperpyyaml import load_hyperpyyaml
from speechbrain.lobes.models.synthesis.fastspeech2 import dataio_prepare


logger = logging.getLogger(__name__)

class FastSpeech2Brain(sb.Brain, PretrainedModelMixin, ProgressSampleImageMixin):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_progress_samples()
        self.last_epoch = 0
        self.last_batch = None
        self.last_loss_stats = {}
    
    def compute_forward(self, batch, stage):
        """ Computes the Forward Pass 
            
            Arguments
            -----------
            batch : a single batch
            stage : speechbrain.stage
            -----------
            Returns : model output
        """
        return super().compute_forward(batch, stage)



if __name__ == "__main__":

    # Load Hyperparameters file with command-line overrides

    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    
    # Plots Results Every N Iterations
    show_results_every = 5

    # Create Experiment Directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Dataset Preparation 
    # Here we create the datasets objects
    datasets = dataio_prepare(hparams)

    # Brain Class Initialization
    fastspeech2_brain = FastSpeech2Brain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Training
    fastspeech2_brain.fit(
        fastspeech2_brain.hparams.epoch_counter,
        datasets["train"],
        datasets["valid"],
    )

    if hparams.get("save_for_pretrained"):
        fastspeech2_brain.save_for_pretrained()
    
    # Test
    if "test" in datasets:
        fastspeech2_brain.evaluate(datasets["test"])


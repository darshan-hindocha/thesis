# Generic CNF/NODE Trainer Experiment

## Goal

[**User to complete: Describe the specific model (likely a generic CNF/NODE setup) and research goal of this experiment. This script seems to be a more general-purpose trainer from the original `ffjord` repository.**]

This script (`train_generic.py`) appears to be a general-purpose training script for Continuous Normalizing Flows or Neural ODE models, likely adapted from the original FFJORD project. It can be configured for various datasets and model parameters.

## How to Run

[**User to complete: Provide specific command-line instructions to run `train_generic.py`. Include any important arguments or link to a relevant script in `example_runner_scripts/`.**]

Example for CIFAR-10 (refer to `example_runner_scripts/cifar10.sh` if available):
```bash
python train_generic.py \
    --data cifar10 \
    --imagesize 32 \
    --datadir /path/to/your/cifar10_dataset \
    --dims 64,64,64 \
    --num_blocks 10 \
    --lr 0.001 \
    --batch_size 128 \
    --save experiments/generic_trainer/results_cifar10 \
    # ... other relevant arguments (see script or example runners)
```

Example for CelebA-HQ (refer to `example_runner_scripts/celeba.sh` if available):
```bash
python train_generic.py \
    --data celebahq \
    --imagesize 256 \
    --datadir /path/to/your/celeba_hq_dataset \
    # ... other arguments similar to above, adjusted for CelebA-HQ
```

(Adjust `--datadir`, dataset-specific arguments, and other parameters as needed.)

## Notes

[**User to complete: Add any specific configurations, dataset preprocessing notes if not covered in the main README, or other relevant information for this experiment.**]
* This script is highly configurable via command-line arguments. Refer to the script's argument parser for all options.
* It supports various ODE solvers and regularization techniques. 
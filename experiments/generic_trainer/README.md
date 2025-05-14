# Generic CNF/NODE Trainer Experiment

## Goal

This script (`train_generic.py`) serves as a general-purpose trainer for Continuous Normalizing Flows (CNFs) or Neural Ordinary Differential Equation (NODE) models. It is adapted from the original FFJORD project and allows for flexible configuration across various datasets (e.g., CIFAR-10, CelebA-HQ, MNIST, SVHN) and model parameters. Its main research utility in the context of the thesis (Chapter 1, Section 1.3) is to train and evaluate standalone CNF models. This allows for:
1.  Density estimation on complex datasets.
2.  Generation of samples from the learned distributions.
3.  Establishing baseline performance for CNFs, which can then be compared against hybrid models like CNF-GANs to assess the impact of adversarial training.

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
* This script is highly configurable via command-line arguments. Refer to the script's argument parser (`parser = argparse.ArgumentParser()`) for a comprehensive list of all options, including choices for ODE solvers, network dimensions (`dims`), number of blocks (`num_blocks`), regularization techniques, and more.
* It supports various datasets like CIFAR-10, MNIST, CelebA-HQ, etc., selectable via the `--data` argument.
* When comparing results, ensure consistent settings for shared parameters (e.g., CNF architecture details) if this script is used to generate a baseline for a CNF-GAN experiment. 
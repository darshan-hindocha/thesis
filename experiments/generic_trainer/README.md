# Generic CNF/NODE Trainer Experiment

## Goal

This script (`train_generic.py`) serves as a general-purpose trainer for Continuous Normalizing Flows (CNFs) or Neural Ordinary Differential Equation (NODE) models. It is adapted from the original FFJORD project and allows for flexible configuration across various datasets (e.g., CIFAR-10, CelebA-HQ, MNIST, SVHN) and model parameters. Its main research utility in the context of the thesis (Chapter 1, Section 1.3) is to train and evaluate standalone CNF models. This allows for:
1.  Density estimation on complex datasets.
2.  Generation of samples from the learned continuous distribution.
3.  Providing baseline performance metrics for CNFs (e.g., bits per dimension, sample quality) for comparison with CNF-GANs.
4.  Investigating the impact of different ODE solvers, regularization techniques (e.g., kinetic energy, Jacobian regularization), and architectural choices within the CNF framework.

## How to Run

To run this experiment, execute the `train_generic.py` script. Ensure your environment is set up and datasets are available as per the main project `README.md`.

Example for CIFAR-10 (refer to `example_runner_scripts/cifar10.sh` if available, or adapt general arguments):
```bash
# From the project root directory
python experiments/generic_trainer/train_generic.py \
    --data cifar10 \
    --imagesize 32 \
    --datadir /path/to/your/cifar10_dataset \
    --dims 64,64,64 \
    --num_blocks 10 \
    --lr 0.001 \
    --batch_size 128 \
    --save experiments/generic_trainer/results_cifar10 \
    # Add other arguments as needed (e.g., --solver, --alpha, regularization flags); consult --help
```

Adjust `--data`, `--datadir`, image/model parameters, and training parameters as required for your specific dataset and experimental setup. Use `python experiments/generic_trainer/train_generic.py --help` for a full list of available arguments.

## Notes

*   **Versatility:** This script is highly configurable and can be used for various datasets beyond CIFAR-10, including MNIST, SVHN, CelebA-HQ, etc., by changing the `--data` argument and other relevant parameters (like `--imagesize`, `--dims`).
*   **Training Objective:** It primarily focuses on likelihood-based training (optimizing bits per dimension).
*   **Regularization:** The script supports various regularization techniques for CNFs, such as kinetic energy, Jacobian norm, etc., which can be enabled via specific command-line flags. These are important for stabilizing training and improving generalization, as discussed in the FFJORD paper and related works.
*   **Solvers:** Different ODE solvers can be specified, impacting performance and computational cost.
*   **Checkpoints:** Model checkpoints, logs, and generated samples are saved to the directory specified by `--save`.

* This script is highly configurable via command-line arguments. Refer to the script's argument parser (`parser = argparse.ArgumentParser()`) for a comprehensive list of all options, including choices for ODE solvers, network dimensions (`dims`), number of blocks (`num_blocks`), regularization techniques, and more.
* It supports various datasets like CIFAR-10, MNIST, CelebA-HQ, etc., selectable via the `--data` argument.
* When comparing results, ensure consistent settings for shared parameters (e.g., CNF architecture details) if this script is used to generate a baseline for a CNF-GAN experiment. 
# Project Title (Thesis: Neural ODEs and Conditional Normalizing Flows for Generative Modeling)

## Overview

[**User to complete: Briefly describe the overall project goal and its context within your thesis. What are you trying to achieve or investigate?**]

This project explores ... using techniques like Neural Ordinary Differential Equations (NODEs) and Conditional Normalizing Flows (CNFs) for generative modeling tasks.

## Project Structure

The project is organized as follows:

*   `core_lib/`: Contains the core library code, including:
    *   `layers/`: Implementations of various neural network layers, including those specific to CNFs and NODEs.
    *   `networks/`: Network architectures, such as U-Net.
    *   `utils.py`: General utility functions.
    *   `distributed_utils.py`: Utilities for distributed training.
    *   `datasets.py`: Dataset handling and loading.
    *   `odenvp.py`: Implementation related to ODE-NVP models.
*   `experiments/`: Contains specific experimental setups:
    *   `celeba_cnf/`: CNF experiments on the CelebA dataset.
    *   `celeba_cnfg_gan/`: CNF-GAN experiments on the CelebA dataset.
    *   `generic_trainer/`: A more generic training script for NODE-based models.
    *   `mnist_cnfg_gan/`: CNF-GAN experiments on the MNIST dataset.
*   `example_runner_scripts/`: Example shell scripts to run various experiments.
*   `preprocessing/`: Scripts or notebooks for data preprocessing. (May be empty if preprocessing is part of dataset loading or not extensive).
*   `PROJECT_READABILITY_REFACTOR.md`: The refactoring plan document.
*   `requirements.txt`: Python dependencies.
*   `README.md`: This file.

## Setup and Installation

1.  **Python Version**: This project is developed with Python 3.x (e.g., Python 3.8+ recommended).
2.  **Create a Virtual Environment** (recommended):
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Running Experiments

[**User to complete: Provide clear instructions on how to run key experiments. You can reference scripts in `example_runner_scripts/` or give direct command-line examples.**]

For example, to run the MNIST CNF-GAN experiment:
```bash
cd experiments/mnist_cnfg_gan/
python train_cnfg_gan_mnist.py --your-arguments
```

Refer to the scripts in `example_runner_scripts/` for more detailed examples for different datasets and models. For instance, `example_runner_scripts/mnist.sh` (if it exists) might contain:
```bash
# example_runner_scripts/mnist.sh (illustrative)
python experiments/mnist_cnfg_gan/train_cnfg_gan_mnist.py \
    --batch_size 64 \
    --lr 0.0001 \
    # ... other relevant arguments
```

## Datasets

[**User to complete: Describe the datasets used (e.g., MNIST, CelebA, CIFAR-10). Mention where they are expected to be located or if they are downloaded automatically. Detail any significant preprocessing steps found in `preprocessing/` or within the dataset loading code itself.**]

*   **MNIST**: ...
*   **CelebA**: ...
*   ...

## Contributing

[**User to complete: Optional section if others might contribute or if you have guidelines for future development.**]

## License

[**User_to_complete: Specify the license, e.g., MIT License.**] 
# Project Title (Thesis: Neural ODEs and Conditional Normalizing Flows for Generative Modeling)

## Overview

This project, forming the basis of a Master of Science thesis, investigates Continuous Normalizing Flow-based Generative Adversarial Networks (CNF-GANs). The primary goals are:
1.  To explore and provide an intuition-driven discussion of differential equations and their relevance to Neural Ordinary Differential Equations (NODEs), aiming to make these concepts more accessible.
2.  To develop and test a CNF-GAN, combining the benefits of continuous normalizing flows (likelihood-based training, reversible transformations) with the sample quality advantages of GANs (adversarial training).
3.  Initially, to demonstrate that such a CNF-GAN can successfully train on the MNIST dataset for unsupervised image generation.
4.  Subsequently (if successful), to compare the CNF-GAN against standalone CNFs, traditional GANs, and other flow-GAN variants in terms of sample quality and latent space interpolation.

This project explores advanced generative modeling techniques using Neural Ordinary Differential Equations (NODEs) and Conditional Normalizing Flows (CNFs).

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

To run experiments, navigate to the specific experiment's subdirectory within `experiments/` and execute its main training script. General command structure often involves specifying dataset locations, learning rates, batch sizes, and save paths.

For example, to run the MNIST CNF-GAN experiment (see `experiments/mnist_cnfg_gan/README.md` for more details):
```bash
cd experiments/mnist_cnfg_gan/
python train_cnfg_gan_mnist.py \
    --data mnist \
    --datadir /path/to/your/mnist_data \
    --save experiments/mnist_cnfg_gan/results \
    # Add other necessary arguments, e.g., --lr, --batch_size
```

For the CelebA CNF-GAN experiment (see `experiments/celeba_cnfg_gan/README.md` for more details):
```bash
cd experiments/celeba_cnfg_gan/
python train_cnfg_gan_celeb.py \
    --dataset celeba --data_root /path/to/your/celeba_data \
    --weights_root experiments/celeba_cnfg_gan/weights \
    --logs_root experiments/celeba_cnfg_gan/logs \
    --samples_root experiments/celeba_cnfg_gan/samples \
    # Add other necessary arguments
```

Please refer to the example scripts in `example_runner_scripts/` (if available) and the individual `README.md` files within each experiment's directory for more specific commands and argument details. The training scripts (e.g., `train_generic.py`, `train_cnfg_gan_mnist.py`) also contain argument parsers that list all available options.

## Datasets

[**User to complete: Describe the datasets used (e.g., MNIST, CelebA, CIFAR-10). Mention where they are expected to be located or if they are downloaded automatically. Detail any significant preprocessing steps found in `preprocessing/` or within the dataset loading code itself.**]

This project primarily focuses on the following datasets for generative modeling tasks:

*   **MNIST**: A dataset of handwritten digits (28x28 pixels, grayscale). It's often downloaded automatically by PyTorch if not found in the specified `datadir`.
*   **CelebA / CelebA-HQ**: A large-scale dataset of celebrity faces. CelebA-HQ is a high-quality version (e.g., 256x256 pixels). You will need to provide the path to your local copy of these datasets via the `--datadir` or `--data_root` arguments in the respective training scripts.

The training scripts are also configurable for other datasets such as:
*   **CIFAR-10**
*   **SVHN**
*   **LSUN Church**
*   **ImageNet64**

Preprocessing steps are generally handled within the dataset loading code in `core_lib/datasets.py` or the specific training scripts, often including resizing, normalization, and potentially dequantization (adding noise and scaling to [0,1]). Refer to the respective scripts for details. The `preprocessing/` directory may contain additional or specific preprocessing utilities if used.

## Contributing

[**User to complete: Optional section if others might contribute or if you have guidelines for future development.**]

## License

[**User_to_complete: Specify the license, e.g., MIT License.**]
The copyright of the original thesis text (contained in `Darshan Thesis.txt`) rests with its author, Darshan Hindocha, as per the declaration in the document. For the codebase itself, please choose an appropriate open-source license (e.g., MIT License, Apache 2.0) if you intend to share it publicly. If no license is specified, all rights are typically reserved. 
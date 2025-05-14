# Thesis: Neural ODEs and Conditional Normalizing Flows for Generative Modeling

## Overview

This project, forming the basis of a Master of Science thesis, investigates Continuous Normalizing Flow-based Generative Adversarial Networks (CNF-GANs). The primary goals are:
1.  To explore and provide an intuition-driven discussion of differential equations and their relevance to Neural Ordinary Differential Equations (NODEs), aiming to make these concepts more accessible.
2.  To develop and test a CNF-GAN, combining the benefits of continuous normalizing flows (likelihood-based training, reversible transformations) with the sample quality advantages of GANs (adversarial training).
3.  Initially, to demonstrate that such a CNF-GAN can successfully train on the MNIST dataset for unsupervised image generation.
4.  Subsequently (if successful), to compare its performance (e.g., sample quality, likelihoods, training stability) against standalone CNFs and GANs, potentially on more complex datasets like CelebA or ImageNet, as detailed in the thesis (Chapter 1, Section 1.3; Chapter 4; Chapter 5).

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

The `example_runner_scripts/` directory contains template shell scripts (e.g., `cifar10.sh`, `mnist.sh`) that demonstrate how to run the training scripts with various configurations. Adapt these scripts to your environment and dataset paths.

For individual experiment scripts, refer to their respective `README.md` files in `experiments/<experiment_name>/`. Generally, you can run them using:

```bash
# Activate your virtual environment first
# e.g., source .venv/bin/activate

# For scripts not structured as modules yet, from the project root:
python experiments/<experiment_name>/<script_name.py> --arg1 value1 --arg2 value2 ...

# For scripts runnable as modules (recommended for consistent imports), from the project root:
python -m experiments.<experiment_name>.<script_name> --arg1 value1 --arg2 value2 ...
```

Consult the `--help` flag on any script for a full list of its arguments.

## Datasets

This project primarily focuses on the following datasets for generative modeling tasks:

*   **MNIST**: A dataset of handwritten digits (28x28 pixels, grayscale). It's often downloaded automatically by PyTorch if not found in the specified `datadir`.
*   **CelebA / CelebA-HQ**: A large-scale dataset of celebrity faces. CelebA-HQ is a high-quality version (e.g., 256x256 pixels). You will need to provide the path to your local copy of these datasets via the `--datadir` or `--data_root` arguments in the respective training scripts.

The training scripts are also configurable for other datasets such as:
*   **CIFAR-10**
*   **SVHN**
*   **LSUN Church**
*   **ImageNet64**

Preprocessing steps are generally handled within the dataset loading code in `core_lib/datasets.py` or the specific training scripts, often including resizing, normalization, and potentially dequantization (adding noise and scaling to [0,1]). Refer to the respective scripts for details. The `preprocessing/` directory may contain additional or specific preprocessing utilities if used.

## Key Components

See the thesis document (`Thesis.pdf` or `Thesis.txt`) for a comprehensive discussion of the theoretical underpinnings, model architectures, and experimental results.

## Future Work and Extensions

Based on the findings and the framework established in this thesis, potential future research directions include:

*   **Exploring advanced ODE solvers:** Investigate the impact of more sophisticated or adaptive ODE solvers on training stability, speed, and generation quality.
*   **Alternative network architectures:** Experiment with different neural network architectures for both the generator (CNF) and the discriminator, including attention mechanisms or transformer-based models.
*   **Conditional generation enhancements:** Further develop conditional CNF-GANs for more fine-grained control over the generation process (e.g., class-conditional generation on more complex datasets, or text-to-image synthesis).
*   **Theoretical analysis:** Conduct a more in-depth theoretical analysis of the hybrid loss function and the interplay between likelihood-based and adversarial training in CNF-GANs.
*   **Scalability to higher-resolution images:** Address challenges in scaling these models to generate higher-resolution images, potentially exploring multi-scale architectures or patch-based approaches.
*   **Applications to other domains:** Apply CNF-GANs and related NODE-based generative models to other data modalities, such as time-series data, audio, or video.
*   **Improved regularization techniques:** Develop novel regularization methods tailored for CNFs within an adversarial training loop to further improve sample quality and training stability.

## Contributing

Contributions to this project are welcome! If you have suggestions, bug reports, or would like to contribute code, please follow these general guidelines:

1.  **Fork the repository.**
2.  **Create a new branch** for your feature or bug fix (e.g., `feature/my-new-feature` or `fix/issue-number`).
3.  **Make your changes** and ensure they adhere to the project's coding style (if any established).
4.  **Test your changes thoroughly.**
5.  **Submit a pull request** with a clear description of your changes and why they are needed.

Please open an issue first to discuss any significant changes or new features.

## License

This project is licensed under the MIT License. See the `LICENSE.md` file (to be created) for details. You are free to use, modify, and distribute this code as permitted by the license.
It is recommended to create a `LICENSE.md` file in the root of your project containing the full text of the MIT license. For example:

```markdown
MIT License

Copyright (c) [Year] [Your Name/Organization]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

(Remember to replace `[Year]` and `[Your Name/Organization]`.) 
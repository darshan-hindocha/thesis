# MNIST CNF-GAN Experiment

## Goal

This experiment is central to the thesis (Chapter 1, Section 1.3; Chapter 5, Section 5.1) and aims to develop and test a Conditional Normalizing Flow GAN (CNFG-GAN) on the MNIST dataset. The specific goals are:
1.  To prove by example that a CNF-GAN, which combines a CNF generator with a GAN framework (e.g., DCGAN-style discriminator), can successfully train for unsupervised image generation on MNIST.
2.  To investigate the effectiveness of a hybrid training objective that incorporates both maximum likelihood estimation (via the CNF's exact log-likelihood calculation) and adversarial loss.
3.  To compare the sample quality (e.g., using FID, Inception Score if applicable, or visual inspection) and training dynamics of the CNFG-GAN against baseline pure CNF and pure GAN models on MNIST.
4.  To serve as a foundational testbed for the CNF-GAN concept before attempting to scale it to more complex datasets.

## How to Run

To run this experiment, execute the `train_cnfg_gan_mnist.py` script. Ensure your environment is set up and datasets are available as per the main project `README.md`.

Example command (ensure you are in the project root and your virtual environment is activated):
```bash
# From the project root directory
python -m experiments.mnist_cnfg_gan.train_cnfg_gan_mnist \
    --data mnist \
    --imagesize 28 \
    --datadir ./data/mnist_dataset_download_location \
    --batch_size 128 \
    --lr 0.0002 \
    --num_epochs 100 \
    --save experiments/mnist_cnfg_gan/results \
    --log_freq 100 \
    # Add other arguments as needed; consult --help
```

Key arguments to consider:
*   `--datadir`: Path to MNIST dataset (will be downloaded if not present).
*   `--save`: Directory to save results, logs, and checkpoints.
*   `--lr`, `--d-lr`, `--g-lr`: Learning rates for the combined, discriminator, and generator optimizers respectively (the script uses `--lr` for the CNF part when in hybrid/likelihood mode, and `--d-lr`/`--g-lr` for GAN components).
*   `--batch_size`: Training batch size.
*   `--num_epochs`: Total number of training epochs.
*   `--training_type`: Can be 'adv' (adversarial), 'lik' (likelihood), or 'hyb' (hybrid), controlling the loss computation.

Use `python -m experiments.mnist_cnfg_gan.train_cnfg_gan_mnist --help` for a full list of arguments.

## Notes

*   **Dataset:** MNIST is typically downloaded automatically by `torchvision.datasets.MNIST` if not found in the specified `--datadir`.
*   **Architectures:** The generator is a CNF model, and the discriminator is a relatively simple Convolutional Neural Network suitable for MNIST (as defined in `core_lib/networks.py` as `Generator` and `Discriminator` respectively, though `Generator` here refers to the CNF itself).
*   **Training Modes:** The `--training_type` argument is crucial for selecting how the model is trained (adversarial, likelihood, or hybrid), which is a key aspect of the thesis investigation.
*   **Logging:** Training progress, including losses (BPD, adversarial), number of function evaluations (NFE), etc., is logged to the console and a CSV file in the `--save` directory.
*   **Outputs:** Model checkpoints (`checkpt.pth`, `best.pth`) and generated image samples (in a `figs` subdirectory) are saved in the directory specified by `--save`.
*   **Runnability:** As of the last tests, this script runs on CPU (after modifications to remove direct CUDA calls if PyTorch isn't CUDA-enabled). It may encounter internal model errors related to tensor shapes during the forward pass (see previous conversational context if debugging). Using `--nworkers 0` in the command was a temporary workaround for a pickling issue with the DataLoader; a more permanent fix would be to move the `fast_collate` function to the global scope.

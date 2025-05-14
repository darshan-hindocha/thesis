# CelebA CNF-GAN Experiment

## Goal

This experiment trains a Conditional Normalizing Flow GAN (CNFG-GAN) model, potentially leveraging an RNODE-based CNF as the generator and a U-Net architecture as the discriminator, on the CelebA (or CelebA-HQ) dataset. The primary objective is to combine the likelihood-based training and reversible properties of CNFs with the strong sample generation capabilities of GANs. This addresses one of the core research ideas of the thesis: to investigate if such a hybrid model can produce high-quality, realistic images by leveraging both adversarial and likelihood-based training signals (Thesis Chapter 4, Section 4.1; Chapter 5, Section 5.2).

## How to Run

To run this experiment, execute the `train_cnfg_gan_celeb.py` script. Ensure your environment is set up and datasets are available as per the main project `README.md`.

Example command:
```bash
# From the project root directory
python experiments/celeba_cnfg_gan/train_cnfg_gan_celeb.py \
    --dataset celeba --data_root /path/to/your/celeba_dataset \
    --batch_size 32 \
    --G_lr 0.0001 --D_lr 0.0004 \
    --save_every 500 --test_every 2000 \
    --num_epochs 200 \
    --weights_root experiments/celeba_cnfg_gan/weights \
    --logs_root experiments/celeba_cnfg_gan/logs \
    --samples_root experiments/celeba_cnfg_gan/samples \
    # Add other arguments as needed; consult --help
```

Adjust dataset paths (`--data_root`), learning rates, batch sizes, and save locations as needed. You can refer to `example_runner_scripts/` for more detailed examples if available, or use `python experiments/celeba_cnfg_gan/train_cnfg_gan_celeb.py --help` for a full list of arguments.

## Notes

*   **Dataset:** This script is typically configured for the CelebA dataset. Ensure it is downloaded and accessible via `--data_root`.
*   **Architectures:** The generator is a CNF (potentially RNODE-based), and the discriminator is often a U-Net, especially for image-to-image tasks or when detailed spatial information is crucial. The specific configurations will be in the script.
*   **Training Mode:** The script likely supports different training modes (adversarial-only, likelihood-only, hybrid). Refer to script arguments for controlling this (e.g., `--training_type` if adapted from other scripts).
*   **Computational Resources:** Training CNF-GANs, especially with U-Nets, on CelebA can be very computationally demanding. Adjust batch size, model complexity, and image resolution accordingly.
*   **Checkpoints & Outputs:** Training progress, model weights, logs, and generated samples are typically saved to directories specified by arguments like `--weights_root`, `--logs_root`, and `--samples_root`.

```bash
python train_cnfg_gan_celeb.py \
    --dataset celeba --data_root /path/to/your/celeba_dataset \
    --batch_size 32 \
    --G_lr 0.0001 --D_lr 0.0004 \
    --save_every 500 --test_every 2000 \
    --num_epochs 200 \
    --weights_root experiments/celeba_cnfg_gan/weights \
    --logs_root experiments/celeba_cnfg_gan/logs \
    --samples_root experiments/celeba_cnfg_gan/samples \
    # ... other relevant arguments (see script or example runners)
```

(Adjust `--data_root` and other parameters as needed.)

## Notes

* This experiment utilizes the `train_cnfg_gan_celeb.py` script.
* The generator is a Continuous Normalizing Flow (CNF), possibly an RNODE variant, and the discriminator can be a U-Net (as explored in the thesis) or a more standard DCGAN-style architecture.
* Key hyperparameters include learning rates for both generator (G_lr) and discriminator (D_lr), batch size, and the specifics of the CNF and discriminator architectures.
* The thesis mentions experiments with an RNODE-U-Net on CelebHQ (Thesis Figure 5.3), so ensure dataset paths and any specific U-Net configurations (e.g., from `core_lib/networks/u_net/`) are correctly set.
* Pay attention to the balance between adversarial loss and any likelihood-based loss/regularization if implementing a hybrid training objective.
* This experiment likely uses the U-Net architecture defined in `core_lib/networks/u_net/`. 
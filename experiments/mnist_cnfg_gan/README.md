# MNIST CNF-GAN Experiment

## Goal

[**User to complete: Describe the specific model and research goal of this experiment using a Conditional Normalizing Flow GAN (CNFG-GAN) on the MNIST dataset.**]

This experiment trains a Conditional Normalizing Flow GAN (CNFG-GAN) model on the MNIST dataset. It appears to use a simpler Generator and Discriminator architecture suitable for MNIST, distinct from the U-Net used in the CelebA CNFG-GAN experiment.

## How to Run

[**User to complete: Provide specific command-line instructions to run `train_cnfg_gan_mnist.py`. Include any important arguments or link to a relevant script in `example_runner_scripts/`.**]

```bash
python train_cnfg_gan_mnist.py \
    --data mnist \
    --imagesize 28 \
    --datadir /path/to/your/mnist_dataset \
    --batch_size 128 \
    --lr 0.0002 \
    --num_epochs 100 \
    --save experiments/mnist_cnfg_gan/results \
    # ... other relevant arguments (see script or example runners)
```

(Adjust `--datadir` and other parameters as needed.)

## Notes

[**User to complete: Add any specific configurations, dataset preprocessing notes if not covered in the main README, or other relevant information for this experiment.**]
* The Generator and Discriminator architectures are likely defined within `core_lib/networks.py` (or were originally in `lib/networks.py`).
* Utility functions previously in `train_misc.py` and `dist_utils.py` within this directory have been moved to `core_lib/utils.py` and `core_lib/distributed_utils.py` respectively. 
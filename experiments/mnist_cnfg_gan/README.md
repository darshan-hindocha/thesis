# MNIST CNF-GAN Experiment

## Goal

This experiment is central to the thesis (Chapter 1, Section 1.3; Chapter 5, Section 5.1) and aims to develop and test a Conditional Normalizing Flow GAN (CNFG-GAN) on the MNIST dataset. The specific goals are:
1.  To prove by example that a CNF-GAN, which combines a CNF generator with a GAN framework (e.g., DCGAN-style discriminator), can successfully train for unsupervised image generation on MNIST.
2.  To investigate the effectiveness of a hybrid training objective that incorporates both maximum likelihood estimation (via the CNF's exact log-likelihood calculation) and adversarial loss.
3.  To compare the performance of this CNF-GAN (e.g., FFJORD-DCGAN or RNODE-DCGAN variants explored in the thesis) against standalone CNFs, traditional GANs, and other flow-GANs in terms of synthetic sample quality, and potentially log-likelihood scores and latent space interpolation quality.

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
* This experiment uses the `train_cnfg_gan_mnist.py` script.
* The generator is a Continuous Normalizing Flow (CNF), such as an FFJORD or RNODE variant (as explored in Thesis Section 5.1.2, Figures 5.1, 5.2). The discriminator is typically a DCGAN-style architecture.
* Key hyperparameters include learning rates for generator and discriminator (thesis suggests G_lr between 1e-3 to 1e-5, D_lr slightly higher), batch size (e.g., 200 as mentioned in thesis), and the specifics of the CNF (number of blocks, ODE solver, tolerances) and discriminator architectures.
* The loss function for the generator combines a likelihood component (bits per dimension) and an adversarial component (binary cross entropy from discriminator output) (Thesis Section 5.1.3).
* The utilities previously in `train_misc.py` and `dist_utils.py` (which handled regularization, standard normal logprob, etc.) have been consolidated into `core_lib/utils.py` and `core_lib/distributed_utils.py` respectively, and are imported from there. 
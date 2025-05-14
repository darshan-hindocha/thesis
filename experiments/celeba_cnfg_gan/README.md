# CelebA CNF-GAN Experiment

## Goal

This experiment trains a Conditional Normalizing Flow GAN (CNFG-GAN) model, potentially leveraging an RNODE-based CNF as the generator and a U-Net architecture as the discriminator, on the CelebA (or CelebA-HQ) dataset. The primary objective is to combine the likelihood-based training and reversible properties of CNFs with the strong sample generation capabilities of GANs. This addresses one of the core research ideas of the thesis: to investigate if such a hybrid model can produce high-quality, realistic images by leveraging both adversarial and likelihood-based training signals (Thesis Chapter 4, Section 4.1; Chapter 5, Section 5.2).

## How to Run

[**User to complete: Provide specific command-line instructions to run `train_cnfg_gan_celeb.py`. Include any important arguments or link to a relevant script in `example_runner_scripts/`.**]

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

[**User to complete: Add any specific configurations, dataset preprocessing notes if not covered in the main README, or other relevant information for this experiment.**]
* This experiment utilizes the `train_cnfg_gan_celeb.py` script.
* The generator is a Continuous Normalizing Flow (CNF), possibly an RNODE variant, and the discriminator can be a U-Net (as explored in the thesis) or a more standard DCGAN-style architecture.
* Key hyperparameters include learning rates for both generator (G_lr) and discriminator (D_lr), batch size, and the specifics of the CNF and discriminator architectures.
* The thesis mentions experiments with an RNODE-U-Net on CelebHQ (Thesis Figure 5.3), so ensure dataset paths and any specific U-Net configurations (e.g., from `core_lib/networks/u_net/`) are correctly set.
* Pay attention to the balance between adversarial loss and any likelihood-based loss/regularization if implementing a hybrid training objective.
* This experiment likely uses the U-Net architecture defined in `core_lib/networks/u_net/`. 
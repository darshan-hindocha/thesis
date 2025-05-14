# CelebA CNF-GAN Experiment

## Goal

[**User to complete: Describe the specific model and research goal of this experiment using a Conditional Normalizing Flow GAN (CNFG-GAN) on the CelebA dataset.**]

This experiment trains a Conditional Normalizing Flow GAN (CNFG-GAN) model, likely leveraging a U-Net architecture, on the CelebA dataset.

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
* This experiment likely uses the U-Net architecture defined in `core_lib/networks/u_net/`. 
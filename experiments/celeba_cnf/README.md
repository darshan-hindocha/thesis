# CelebA CNF Experiment

## Goal

[**User to complete: Describe the specific model and research goal of this experiment using Continuous Normalizing Flows on the CelebA dataset.**]

This experiment trains a Continuous Normalizing Flow (CNF) model on the CelebA dataset.

## How to Run

[**User to complete: Provide specific command-line instructions to run `train_cnf_celeb.py`. Include any important arguments or link to a relevant script in `example_runner_scripts/`.**]

```bash
python train_cnf_celeb.py \
    --data celebahq \
    --imagesize 256 \
    --datadir /path/to/your/celeba_hq_dataset \
    --dims 64,64,64 \
    --num_blocks 10 \
    --lr 0.0001 \
    --batch_size 8 \
    --save experiments/celeba_cnf/results \
    # ... other relevant arguments (see script or example runners)
```

(Adjust `--datadir` and other parameters as needed.)

## Notes

[**User to complete: Add any specific configurations, dataset preprocessing notes if not covered in the main README, or other relevant information for this experiment.**] 
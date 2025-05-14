# CelebA CNF Experiment

## Goal

This experiment aims to train a standalone Continuous Normalizing Flow (CNF) model on the CelebA dataset. The primary goal is to perform density estimation and generate samples from the learned distribution. This setup can serve as a baseline to evaluate the performance of CNFs alone, for comparison against more complex models like CNF-GANs, and to explore the capabilities of NODEs for high-dimensional image data as discussed in the associated thesis (Chapter 1, Section 1.3).

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
* This experiment utilizes the `train_cnf_celeb.py` script (originally `CNFceleb.py`, then refactored from `train_generic.py` for this specific task).
* Key aspects to configure include the ODE solver, network architecture within the CNF (e.g., `dims`, `num_blocks`), learning rate, and batch size.
* Ensure the CelebA or CelebA-HQ dataset is correctly specified via `--datadir` and is preprocessed or handled appropriately by the dataset loader. 
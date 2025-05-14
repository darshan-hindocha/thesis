# CelebA CNF Experiment

## Goal

This experiment trains a Continuous Normalizing Flow (CNF) model on the CelebA dataset. The primary goal is to perform density estimation and generate samples from the learned distribution. This setup can serve as a baseline to evaluate the performance of CNFs alone, for comparison against more complex models like CNF-GANs, and to explore the capabilities of NODEs for high-dimensional image data as discussed in the associated thesis (Chapter 1, Section 1.3).

## How to Run

To run this experiment, execute the `train_cnf_celeb.py` script. Ensure your environment is set up and datasets are available as per the main project `README.md`.

Example command:
```bash
# From the project root directory
python experiments/celeba_cnf/train_cnf_celeb.py \
    --data celebahq \
    --imagesize 256 \
    --datadir /path/to/your/celeba_hq_dataset \
    --dims 64,64,64 \
    --num_blocks 10 \
    --lr 0.0001 \
    --batch_size 8 \
    --save experiments/celeba_cnf/results \
    # Add other arguments as needed; consult --help
```

Adjust `--datadir`, `--imagesize`, network parameters (`--dims`, `--num_blocks`), and training parameters (`--lr`, `--batch_size`, `--save`) as required for your setup and goals. You can refer to `example_runner_scripts/` for more detailed examples if available, or use `python experiments/celeba_cnf/train_cnf_celeb.py --help` for a full list of arguments.

## Notes

*   **Dataset:** This script is typically configured for the CelebA-HQ dataset. Ensure the dataset is downloaded and preprocessed if necessary. The `--datadir` argument should point to the root directory of the dataset.
*   **Computational Resources:** Training CNFs on high-resolution images like CelebA-HQ can be computationally intensive. Adjust batch size and model complexity based on available resources (GPU memory, etc.).
*   **Checkpoints:** The script saves model checkpoints and generated samples to the directory specified by the `--save` argument.
*   This script likely corresponds to the generic FFJORD training script adapted for CelebA, focusing on likelihood-based training.

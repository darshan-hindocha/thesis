# Project Readability Refactor Plan

This document outlines the steps to refactor the `neural-odes-thesis` project to improve its structure, clarity, and maintainability.

## Current Issues Identified

1.  **Major Code Redundancy**: The entire `lib` directory (including `layers`, `diffeq_layers`, `wrappers`, and various Python utility files) is duplicated between the `CNFGANMNIST` and `ffjord-rnode-master-master` directories.
2.  **Unclear Project Structure**: The relationship between `CNFGANMNIST` and `ffjord-rnode-master-master` is not immediately obvious, making it hard to understand the overall project narrative.
3.  **Lack of Clear Entry Points**: It's difficult to determine how to run experiments or what the primary goals of each sub-directory are.
4.  **Missing Dependencies**: No `requirements.txt` file makes it hard to set up the environment.
5.  **Minimal Documentation**: Lack of a comprehensive README hinders understanding and usability.

## Proposed Refactoring Strategy

The refactor will be performed in stages. The goal is to create a clean, understandable, and runnable codebase.

### Stage 1: Consolidate Core Library and Restructure Directories

1.  **Centralize the Core Library (`lib`)**:
    *   Create a new top-level directory: `core_lib/`.
    *   Move the contents of `ffjord-rnode-master-master/lib/` into `neural-odes-thesis/core_lib/`.
    *   Delete the redundant `CNFGANMNIST/lib/` directory.
    *   Delete the now empty `ffjord-rnode-master-master/lib/` directory.

2.  **Restructure Project Directories**:
    *   Rename `ffjord-rnode-master-master` to a more descriptive name, e.g., `generative_models_framework/` or `neural_ode_flows_project/`. (Decision: Let's provisionally call it `neural_ode_flows_project/` for now).
    *   Create a new top-level directory: `experiments/`.
    *   Move the contents of `CNFGANMNIST/` into `experiments/mnist_cnfg_gan/`.
        *   Rename `CNFGANMNIST/trainCNFGAN.py` to `experiments/mnist_cnfg_gan/train_cnfg_gan_mnist.py`.
    *   Identify other distinct experimental setups within the (newly renamed) `neural_ode_flows_project/` (e.g., from `CNFGANceleb.py`, `CNFceleb.py`, `train.py`) and move them into appropriately named subdirectories under `experiments/`. For example:
        *   `experiments/celeba_cnfg_gan/` (for `CNFGANceleb.py`)
        *   `experiments/celeba_cnf/` (for `CNFceleb.py`)
        *   `experiments/generic_trainer/` (for `train.py`)
    *   Move `ffjord-rnode-master-master/example-scripts/` to a top-level `example_runner_scripts/`.
    *   Move `ffjord-rnode-master-master/preprocessing/` to top-level (if generic) or into `core_lib/` (if it's core processing logic).
    *   Move `ffjord-rnode-master-master/u_net/` to `core_lib/networks/` or `core_lib/architectures/` if it's a general network component, or into a specific experiment if it's only used there.

### Stage 2: Update Code and Consolidate Utilities

1.  **Update Import Statements**:
    *   Systematically go through all Python files in `core_lib/` and `experiments/` and update import statements to reflect the new `core_lib` structure and relocated experiment files.
    *   Example: `from lib.layers.odefunc import ...` becomes `from core_lib.layers.odefunc import ...`.
    *   Imports within experiment files referencing other modules within the same experiment will also need adjustment if files were renamed or moved.

2.  **Consolidate Utility Functions**:
    *   Review `CNFGANMNIST/train_misc.py` and `CNFGANMNIST/dist_utils.py` (now in `experiments/mnist_cnfg_gan/`).
    *   Review `ffjord-rnode-master-master/train_misc.py` (if it exists and is different, or was part of the main training scripts).
    *   Move generalizable utility functions to `core_lib/utils.py` or create new relevant modules within `core_lib` (e.g., `core_lib/distributed_utils.py`).
    *   Ensure experiment-specific utilities remain within their respective experiment directories or are clearly namespaced if moved to `core_lib`.

### Stage 3: Improve Runnability and Documentation

1.  **Create `requirements.txt`**:
    *   Identify all external Python dependencies (e.g., PyTorch, NumPy, SciPy, Matplotlib, etc.).
    *   Create a `requirements.txt` file in the project root, listing these dependencies with their versions.

2.  **Develop Comprehensive `README.md`**:
    *   Create/Update a top-level `README.md` in the project root.
    *   It should include:
        *   The overall project goal and thesis context.
        *   A description of the new project structure (`core_lib/`, `experiments/`).
        *   Instructions for setting up the Python environment (Python version, `pip install -r requirements.txt`).
        *   Instructions on how to run key experiments, referencing `example_runner_scripts/` or providing direct commands.
        *   Information about the datasets used and any preprocessing steps (`preprocessing/`).

3.  **Experiment-Specific `README.md` Files (Optional but Recommended)**:
    *   For each subdirectory in `experiments/`, consider adding a small `README.md` explaining:
        *   The specific model/goal of that experiment.
        *   How to run that particular experiment.
        *   Any specific configurations or notes.

4.  **Update `.gitignore`**:
    *   Ensure `.DS_Store`, `__pycache__/`, `*.pyc`, and any environment-specific files (e.g., `.venv/`, `*.egg-info/`) are added to the top-level `.gitignore` file. Merge contents from the existing `.gitignore` if relevant.

### Stage 4: Testing and Refinement

1.  **Test Core Functionality**:
    *   Run the main training scripts for a few epochs to ensure the refactored code works.
    *   Verify data loading, model construction, and training loops.

2.  **Test Utility Scripts**:
    *   If there are validation or visualization scripts, test them.

3.  **Review and Iterate**:
    *   Review the new structure and code for any remaining issues or areas for improvement.
    *   Ensure the "narrative" of the project is clear from the file structure and documentation.

## Proposed New Directory Structure (Post-Refactor)

```
neural-odes-thesis/
├── core_lib/
│   ├── datasets.py
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── cnf.py
│   │   ├── diffeq_layers/
│   │   └── ... (other layer files)
│   ├── networks.py
│   ├── odenvp.py
│   ├── utils.py
│   └── ... (other core library files)
├── experiments/
│   ├── mnist_cnfg_gan/
│   │   ├── train_cnfg_gan_mnist.py
│   │   ├── train_misc.py         # (or moved to core_lib)
│   │   └── dist_utils.py         # (or moved to core_lib)
│   ├── celeba_cnfg_gan/
│   │   └── train_cnfg_gan_celeb.py
│   ├── celeba_cnf/
│   │   └── train_cnf_celeb.py
│   └── ... (other self-contained experiments)
├── example_runner_scripts/
│   ├── celeba5bit.sh
│   ├── cifar10.sh
│   └── ...
├── preprocessing/                # (If generic)
├── .gitignore
├── PROJECT_READABILITY_REFACTOR.md
├── README.md
└── requirements.txt
```

We will track progress against these stages.

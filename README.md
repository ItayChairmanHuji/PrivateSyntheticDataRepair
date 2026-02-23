# Private Synthetic Data Repair

This codebase contains code for the paper "Repairing Privately Generated Synthetic Data with Integrity and
Statistical Constraints".

## Installation

1. Create a new virtual environment (venv):
    ```sh
    python -m venv .venv
    ```

2. Activate the environment:
    ```sh
    source .venv/bin/activate
    ```

3. Install the required packages from the `requirements.txt` file:
    ```sh
    pip install -r requirements.txt
    ```
   If problems arise, a minimal requirements.txt file is provided in the repository.
   This file contains the necessary packages for running the code without versioning that might cause the installation
   issue. Note that this file is not a complete requirements.txt file and might not work for all systems.

4. If you with to run one of the ILP methods, make sure to plug your Gurobi license information in the file
   `config/repair_algorithm/ilp_repair_algorithm.yaml`. Note: A licensed version of Gurobi is required for datasets with
   larger rows and constraints.

5. If you with to use the WANDB logger, make sure to plug your WANDB details in the file
   `config/logger/wandb_logger.yaml`.

## Usage

The codebase uses Hydra to run experiments.
The main script is `src/main.py`, and differnt experiments can be run using different configuration files in the
`config` directory or overriding the default values using the command line arguments in your running script.

For example, to run the example script, use the following command:

```sh
./scripts/example_script.sh
```

This script runs a simple experiment using the co-noise synthesizer and the WVC dynamic repair algorithm. Similar
parameters can be overridden in this mannar.
In order to "swap" over several overridden parameters a ',' can be used, hydra will run the experiment for each
overridden parameter. For example, if we change line 11 in the example script to contain

```sh
./scripts/example_script.sh -m dataset=adult,census,compas,tax
```

Hydra will run the experiment for each dataset. Note that the line `hydra/launcher=joblib hydra.launcher.n_jobs=20`
makes hydra run the experiments in parallel and limit the number of jobs to 20. This functionality can be disabled by
removing this line.

Note that this code was used mainly for experiments running and is not a production ready codebase. 
For farther support, please contact itay.chairman@mail.huji.ac.il

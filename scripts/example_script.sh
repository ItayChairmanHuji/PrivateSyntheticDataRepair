#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR"
source .venv/bin/activate
export PYTHONPATH=$(pwd)
pip install -r requirements.txt
export HYDRA_FULL_ERROR=1
python src/main.py -m \
    hydra/launcher=joblib hydra.launcher.n_jobs=20\
    dataset=adult\
    synthesizer=co_noise_synthesizer\
    synthesizer.num_of_iterations=500\
    marginals_obtainer=top_k_marginals_obtainer\
    marginals_obtainer.num_of_marginals=100\
    repair_algorithm=wvc_dynamic_repair_algorithm\
    logger=wandb_logger\
    logger.run_name="Example"\
    logger.metric=synthesizer_num_of_iterations\
    +logger.configuration.sample_size='${dataset.sample_size}'\
    +logger.configuration.synthesizer=co_noise_synthesizer\
    +logger.configuration.repair_algorithm=vc_repair_algorithm


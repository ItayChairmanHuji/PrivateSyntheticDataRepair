import traceback

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig


def print_error(e):
    print("=" * 80)
    print("EXCEPTION CAUGHT")
    print("=" * 80)
    print(f"Error Type: {type(e).__name__}")
    print(f"Error Message: {str(e)}")
    print("\nFull Traceback:")
    print(traceback.format_exc())
    print("=" * 80)


def pipeline(config):
    logger = instantiate(config["logger"])
    print("Loading datasets...")
    private_data = instantiate(config["dataset"])
    print("Synthesizing data...")
    synthesizer = instantiate(config["synthesizer"])(logger=logger)
    synthetic_data = synthesizer.synthesize(private_data)
    synthetic_data.data.to_csv("synthetic_data.csv", index=False)
    print("Obtaining marginals...")
    marginals_obtainer = instantiate(config["marginals_obtainer"])(logger=logger)
    marginals = marginals_obtainer.obtain_marginals(synthetic_data, private_data)
    print("Repairing synthetic data...")
    repair_algorithm = instantiate(config[f"repair_algorithm"])(logger=logger)
    repaired_data = repair_algorithm.repair(synthetic_data, marginals)
    print("Evaluating results...")
    evaluator = instantiate(config["evaluator"])(logger=logger)
    evaluator.evaluate(private_data, synthetic_data, repaired_data, marginals)
    print("Done")


def run(config: DictConfig):
    try:
        pipeline(config)
    except Exception as e:
        print_error(e)
        raise


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(config: DictConfig):
    run(config)


if __name__ == "__main__":
    main()

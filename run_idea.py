import os
import json

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from ideas.idea_one.idea import IdeaOne

# dict for mapping idea names to the correct classes
IDEA_MAP = {
    "one": IdeaOne,
}


# TODO idea 2: try kalman filters
# TODO idea 3: try mixing self-supervised methods and supervised ones for dimensionality reduction and improved performance
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    idea_name = cfg.ideas.idea_name
    if idea_name not in IDEA_MAP:
        raise KeyError(
            f"[!] Idea name not recognised. Available names are: {list(IDEA_MAP.keys())}"
        )

    # preprocess the data
    idea = IDEA_MAP[idea_name](config=cfg.ideas)
    print(
        f"[+] {idea.__class__.__name__}: Preprocessing the trainining and the test data..."
    )
    idea.preprocess_data()

    # train the model
    print(f"[+] {idea.__class__.__name__}: Training the model...")
    idea.train_model()

    # run the model
    print(f"[+] {idea.__class__.__name__}: Running the model on the test data...")
    out_dict = idea.run_model()

    # save output to file
    out_file_path = os.path.join(
        HydraConfig.get().runtime.output_dir, "out_velocities.json"
    )
    print(f"[+] {idea.__class__.__name__}: Saving the result to {out_file_path}...")
    with open(out_file_path, "w") as f:
        json.dump(out_dict, f)

    print(
        f"[+] {idea.__class__.__name__}: All done! Now you can upload your submission to Kelvins."
    )


if __name__ == "__main__":
    main()

import hydra
from omegaconf import DictConfig

from ideas.idea_one.idea import IdeaOne

# dict for mapping idea names to the correct classes
IDEA_MAP = {
    "one": IdeaOne,
}


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    idea_name = cfg.ideas.idea_name
    if idea_name not in IDEA_MAP:
        raise KeyError(
            f"[!] Idea name not recognised. Available names are: {list(IDEA_MAP.keys())}"
        )

    idea = IDEA_MAP[idea_name](config=cfg.ideas)
    print(
        f"[+] {idea.__class__.__name__}: Preprocessing the trainining and the test data..."
    )
    idea.preprocess_data()
    # TODO what if other ideas do not involve training a neural net?
    print(f"[+] {idea.__class__.__name__}: Training the neural net...")
    idea.train_net()
    # idea.run()


if __name__ == "__main__":
    main()

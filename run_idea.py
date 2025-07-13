import argparse

from ideas.idea_one.idea import IdeaOne

IDEA_MAP = {
    "one": IdeaOne,
}


def main(idea_name: str) -> None:
    if idea_name not in IDEA_MAP:
        raise KeyError(
            f"[!] Idea name not recognised. Available names are: {list(IDEA_MAP.keys())}"
        )

    idea = IDEA_MAP[idea_name]()
    print(
        f"[+] {idea.__class__.__name__}: Preprocessing the trainining and the test data..."
    )
    idea.preprocess_data()
    # TODO what if other ideas do not involve training a neural net?
    print(f"[+] {idea.__class__.__name__}: Training the neural net...")
    # TODO keep the configuration for the various ideas into separate YAMLs
    idea.train_net()
    # idea.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an idea.")
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        required=True,
        help="Name of the idea to consider",
        metavar="IDEA_NAME",
    )

    args = parser.parse_args()
    main(
        args.name,
    )

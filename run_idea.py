import argparse

from ideas import *


def main(idea_name: str) -> None:
    idea_map = {
        "one": IdeaOne,
    }

    if idea_name not in idea_map:
        print(
            f"[+] Idea name not recognised! Available names are: {list(idea_map.keys())}"
        )
        return

    # TODO this is kind of ugly, perhaps use functions instead of classes?
    idea = idea_map[idea_name]
    idea()
    idea.run()


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

import os
from setuptools import setup, find_packages

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

with open(os.path.join(this_directory, "requirements.txt"), encoding="utf-8") as f:
    requirements = f.read().splitlines()
requirements = [req for req in requirements if req and not req.startswith("#")]

setup(
    name="elope",
    version="0.1.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/de-lachende-cavalier/elope",
    packages=find_packages(exclude=["*.egg-info", "tests*", "docs*", "build*"]),
    install_requires=requirements,
    python_requires=">=3.13",
)

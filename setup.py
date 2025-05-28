from setuptools import find_packages
from setuptools import setup


setup(
    name="ConfidNet",
    version='0.1.0',
    author="Charles Corbiere",
    url="https://github.com/valeoai/ConfidNet",
    install_requires=[
        "pyyaml",
        "coloredlogs",
        "torchsummary",
        "verboselogs",
        "setuptools",
        "tqdm",
        "click",
        "scikit-learn",
    ],
    packages=find_packages(),
)

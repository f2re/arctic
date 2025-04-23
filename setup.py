"""
Setup script for the ArcticCyclone package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="arctic_cyclone",
    version="0.1.0",
    author="Arctic Research Team",
    author_email="research@arctic-example.org",
    description="A framework for detecting and analyzing Arctic mesocyclones",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/arctic-research/arctic_cyclone",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "arctic_cyclone=arctic_cyclone.main:main",
        ],
    },
    include_package_data=True,
)
from setuptools import setup, find_packages

setup(
    name="cnnrfvis",
    author="Chen Chen",
    url="https://github.com/Phi-C/CNN-Receptive-Field-Visualization",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "cnnrfvis=cnnrfvis.cli:main",
        ],
    },
)
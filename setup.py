from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ochem-helper",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Neural network for molecular discovery and organic chemistry",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GreatPyreneseDad/OchemHelper",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        # Core requirements loaded from requirements.txt
    ],
    extras_require={
        "dev": ["pytest", "black", "flake8", "mypy"],
        "gpu": ["cuda-toolkit"],
        "viz": ["pymol", "nglview"],
    },
    entry_points={
        "console_scripts": [
            "ochem-train=scripts.train_model:main",
            "ochem-generate=scripts.generate_molecules:main",
            "ochem-api=api.app:main",
        ],
    },
)
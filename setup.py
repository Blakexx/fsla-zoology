from setuptools import find_namespace_packages, setup


_REQUIRED = [
    "numpy",
    "einops",
    "tqdm",
    "click",
    "pydantic",
    "wandb",
    "flash-linear-attention",
    "rotary-embedding-torch",
    "einx",
    "transformers",
]

_OPTIONAL = {
    "analysis": [
        "pandas",
        "seaborn",
        "matplotlib",
    ],
    "extra":[
        "rich", 
        "ray",
        "PyYAML",
    ]
}

# ensure that torch is installed, and send to torch website if not
try:
    import torch
except ModuleNotFoundError:
    raise ValueError("Please install torch first: https://pytorch.org/get-started/locally/")

setup(
    name="zoology", 
    version="0.0.1",
    description="",
    # every package under zoology/, its folders without an __init__.py among them, and the MRA mixer's CUDA sources
    # (compiled at import): an install from source is the whole tree, as an editable one is
    packages=find_namespace_packages(include=["zoology", "zoology.*"]),
    package_data={"zoology.mixers.mra.cuda": ["*.cu", "*.h", "*.cpp"]},
    install_requires=_REQUIRED,
    extras_require=_OPTIONAL,
    entry_points={
        'console_scripts': ['zg=zoology.cli:cli'],
    },
)

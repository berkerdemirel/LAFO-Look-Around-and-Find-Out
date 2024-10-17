# LAFO: Look Around and Find Out

<p align="center">
    <a href="https://github.com/berkerdemirel/LAFO-Look-Around-and-Find-Out/actions/workflows/test_suite.yml"><img alt="CI" src=https://img.shields.io/github/workflow/status/berkerdemirel/LAFO-Look-Around-and-Find-Out/Test%20Suite/main?label=main%20checks></a>
    <a href="https://berkerdemirel.github.io/LAFO-Look-Around-and-Find-Out"><img alt="Docs" src=https://img.shields.io/github/deployments/berkerdemirel/LAFO-Look-Around-and-Find-Out/github-pages?label=docs></a>
    <a href="https://github.com/grok-ai/nn-template"><img alt="NN Template" src="https://shields.io/badge/nn--template-0.4.0-emerald?style=flat&labelColor=gray"></a>
    <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-y-blue.svg"></a>
    <a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

Implementation of LAFO: OOD Detection with Relative Angles


## Installation

```bash
pip install git+ssh://git@github.com/berkerdemirel/LAFO-Look-Around-and-Find-Out.git
```


## Quickstart

[comment]: <> (> Fill me!)


## Development installation

Setup the development environment:

```bash
git clone git@github.com:berkerdemirel/LAFO-Look-Around-and-Find-Out.git
cd LAFO-Look-Around-and-Find-Out
conda env create -f env.yaml
conda activate lafo
pre-commit install
```

Run the tests:

```bash
pre-commit run --all-files
pytest -v
```


### Update the dependencies

Re-install the project in edit mode:

```bash
pip install -e .[dev]
```


## Citation
```BibTeX
@misc{demirel2024lafo,
      title={Look Around and Find Out: OOD Detection with Relative Angles},
      author={Berker Demirel and Marco Fumero and Francesco Locatello},
      year={2024},
      eprint={2410.04525},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.04525},
}
```


## License

This source code is released under the MIT license, included [here](LICENSE).

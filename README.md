# Nystromformer [![Twitter](https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Fgithub.com%2FRishit-dagli%2FNystromformer)](https://twitter.com/intent/tweet?text=Wow:&url=https%3A%2F%2Fgithub.com%2FRishit-dagli%2FNystromformer)

![PyPI](https://img.shields.io/pypi/v/Nystromformer)
[![Upload Python Package](https://github.com/Rishit-dagli/Nystromformer/actions/workflows/python-publish.yml/badge.svg)](https://github.com/Rishit-dagli/Nystromformer/actions/workflows/python-publish.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

![GitHub License](https://img.shields.io/github/license/Rishit-dagli/Nystromformer)
[![GitHub stars](https://img.shields.io/github/stars/Rishit-dagli/Nystromformer?style=social)](https://github.com/Rishit-dagli/Nystromformer/stargazers)
[![GitHub followers](https://img.shields.io/github/followers/Rishit-dagli?label=Follow&style=social)](https://github.com/Rishit-dagli)
[![Twitter Follow](https://img.shields.io/twitter/follow/rishit_dagli?style=social)](https://twitter.com/intent/follow?screen_name=rishit_dagli)

An implementation of the [Nyströmformer: A Nyström-Based Algorithm for Approximating Self-Attention](https://arxiv.org/abs/2102.03902) paper by Xiong et al. The self-attention mechanism that encodes the influence or dependence of other tokens on each specific token is a key component of the performance of Transformers. This uses the Nyström method to approximate standard self-attention with O(n) complexity allowing to exhibit scalability as a function of sequence length.

![](media/nystromformer.png)

## Installation

Run the following to install:

```sh
pip install nystromformer
```

## Developing nystromformer

To install `nystromformer`, along with tools you need to develop and test, run the following in your virtualenv:

```sh
git clone https://github.com/Rishit-dagli/Nystromformer.git
# or clone your own fork

cd Nystromformer
pip install -e .[dev]
```

To run rank and shape tests run the following:

```
pytest -v --disable-warnings --cov
```


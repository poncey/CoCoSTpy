# CoCo-ST (Python Implementation)

This repository provides an **unofficial Python implementation** of the *CoCo-ST* algorithm for contrastive spatial transcriptomics analysis.

The original R version ([here](https://github.com/WuLabMDA/CoCo-ST/tree/main)) was released under the MIT License.

## Overview

CoCo-ST identifies contrastive spatial structures by comparing two datasets (foreground vs. background) using normalized graph Laplacians and contrastive component analysis.

## Requirements

```
numpy
scipy
scanpy
scikit-learn
```

## Example Usage

```python
import CoCoSTpy as cocost


cocost.spatial_affinity(adata_b)
cocost.spatial_affinity(adata_t)
cocost.cocost(adata_b, adata_t)
```

You can also check the [example notebook](./example.ipynb) for a demonstration.

## Acknowledgement

This project is based on the original [CoCo-ST R code](https://github.com/WuLabMDA/CoCo-ST/tree/main) (MIT License).

Python translation © 2025 \<Yuxuan Pang\>.

Original publication link: [here](https://www.nature.com/articles/s41556-025-01781-z)

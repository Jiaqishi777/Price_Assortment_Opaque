# Price and Assortment Optimization with Opaque Products

This repository contains the numerical code accompanying the paper:

**“Price and Assortment Optimization under the Multinomial Logit Model with Opaque Products”**  
Omar El Housni, Adam N. Elmachtoub, Harsh Sheth, and Jiaqi Shi.

[arXiv:2502.08124](https://arxiv.org/abs/2502.08124)

## Numerical Experiments

[`Order-statistic-NRV-Tables-1-5.ipynb`](Order-statistic-NRV-Tables-1-5.ipynb) contains the numerical experiments reported in Section 5.6 and Appendix B of the paper.

The notebook reproduces Tables 1–5, which evaluate the NRV heuristic under different order statistics and lognormal price distributions. Each experiment uses 2,000 randomly generated instances with product counts ranging from 2 to 9 and random seed 48.

## Requirements

- Python 3
- NumPy
- pandas
- SciPy
- Jupyter Notebook

Install the required packages with:

```bash
pip install numpy pandas scipy notebook

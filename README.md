# DJEnsemble

This is a Python implementation of the Algorithm Stream DJEnsemble in the following paper:
[DJEnsemble: a Cost-Based Selection and Allocation of a Disjoint Ensemble of Spatio-temporal Models](https://dl.acm.org/doi/10.1145/3468791.3468806) by Rafael Pereira, Yania Souto, Anderson Chaves, Rocio Zorilla, Brian Tsan, Florin Rusu, Eduardo Ogasawara, Artur Ziviani, Fabio Porto. SSDBM 2021

The full version of the paper can be accessed at [arXiv](https://arxiv.org/abs/2005.11093).

Disclaimer: this is a unified Python implementation of the original work presented at SSDBM, 

## Requirements
- dtw==1.4.0
- dtw_python==1.1.12
- gldpy==0.2
- h5py==3.6.0
- matplotlib==3.1.2
- numpy==1.17.4
- pandas==0.25.3
- scikit_learn==1.1.2
- tensorflow==2.7.0
- xarray==0.20.2

Dependencies can be installed using the following command:

```bash
pip install -r requirements.txt
```

The projet should be run from main.py, informing the configuration file.
The example selects the best ensemble for a set of three models trained on the alerta-rio dataset:

```bash
python main.py queries/query-alerta-rio.query
```

# Stream DJEnsemble

This is a Python implementation of the Algorithm Stream DJEnsemble. Paper in Development, by Anderson Silva. 

## Requirements
dtw==1.4.0
dtw_python==1.1.12
gldpy==0.2
h5py==3.6.0
matplotlib==3.1.2
netCDF4==1.6.2
numpy==1.17.4
PySide6==6.4.1
PySimpleGUI==4.60.4
rpy2==3.5.5
scikit_learn==1.2.0
tensorflow==2.7.0
xarray==0.20.2

Dependencies can be installed using the following command:

```bash
pip install -r requirements.txt
```

The projet should be run from main.py, informing the configuration file.
The example selects the best ensemble for a set of three models trained on the alerta-rio dataset:

```bash
python main.py queries/query-alerta-rio.query
```

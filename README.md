# TAXA-Net

## Project Overview

A demo for TAXA-Net, a deep learning model for multi-step solar irradiance forecasting using historical ground-station observations.

## Quick Demo

From this directory, install the dependencies and open the notebook:

```sh
python -m pip install -r requirements.txt notebook
jupyter notebook demo.ipynb
```

Run all cells sequentially to load the sample data and weights, evaluate forecasts, and display an example.

## Data and Pretrained Weights

Raw observations are available from [https://dkasolarcentre.com.au/download?location=alice-springs].

`checkpoints/taxa_as_h96.pth` contains TAXA-Net weights trained on preprocessed observations from the DKASC Alice Springs site from December 2011 to November 2014.

Inputs include historical GHI, meteorological variables, and calendar features.

`data/sample_as_15min.csv` contains preprocessed observations from June 2015. 


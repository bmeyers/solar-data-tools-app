# solar-data-tools-app
Marimo application for using Solar Data Tools

## Installation instructions

1. Clone repository
3. Create a fresh Python virtual environment, e.g.,
```
conda create -n sdt-app pip
conda activate sdt-app
```
4. Install the packages in `requirements.txt`, e.g.,
```
pip install -r requirements.txt
```

## Run app
```
marimo run solar-data-tools-app.py
```

## Data access

The data are provided through the SETO [Solar Data Bounty Prize](https://www.herox.com/solardatabounty/update/6264). This applicaton automatically loads power generation data from site 2107 in the [OEDI S3 bucket](https://data.openei.org/s3_viewer?bucket=oedi-data-lake&prefix=pvdaq%2F2023-solar-data-prize%2F), and it will save a local copy in your working directory. Future use of the application does not require an internet connection.

## How to use application

1. Select an inverter source from the drop down menu in the upper left
2. The analysis of each power signal takeks ~30 seconds, but the results are cached
3. After analysis is complete there are 5 tabs to explore: data viewer, losses, daily data quality, capacity changes, and clipping analysis

The image below shows a view of the "losses" tab:
![result(9)](https://github.com/bmeyers/solar-data-tools-app/assets/1463184/42dd302f-6ba6-46c0-94ca-f66379370032)

### Acknowledgement

This material is based upon work supported by the U.S. Department of Energy's Office of Energy Efficiency and Renewable Energy (EERE) under the Solar Energy Technologies Office Award Number 38529.

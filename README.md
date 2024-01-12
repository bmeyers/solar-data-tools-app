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



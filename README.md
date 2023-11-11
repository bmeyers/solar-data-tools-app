# solar-data-tools-app
Marimo application for using Solar Data Tools

## Installation instructions

1. Clone repository
3. Create a fresh Python 3.10 virtual environment, e.g.,
```
conda create -n sdt-app python=3.10
conda activate sdt-app
```
4. Install the packages in `requirements.txt`, e.g.,
```
pip install -r requirements.txt
```
5. Move/copy the file `cassandra_cluster` to the `~/.aws` directory on your machine

## Run app
```
marimo run solar-data-tools-app.py
```

## Data access

Some csv files are avaible to use with the file loader in `system-data/` directory. Additionally, with the Cassandara cluster configured correctly and an internet connection, you can use the drop down menus to access data from the database.

## Data confidentiality

The data in the CSV files and the Cassandra database should be considered private and should not be retained beyond the testing of this app nor copied to any 3rd parties. Please reach out if you have any questions regarding handling of the data.

import marimo

__generated_with = "0.1.50"
app = marimo.App(layout_file="layouts/solar-data-tools-app.grid.json")


@app.cell
def __(mo, site_list):
    file_source = mo.ui.file(kind='area')
    text_source = mo.ui.dropdown(site_list['site'].values)
    data_source = mo.ui.dropdown(options={
        "file loader": file_source,
        "DB access": text_source
    })
    return data_source, file_source, text_source


@app.cell
def __(data_source, mo, site_list, text_source):
    system_picker = mo.ui.dropdown([])
    if data_source.selected_key is not None:
        if text_source.selected_key is not None:
            num_systems = int(site_list\
                              .where(site_list['site'] == text_source.selected_key)\
                              .dropna()['count']\
                              .values[0])
            system_list = [f"ac_power_{_ix+1:02}" for _ix in range(num_systems)]
            system_picker = mo.ui.dropdown(system_list)
    return num_systems, system_list, system_picker


@app.cell
def __(data_source, mo):
    isinstance(data_source.value, mo.ui.file)
    return


@app.cell
def __(data_source, mo, system_picker):

    if data_source.selected_key == "DB access":
        data_picker = mo.hstack([data_source, data_source.value, system_picker])
    else:
        data_picker = mo.hstack([data_source, data_source.value])
    return data_picker,


@app.cell
def __(data_picker):
    data_picker
    return


@app.cell
def __(
    custom_cache,
    data_source,
    file_source,
    mo,
    plaintext,
    process_file,
    process_file2,
    system_picker,
    text_source,
):
    if data_source.selected_key == "file loader":
        if len(file_source.value) == 0:
            mo.stop(True)
        custom_cache[file_source.value[0].name] = file_source.value[0].contents
        current_key = file_source.value[0].name
        dh, _output = process_file(current_key)
    elif data_source.selected_key == "DB access":
        if system_picker.value is not None:
            dh, _output = process_file2(text_source.value, system_picker.value)
    plaintext(_output).center()
    return current_key, dh


@app.cell
def __(dh, mo, plaintext):
    with mo.capture_stdout() as buffer:
        dh.report()
    plaintext(buffer.getvalue()).center()
    # with mo.redirect_stdout():
        # dh.report()
    return buffer,


@app.cell
def __():
    # dh.setup_location_and_orientation_estimation(-8)
    return


@app.cell
def __():
    # lat, lon, tilt, az = dh.estimate_location_and_orientation()
    return


@app.cell
def __():
    return


@app.cell
def __():
    # data_file = mo.ui.file(kind='area')
    # data_file
    return


@app.cell
def __():
    # if len(data_file.value) == 0:
    #     mo.stop(True)
    # custom_cache[data_file.value[0].name] = data_file.value[0].contents
    # current_key = data_file.value[0].name
    return


@app.cell
def __(dh, mo):
    _heatmaps = mo.vstack([
        dh.plot_heatmap('raw'),
        dh.plot_heatmap('filled')
    ])
    _daily = mo.vstack([
        dh.plot_daily_energy(flag='clear'),
        mo.hstack([
            dh.plot_density_signal(flag='bad', show_fit=True),
            dh.plot_data_quality_scatter()
        ])
    ])
    _capacity = dh.plot_capacity_change_analysis()
    _clipping = mo.vstack([
        dh.plot_clipping(),
        dh.plot_daily_max_cdf_and_pdf()
    ])
    mo.tabs(
        {
            "heatmaps": _heatmaps,
            "daily data quality": _daily,
            "capacity changes": _capacity,
            "clipping analysis": _clipping
        }
    )
    return


@app.cell
def __():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import io
    from pathlib import Path
    from contextlib import redirect_stdout
    from functools import cache
    from solardatatools import DataHandler
    from solardatatools.dataio import load_cassandra_data
    return (
        DataHandler,
        Path,
        cache,
        io,
        load_cassandra_data,
        mo,
        np,
        pd,
        plt,
        redirect_stdout,
    )


@app.cell
def __(process_file):
    print(process_file.cache_info())
    return


@app.cell
def __(
    DataHandler,
    cache,
    custom_cache,
    io,
    load_cassandra_data,
    pd,
    redirect_stdout,
):
    @cache
    def process_file(key):
        fb = custom_cache[key]
        df = pd.read_csv(io.BytesIO(fb), parse_dates=[0], index_col=0)
        dh = DataHandler(df)
        dh.fix_dst()
        with redirect_stdout(io.StringIO()) as _f:
            dh.run_pipeline()
        output = _f.getvalue()
        return dh, output

    @cache
    def process_file2(site_key, system_key):
        df = load_cassandra_data(site_key)
        dh = DataHandler(df, convert_to_ts=True)
        with redirect_stdout(io.StringIO()) as _f:
            dh.run_pipeline(power_col=system_key)
        output = _f.getvalue()
        return dh, output
    return process_file, process_file2


@app.cell
def __(mo):
    def plaintext(text):
        return mo.Html(f"<pre style='font-size: 12px'>{text}</pre>")
    return plaintext,


@app.cell
def __(Path, pd):
    custom_cache = {}
    site_list = pd.read_csv(Path('.') / "layouts" / "system_counts_per_site.csv", header=0)
    site_list = site_list.sort_values('site')
    return custom_cache, site_list


if __name__ == "__main__":
    app.run()

import marimo

__generated_with = "0.1.76"
app = marimo.App(layout_file="layouts/solar-data-tools-app.grid.json")


@app.cell
def __(DF, mo):
    columns = list(DF.columns)
    columns.sort()
    inverter_select = mo.ui.dropdown(columns, value=columns[0])
    inverter_select
    return columns, inverter_select


@app.cell
def __(inverter_select, mo, process_file):
    dh, _captured, dse = process_file(inverter_select.value)
    with mo.redirect_stdout():
        print(_captured.getvalue())
    return dh, dse


@app.cell
def __(dh, mo):
    with mo.redirect_stdout():
        dh.report()
    return


@app.cell
def __(
    dh,
    get_day,
    mo,
    num_day_select,
    plt,
    start_day_select,
    start_day_select2,
):
    dh.plot_heatmap("raw", figsize=(12, 5))
    plt.axvline(get_day(), color="yellow", ls="--", linewidth=1)
    hmfig = plt.gcf()
    heatmaps = mo.vstack(
        [
            hmfig,
            mo.hstack([start_day_select, start_day_select2, num_day_select]),
            dh.plot_daily_signals(
                start_day=get_day(),
                num_days=num_day_select.value,
                figsize=(12, 4),
            ),
        ]
    )
    return heatmaps, hmfig


@app.cell
def __(dse, mo, np, plt):
    _pie = dse.plot_pie()
    plt.figure()
    _waterfall = dse.plot_waterfall()
    plt.figure()
    _fig_decomp = dse.problem.plot_decomposition(
        exponentiate=True, figsize=(16, 8.5)
    )
    _ax = _fig_decomp.axes
    _ax[0].plot(
        np.arange(len(dse.energy_data))[~dse.use_ixs],
        dse.energy_model[-1, ~dse.use_ixs],
        color="red",
        marker=".",
        ls="none",
    )
    _ax[0].set_title("weather and system outages")
    _ax[1].set_title("capacity changes")
    _ax[2].set_title("soiling")
    _ax[3].set_title("degradation")
    _ax[4].set_title("baseline")
    _ax[5].set_title("measured energy (green) and model minus weather")
    plt.tight_layout()
    plt.figure()
    _val = np.round(dse.degradation_rate, 2)
    losses = mo.vstack(
        [
            mo.center(mo.md(f"##estimated degradation rate: {_val:.02}%/yr")),
            mo.hstack([_pie, _waterfall]),
            mo.center(_fig_decomp),
        ]
    )
    return losses,


@app.cell
def __(dh, mo):
    _report = dh.report(verbose=False, return_values=True)
    _qval = 100 * (1 - _report["quality score"])
    daily = mo.vstack(
        [
            mo.center(
                mo.md(f"##{_qval:.0f}% of days experienced a system outage")
            ),
            mo.center(dh.plot_daily_energy(flag="bad", figsize=(12, 3))),
            mo.center(
                dh.plot_density_signal(flag="bad", show_fit=True, figsize=(12, 3))
            ),
        ]
    )
    return daily,


@app.cell
def __(dh, mo):
    _num_clusters = len(set(dh.capacity_analysis.labels))
    capacity = mo.vstack(
        [
            mo.center(mo.md(f"##{_num_clusters} capacity levels detected")),
            mo.center(dh.plot_capacity_change_analysis(figsize=(8, 5))),
        ]
    )
    return capacity,


@app.cell
def __(dh, mo):
    _report = dh.report(verbose=False, return_values=True)
    _cval = 100 * _report["clipped fraction"]
    clipping = mo.vstack(
        [
            mo.center(mo.md(f"##{_cval:.0f}% of days experienced clipping")),
            mo.center(dh.plot_clipping(figsize=(12, 6))),
        ]
    )
    return clipping,


@app.cell
def __(capacity, clipping, daily, heatmaps, losses, mo):
    mo.tabs(
        {
            "data viewer": heatmaps,
            "losses": losses,
            "daily data quality": daily,
            "capacity changes": capacity,
            "clipping analysis": clipping,
        }
    )
    return


@app.cell
def __():
    import marimo as mo
    import matplotlib.pyplot as plt
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    import numpy as np
    import pandas as pd
    import boto3
    import os
    import sys
    from pathlib import Path
    from contextlib import contextmanager
    from io import StringIO
    from functools import cache
    from solardatatools import DataHandler
    from solardatatools.dataio import load_cassandra_data
    from gfosd import Problem
    import gfosd.components as comp
    from spcqe.functions import make_basis_matrix, make_regularization_matrix


    @contextmanager
    def capture_stdout():
        # Save the original stdout
        original_stdout = sys.stdout

        # Create a StringIO object to capture the output
        captured_output = StringIO()
        sys.stdout = captured_output

        try:
            # Yield control to the code within the 'with' block
            yield captured_output
        finally:
            # Restore the original stdout
            sys.stdout = original_stdout
    return (
        DataHandler,
        Path,
        Problem,
        StringIO,
        boto3,
        cache,
        capture_stdout,
        comp,
        contextmanager,
        load_cassandra_data,
        make_basis_matrix,
        make_regularization_matrix,
        mo,
        np,
        os,
        pd,
        plt,
        sys,
    )


@app.cell
def __(boto3, os, pd):
    def load_data(filename, s3_bucket, s3_key):
        local_file_path = filename
        # Check if the file exists locally
        if os.path.exists(local_file_path):
            print(f"Loading local CSV file: {local_file_path}")
            data_frame = load_csv(local_file_path)
        else:
            print(f"Local CSV file not found. Downloading from S3.")
            download_csv_from_s3(s3_bucket, s3_key, local_file_path)
            data_frame = load_csv(local_file_path)

        return data_frame


    def download_csv_from_s3(bucket_name, s3_key, local_destination):
        s3 = boto3.client("s3")
        s3.download_file(bucket_name, s3_key, local_destination)


    def load_csv(file_path):
        df = pd.read_csv(
            file_path,
            index_col=0,
            parse_dates=[0],
            usecols=lambda x: "ac_power" in x.lower()
            or x.lower() == "measured_on",
        )
        return df
    return download_csv_from_s3, load_csv, load_data


@app.cell
def __(load_data):
    DF = load_data(
        "2107_electrical_data.csv",
        "oedi-data-lake",
        "pvdaq/2023-solar-data-prize/2107_OEDI/data/2107_electrical_data.csv",
    )
    return DF,


@app.cell
def __(np, pd):
    def waterfall_plot(data, index, figsize=(10, 4)):
        # Store data and create a blank series to use for the waterfall
        trans = pd.DataFrame(data=data, index=index)
        blank = trans.amount.cumsum().shift(1).fillna(0)

        # Get the net total number for the final element in the waterfall
        total = trans.sum().amount
        trans.loc["measured energy"] = total
        blank.loc["measured energy"] = total

        # The steps graphically show the levels as well as used for label placement
        step = blank.reset_index(drop=True).repeat(3).shift(-1)
        step[1::3] = np.nan

        # When plotting the last element, we want to show the full bar,
        # Set the blank to 0
        blank.loc["measured energy"] = 0

        # Plot and label
        my_plot = trans.plot(
            kind="bar",
            stacked=True,
            bottom=blank,
            legend=None,
            figsize=figsize,
            title="System Loss Factor Waterfall",
        )
        my_plot.plot(step.index, step.values, "k")
        my_plot.set_xlabel("Loss Factors")
        my_plot.set_ylabel("Energy (Wh)")

        # Get the y-axis position for the labels
        y_height = trans.amount.cumsum().shift(1).fillna(0)

        # Get an offset so labels don't sit right on top of the bar
        max = trans.max()
        max = max.iloc[0]
        neg_offset = max / 25
        pos_offset = max / 50
        plot_offset = int(max / 15)

        # Start label loop
        loop = 0
        for index, row in trans.iterrows():
            # For the last item in the list, we don't want to double count
            if row["amount"] == total:
                y = y_height.iloc[loop]
            else:
                y = y_height.iloc[loop] + row["amount"]
            # Determine if we want a neg or pos offset
            if row["amount"] > 0:
                y += pos_offset
            else:
                y -= neg_offset
            my_plot.annotate(
                "{:,.0f}".format(row["amount"]), (loop, y), ha="center"
            )
            loop += 1

        # Scale up the y axis so there is room for the labels
        my_plot.set_ylim(0, blank.max() + int(plot_offset))
        # Rotate the labels
        my_plot.set_xticklabels(trans.index, rotation=0)
        fig = my_plot.get_figure()
        fig.set_layout_engine(layout="tight")
        return fig
    return waterfall_plot,


@app.cell
def __(
    DF,
    DataHandler,
    DegradationSoilingEstimator,
    cache,
    capture_stdout,
):
    @cache
    def process_file(col_key):
        dh = DataHandler(DF)
        dh.fix_dst()
        with capture_stdout() as captured:
            dh.run_pipeline(power_col=col_key)
        ds = DegradationSoilingEstimator(
            dh.daily_signals.energy,
            capacity_change_labels=dh.capacity_analysis.labels,
            outage_flags=~dh.daily_flags.no_errors,
            weight_soiling_stiffness=1e0,
            weight_soiling_sparsity=1e-2,
        )
        ds.estimate_losses()
        return dh, captured, ds
    return process_file,


@app.cell
def __(np):
    def model_wrapper(energy_model, use_ixs):
        n = energy_model.shape[0]

        def model_f(**kwargs):
            defaults = {f"arg{i+1}": False for i in range(n)}
            defaults.update(kwargs)
            slct = [True] + [item for _, item in defaults.items()]
            apply_outages = slct[-1]
            slct = slct[:-1]
            model_select = energy_model[slct]
            daily_energy = np.product(model_select, axis=0)
            if apply_outages:
                daily_energy = daily_energy[use_ixs]
            return np.sum(daily_energy)

        return model_f
    return model_wrapper,


@app.cell
def __(np):
    def enumerate_paths_full(origin, destination, path=None):
        """
        recursive algorithm for generating all possible monotonicly increasing paths between
        two points on a n-dimensional hypercube
        """
        origin = list(origin)
        destination = list(destination)
        correct_ordering = np.all(
            np.asarray(destination, dtype=int) - np.asarray(origin, dtype=int) >= 0
        )
        if not correct_ordering:
            raise Exception("destination must be larger than origin in all dimensions")
        if path is None:
            path = []
        paths = []
        if origin == destination:
            # a path has been completed
            paths.append(path + [origin])
        else:
            # find the next index that can be incremented
            for i in range(len(origin)):
                if origin[i] != destination[i]:
                    # create the next point in this path
                    next_position = list(origin)
                    next_position[i] = destination[0]
                    # recurse to finish all paths that begin on this path
                    paths.extend(
                        enumerate_paths_full(
                            next_position, destination, path + [origin]
                        )
                    )

        return paths


    def enumerate_paths(n, dtype=int):
        """
        enumerates all possible paths from the origin to the ones vector in R^n
        """
        origin = np.zeros(n, dtype=dtype)
        destination = np.ones(n, dtype=dtype)
        return np.asarray(enumerate_paths_full(origin, destination))
    return enumerate_paths, enumerate_paths_full


@app.cell
def __(enumerate_paths, model_wrapper, np):
    def attribute_losses(energy_model, use_ixs):
        """This function assigns a total attribution to each loss factor, given a
        multiplicative loss factor model relative to a baseline, using Shapley
        attribution.

        :param energy_model: a multiplicative decomposition of PV daily energy, with the
            baseline first -- ie: baseline, degradation, soiling, capacity changes, and
            weather (residual)
        :type energy_model: 2d numpy array of shape n x T, where T is the number of days
            and n is the number of model factors
        :param use_ixs: a numpy boolean index where False records a system outage
        :type use_ixs: 1d numpy boolean array

        :return: a list of energy loss attributions, in the input order
        :rtype: 1d numpy float array
        """
        model_f = model_wrapper(energy_model, use_ixs)
        paths = enumerate_paths(energy_model.shape[0], dtype=bool)
        energy_estimates = np.zeros((paths.shape[0], paths.shape[1]))
        for ix, path in enumerate(paths):
            for jx, point in enumerate(path):
                kwargs = {f"arg{i+1}": v for i, v in enumerate(point)}
                energy = model_f(**kwargs)
                energy_estimates[ix, jx] = energy
        lifts = np.diff(energy_estimates, axis=1)
        path_diffs = np.diff(paths, axis=1)
        ordering = np.argmax(path_diffs, axis=-1)
        ordered_lifts = np.take_along_axis(
            lifts, np.argsort(ordering, axis=1), axis=1
        )
        # print(energy_estimates)
        # print(lifts)
        # print(ordered_lifts)
        attributions = np.average(ordered_lifts, axis=0)
        total_energy = energy_estimates[0, -1]
        baseline_energy = energy_estimates[0, 0]
        # check that we've attributed all losses
        assert np.isclose(np.sum(attributions), total_energy - baseline_energy)
        return attributions
    return attribute_losses,


@app.cell
def __(
    Problem,
    attribute_losses,
    comp,
    deg_select,
    make_basis_matrix,
    make_regularization_matrix,
    np,
    plt,
    sp,
    waterfall_plot,
):
    class DegradationSoilingEstimator:
        def __init__(
            self,
            energy_data,
            capacity_change_labels=None,
            outage_flags=None,
            **kwargs
        ):
            self.energy_data = energy_data
            log_energy = np.zeros_like(self.energy_data)
            is_zero = np.isclose(energy_data, 0, atol=1e-1)
            log_energy[is_zero] = np.nan
            log_energy[~is_zero] = np.log(energy_data[~is_zero])
            self.log_energy = log_energy
            self.use_ixs = ~is_zero
            if outage_flags is not None:
                self.use_ixs = np.logical_and(self.use_ixs, ~outage_flags)
            self.capacity_change_labels = capacity_change_labels
            self.total_measured_energy = np.sum(self.energy_data[self.use_ixs])
            self.problem = self.make_problem(**kwargs)
            self.degradation_rate = None
            self.energy_model = None
            self.log_energy_model = None
            self.total_energy_loss = None
            self.total_percent_loss = None
            self.degradation_energy_loss = None
            self.soiling_energy_loss = None
            self.capacity_change_loss = None
            self.weather_energy_loss = None
            self.weather_percent_loss = None
            self.outage_energy_loss = None
            self.degradation_percent = None
            self.soiling_percent = None
            self.capacity_change_percent = None
            self.weather_percent = None
            self.outage_percent = None

        def estimate_losses(self, solver="CLARABEL"):
            self.problem.decompose(solver=solver, verbose=False)
            # in the SD formulation, we put the residual term first, so it's the reverse order of how we specify this model (weather last)
            self.log_energy_model = self.problem.decomposition[::-1]
            self.energy_model = np.exp(self.log_energy_model)
            self.degradation_rate = 100 * np.median(
                (self.energy_model[1][365:] - self.energy_model[1][:-365])
                / self.energy_model[1][365:]
            )
            # self.energy_lost_outages = np.sum(self.energy_model[:, self.use_ixs]) - np.sum(self.energy_model)
            total_energy = np.sum(self.energy_data[self.use_ixs])
            baseline_energy = np.sum(self.energy_model[0])
            self.total_energy_loss = total_energy - baseline_energy
            self.total_percent_loss = (
                100 * self.total_energy_loss / baseline_energy
            )

            out = attribute_losses(self.energy_model, self.use_ixs)
            self.degradation_energy_loss = out[0]
            self.soiling_energy_loss = out[1]
            self.capacity_change_loss = out[2]
            self.weather_energy_loss = out[3]
            self.outage_energy_loss = out[4]

            self.degradation_percent = out[0] / self.total_energy_loss
            self.soiling_percent = out[1] / self.total_energy_loss
            self.capacity_change_percent = out[2] / self.total_energy_loss
            self.weather_percent = out[3] / self.total_energy_loss
            self.outage_percent = out[4] / self.total_energy_loss

            assert np.isclose(
                self.total_energy_loss,
                self.degradation_energy_loss
                + self.soiling_energy_loss
                + self.capacity_change_loss
                + self.weather_energy_loss
                + self.outage_energy_loss,
            )
            return

        def report(self):
            if self.total_energy_loss is not None:
                out = {
                    "degradation rate [%/yr]": self.degradation_rate,
                    "total energy loss [kWh]": self.total_energy_loss,
                    "degradation energy loss [kWh]": self.degradation_energy_loss,
                    "soiling energy loss [kWh]": self.soiling_energy_loss,
                    "capacity change energy loss [kWh]": self.capacity_change_loss,
                    "weather energy loss [kWh]": self.weather_energy_loss,
                    "system outage loss [kWh]": self.outage_energy_loss,
                }
                return out

        def holdout_validate(self, seed=None, solver="CLARABEL"):
            residual, test_ix = self.problem.holdout_decompose(
                seed=seed, solver=solver
            )
            error_metric = np.sum(np.abs(residual))
            return error_metric

        def make_problem(
            self,
            tau=0.9,
            num_harmonics=4,
            deg_type="linear",
            include_soiling=True,
            weight_seasonal=10e-2,
            weight_soiling_stiffness=1e0,
            weight_soiling_sparsity=1e-2,
            weight_deg_nonlinear=10e4,
        ):
            # Pinball loss noise
            c1 = comp.SumQuantile(tau=tau)
            # Smooth periodic term
            length = len(self.log_energy)
            periods = [365.2425]  # average length of a year in days
            _B = make_basis_matrix(num_harmonics, length, periods)
            _D = make_regularization_matrix(
                num_harmonics, weight_seasonal, periods
            )
            c2 = comp.Basis(basis=_B, penalty=_D)
            # Soiling term
            if include_soiling:
                c3 = comp.Aggregate(
                    [
                        comp.Inequality(vmax=0),
                        comp.SumAbs(weight=weight_soiling_stiffness, diff=2),
                        comp.SumQuantile(
                            tau=0.98, weight=10 * weight_soiling_sparsity, diff=1
                        ),
                        comp.SumAbs(weight=weight_soiling_sparsity),
                    ]
                )
            else:
                c3 = comp.Aggregate([comp.NoSlope(), comp.FirstValEqual(value=0)])
            # Degradation term
            if deg_type == "linear":
                c4 = comp.Aggregate(
                    [comp.NoCurvature(), comp.FirstValEqual(value=0)]
                )
            elif deg_type == "nonlinear":
                n_tot = length
                n_reduce = int(0.9 * n_tot)
                bottom_mat = sp.lil_matrix((n_tot - n_reduce, n_reduce))
                bottom_mat[:, -1] = 1
                custom_basis = sp.bmat([[sp.eye(n_reduce)], [bottom_mat]])
                c4 = comp.Aggregate(
                    [
                        comp.Inequality(vmax=0, diff=1),
                        comp.SumSquare(diff=2, weight=weight_deg_nonlinear),
                        comp.FirstValEqual(value=0),
                        comp.Basis(custom_basis),
                    ]
                )
            elif deg_select.value == "none":
                c4 = comp.Aggregate([comp.NoSlope(), comp.FirstValEqual(value=0)])
            # capacity change term — leverage previous analysis from SDT pipeline
            if self.capacity_change_labels is not None:
                basis_M = np.zeros((length, len(set(self.capacity_change_labels))))
                for lb in set(self.capacity_change_labels):
                    slct = np.array(self.capacity_change_labels) == lb
                    basis_M[slct, lb] = 1
                c5 = comp.Aggregate(
                    [
                        comp.Inequality(vmax=0),
                        comp.Basis(basis=basis_M),
                        comp.SumAbs(weight=1e-6),
                    ]
                )
            else:
                c5 = comp.Aggregate([comp.NoSlope(), comp.FirstValEqual(value=0)])

            prob = Problem(
                self.log_energy, [c1, c5, c3, c4, c2], use_set=self.use_ixs
            )
            return prob

        def plot_pie(self):
            plt.pie(
                [
                    np.clip(-self.degradation_energy_loss, 0, np.inf),
                    np.clip(-self.soiling_energy_loss, 0, np.inf),
                    np.clip(-self.capacity_change_loss, 0, np.inf),
                    np.clip(-self.weather_energy_loss, 0, np.inf),
                    np.clip(-self.outage_energy_loss, 0, np.inf),
                ],
                labels=[
                    "degradation",
                    "soiling",
                    "capacity change",
                    "weather",
                    "outages",
                ],
                autopct="%1.1f%%",
            )
            plt.title("System loss breakdown")
            return plt.gcf()

        def plot_waterfall(self):
            index = [
                "baseline",
                "weather",
                "outages",
                "capacity changes",
                "soiling",
                "degradation",
            ]
            bl = np.sum(self.energy_model[0])
            data = {
                "amount": [
                    bl,
                    self.weather_energy_loss,
                    self.outage_energy_loss,
                    self.capacity_change_loss,
                    self.soiling_energy_loss,
                    self.degradation_energy_loss,
                ]
            }
            fig = waterfall_plot(data, index)
            return fig
    return DegradationSoilingEstimator,


@app.cell
def __(mo):
    get_day, set_day = mo.state(50)
    return get_day, set_day


@app.cell
def __(dh, get_day, mo, set_day):
    start_day_select = mo.ui.slider(
        0,
        len(dh.day_index) - 1,
        1,
        label="start day",
        value=get_day(),
        on_change=set_day,
    )
    return start_day_select,


@app.cell
def __(mo):
    num_day_select = mo.ui.slider(1, 14, 1, value=5, label="number of days")
    return num_day_select,


@app.cell
def __(dh, get_day, mo, set_day):
    start_day_select2 = mo.ui.number(
        0,
        len(dh.day_index) - 1,
        1,
        label="start day",
        value=get_day(),
        on_change=set_day,
    )
    return start_day_select2,


if __name__ == "__main__":
    app.run()

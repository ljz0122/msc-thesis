import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox, Button, RadioButtons
import pandas as pd
from astropy.io import fits
import pickle
from dust_extinction.grain_models import Y24
import astropy.units as u
import re

with open("/Users/ericljz/astro/MSc-Thesis/data/interpolated_compoM.pkl", "rb") as f:
    compoM_bspl = pickle.load(f)

ext_model = Y24()


def read_spectrum(data_path):
    """
    Reads an observed spectrum from a file.

    Parameters
    ----------
    data_path : str
        Path to the observed spectrum data file (2 or 3 columns: wavelength, flux, [error]).

    Returns
    -------
    lam : ndarray
        Wavelength array of the observed spectrum.
    flux : ndarray
        Flux array of the observed spectrum.
    err_flux : ndarray
        Error array of the observed spectrum. If not provided, returns an array of ones.
    """
    data = np.loadtxt(data_path, skiprows=1)
    wavelength = data[:, 0].astype(float)
    flux = data[:, 1].astype(float)
    if data.shape[1] > 2:
        err_flux = np.sqrt(data[:, 2].astype(float))
    else:
        err_flux = np.ones_like(flux)

    spec = pd.DataFrame({"wavelength": wavelength, "flux": flux, "err_flux": err_flux})
    spec = spec.dropna().reset_index(drop=True)
    threshold = 5e-15
    spec_filtered = spec[(spec["flux"] <= threshold)]

    filename = os.path.basename(data_path)
    print(filename)

    return spec_filtered


def plot_spectrum(
    spec, objname, init_z=0.5, init_Av=0.5, source_id=None, save_path=None
):
    fig, ax = plt.subplots(figsize=(16, 8))

    wl = spec["wavelength"].values
    flux = spec["flux"].values
    err_flux = spec["err_flux"].values

    ax.plot(wl, flux, color="blue", alpha=0.7, label="Observed Spectrum")
    ax.fill_between(
        wl,
        flux - err_flux,
        flux + err_flux,
        color="grey",
        alpha=0.3,
        label="1-sigma Error",
    )

    temp = compoM_bspl(wl / (1 + init_z)) * ext_model.extinguish(wl * u.AA, Av=init_Av)

    fit_area = np.nonzero((wl > 5700) & (wl < 6200))
    filter_area = np.nonzero((wl > 3700) & (wl < 9200))

    model_factor = np.nanmean(flux[fit_area]) / np.nanmean(temp[fit_area])
    temp *= model_factor
    ymax = np.nanmax([np.nanmax(flux[filter_area]), np.nanmax(temp[filter_area])])
    (model_plot,) = ax.plot(
        wl, temp, color="red", linestyle="--", linewidth=2, label="Model Spectrum"
    )

    plt.subplots_adjust(bottom=0.30, right=0.85)
    ax.set_xlabel("Wavelength (Å)")
    ax.set_ylabel("Flux (erg/s/cm²/Å)")
    ax.set_xlim(3650, np.max(wl))
    ax.set_ylim(0, ymax * 1.1)
    ax.set_title(
        f"Spectrum for {objname}\n(GAIA ID: {source_id})\nRedshift (z): {init_z:.5f}, Av: {init_Av:.5f}"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax_z_slider = plt.axes([0.15, 0.20, 0.65, 0.03])
    ax_z_text = plt.axes([0.82, 0.20, 0.1, 0.03])
    ax_Av_slider = plt.axes([0.15, 0.15, 0.65, 0.03])
    ax_Av_text = plt.axes([0.82, 0.15, 0.1, 0.03])

    z_slider = Slider(
        ax_z_slider, "Redshift (z)", 0.0, 8.0, valinit=init_z, valfmt=None
    )
    z_slider.valtext.set_visible(False)
    z_textbox = TextBox(ax_z_text, "", initial=f"{init_z:.5f}")
    z_textbox.on_submit(lambda val: z_slider.set_val(float(val)))
    Av_slider = Slider(ax_Av_slider, "Av", 0.0, 3.0, valinit=init_Av, valfmt=None)
    Av_slider.valtext.set_visible(False)
    Av_textbox = TextBox(ax_Av_text, "", initial=f"{init_Av:.5f}")
    Av_textbox.on_submit(lambda val: Av_slider.set_val(float(val)))

    ax_button = plt.axes([0.45, 0.05, 0.1, 0.05])
    btn_confirm = Button(
        ax_button, "Save&Close", color="lightgoldenrodyellow", hovercolor="0.975"
    )

    ax_radio = plt.axes([0.85, 0.85, 0.15, 0.12], facecolor="#f0f0f0")
    radio = RadioButtons(ax_radio, ("QSO", "GALAXY", "STAR", "UNKNOWN"), active=0)

    ui_axes = [ax_z_slider, ax_z_text, ax_Av_slider, ax_Av_text, ax_radio, ax_button]
    final_params = {}
    final_params["Type"] = "QSO"

    def update(val):
        z = z_slider.val
        Av = Av_slider.val
        model_updated = compoM_bspl(wl / (1 + z)) * ext_model.extinguish(
            wl * u.AA, Av=Av
        )
        model_factor = np.nanmean(flux[fit_area]) / np.nanmean(model_updated[fit_area])
        model_updated *= model_factor
        model_plot.set_ydata(model_updated)
        ymax = np.nanmax(
            [np.nanmax(flux[filter_area]), np.nanmax(model_updated[filter_area])]
        )
        ax.set_ylim(0, ymax * 1.1)
        ax.set_title(
            f"Spectrum for {objname}\n(GAIA ID: {source_id})\nRedshift (z): {z:.5f}, Av: {Av:.5f}, Type: {final_params['Type']}"
        )
        z_textbox.set_val(f"{z:.5f}")
        Av_textbox.set_val(f"{Av:.5f}")
        fig.canvas.draw_idle()

    z_slider.on_changed(update)
    Av_slider.on_changed(update)
    z_textbox.on_submit(lambda val: z_slider.set_val(float(val)))
    Av_textbox.on_submit(lambda val: Av_slider.set_val(float(val)))

    def update_type(label):
        final_params["Type"] = label
        if label != "QSO":
            model_plot.set_visible(False)
            ymax = np.nanmax(flux[filter_area])
            ax.set_ylim(0, ymax * 1.1)
            ax.set_title(
                f"Spectrum for {objname}\n(GAIA ID: {source_id}),type: {final_params['Type']}\nRedshift (z) and Av are not applicable for non-QSO types"
            )
        else:
            model_plot.set_visible(True)
        fig.canvas.draw_idle()

    radio.on_clicked(update_type)

    def confirm(event):
        z = z_slider.val
        Av = Av_slider.val
        final_params["Redshift"] = z
        final_params["Av"] = Av
        if final_params["Type"] not in ["QSO", "UNKNOWN"]:
            final_params["Redshift"] = 0.0
            final_params["Av"] = 0.0
        if save_path is not None:
            save_name = re.sub(r"\.\d+", "", objname)
            for ui_ax in ui_axes:
                ui_ax.set_visible(False)
            print(f"Saving fitted plot to {os.path.join(save_path, save_name+'.png')}")
            fig.savefig(
                os.path.join(save_path, save_name + ".png"),
                dpi=300,
                transparent=True,
                bbox_inches="tight",
            )
        print(
            f"Confirmed values for {objname} (GAIA ID: {source_id}): Redshift (z) = {final_params['Redshift']:.5f}, Av = {final_params['Av']:.5f}"
        )
        plt.close(fig)

    btn_confirm.on_clicked(confirm)
    plt.show()

    return final_params


if __name__ == "__main__":
    base_path = os.path.dirname(os.path.abspath(__file__))
    spec_path = os.path.join(base_path, "results")
    os.makedirs(spec_path, exist_ok=True)

    spec_list_all = pd.read_csv(os.path.join(base_path, "SNAQS_observed_targets.csv"))
    auto_results = pd.read_csv(
        os.path.join(base_path, "auto_fit_results/fit_results.txt"),
        skiprows=3,
        names=[
            "source_id",
            "sdss_name",
            "z_fit",
            "z_err",
            "Av_fit",
            "Av_err",
        ],
    )

    res_file = os.path.join(spec_path, "manual_fit_results.csv")
    if not os.path.exists(res_file):
        with open(res_file, "w") as f:
            f.write("source_id,sdss_name,z_fit,Av_fit,Type,specfile\n")

    manual_results = pd.read_csv(res_file)
    known_manual_ids = manual_results["source_id"][manual_results["Type"] != "UNKNOWN"].tolist()
    spec_list = spec_list_all[
        (~spec_list_all["source_id"].isin(known_manual_ids))
    ]

    f = open(res_file, "a")

    for row in spec_list.itertuples():
        file_path = os.path.join(base_path, row.specfile)
        init_res = auto_results[auto_results["source_id"] == row.source_id]
        if not init_res.empty:
            init_z = float(init_res["z_fit"].values[0])
            init_Av = float(init_res["Av_fit"].values[0])
        else:
            init_z = 0.0
            init_Av = 0.0

        spec = read_spectrum(file_path)
        params = plot_spectrum(
            spec,
            objname=row.sdss_name,
            init_z=init_z,
            init_Av=init_Av,
            source_id=row.source_id,
            save_path=spec_path,
        )

        print(
            f"Final parameters for {row.sdss_name} (GAIA ID: {row.source_id}): {params['Redshift']:.5f},{params['Av']:.5f},{params['Type']}"
        )
        f.write(
            f"{row.source_id},{row.sdss_name},{params['Redshift']:.5f},{params['Av']:.5f},{params['Type']},{row.specfile}\n"
        )

    f.close()
    print(f"Manual fitting results updated and saved to {res_file}")

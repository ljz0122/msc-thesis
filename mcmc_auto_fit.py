import numpy as np
import matplotlib.pyplot as plt
import os
import re
import pandas as pd
from dust_extinction.grain_models import Y24
import astropy.units as u
from spectres import spectres
import pickle
import emcee
import corner
from multiprocessing import Pool
from tqdm import tqdm
import gc
import time
import logging

logging.basicConfig(level=logging.ERROR)

with open("/Users/ericljz/astro/MSc-Thesis/data/interpolated_compoM.pkl", "rb") as f:
    compoM_bspl = pickle.load(f)
ext_model = Y24()


def read_spectrum(data_path, ins="alfosc"):
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

    threshold = 2e-15
    spec_filtered = spec[(spec["flux"] <= threshold)]
    spec_filtered = spec_filtered[spec_filtered["flux"] > 0]
    spec_filtered = spec_filtered[spec_filtered["err_flux"] > 1e-20]

    if ins == "alfosc":
        new_wl = np.arange(3800, 8750, 3.0)
        sigma_clip = 1.0
    elif ins == "osiris":
        new_wl = np.arange(3600, 7000, 2.0)
        sigma_clip = 1.5

    spec_cut = get_emission_lines_only(
        spec_filtered["wavelength"].values,
        spec_filtered["flux"].values,
        method="fill",
        sigma_clip=sigma_clip,
    )
    spec_cut = pd.DataFrame(
        {
            "wavelength": spec_filtered["wavelength"].values,
            "flux": spec_cut,
            "err_flux": spec_filtered["err_flux"].values,
        }
    )
    threshold = 2e-16
    index_to_remove = spec_cut[spec_cut["wavelength"].between(7590, 7690)].index
    spec_cut = spec_cut.drop(index_to_remove)
    index_to_remove = spec_cut[spec_cut["wavelength"].between(6860, 6915)].index
    spec_cut = spec_cut.drop(index_to_remove)
    spec_cut = spec_cut[(spec_cut["flux"] <= threshold)]
    spec_cut = spec_cut.reset_index(drop=True)
    spec_cut = spec_cut.dropna().reset_index(drop=True)

    binned_flux, binned_err = spectres(
        new_wl,
        spec_cut["wavelength"].values,
        spec_cut["flux"].values,
        spec_cut["err_flux"].values,
    )

    binned_flux = smooth(binned_flux, box_pts=3)
    binned_spec = pd.DataFrame(
        {"wavelength": new_wl, "flux": binned_flux, "err_flux": binned_err}
    )
    binned_spec = binned_spec.dropna().reset_index(drop=True)

    return spec_filtered, binned_spec

def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def get_model(theta, wave_obs):
    z, Av = theta

    wave_rest = wave_obs / (1 + z)
    model = compoM_bspl(wave_rest) * ext_model.extinguish(wave_rest * u.AA, Av=Av)
    return model


def get_emission_lines_only(wave, flux, method="extract", sigma_clip=1.5):
    """
    Removes absorption lines, keeping only emission lines.

    method:
        - 'fill': Fills in absorption troughs, preserving the overall spectral shape.
        - 'extract': Subtracts the continuum, leaving only emission peaks with a baseline of 0.
    sigma_clip: Threshold in standard deviations below the continuum to identify absorption lines.
    """

    # --- Step 1: Estimate the continuum ---
    # A simpler, more direct continuum estimation:
    # Assume emission lines are sparse and absorption lines are narrow.
    # We use a simple polynomial fit, excluding points that are too high (emission) or too low (absorption).
    p = np.polyfit(wave, flux, 5)  # 5th order polynomial
    continuum_poly = np.polyval(p, wave)

    # --- Step 2: Identify and process ---

    # Calculate residuals (spectrum after continuum subtraction)
    residuals = flux - continuum_poly

    # Calculate the local noise level
    # Take the standard deviation of the relatively flat part of the residuals
    noise_level = np.std(residuals[(residuals > -0.1) & (residuals < 0.1)])
    if np.isnan(noise_level) or noise_level == 0:
        noise_level = np.std(residuals)

    if method == "fill":
        # Option A: Filling method
        # Create a copy
        clean_flux = flux.copy()
        # Find all places significantly below the continuum (absorption lines)
        absorption_mask = flux < (continuum_poly - sigma_clip * noise_level)
        # Fill these troughs with the continuum value
        clean_flux[absorption_mask] = continuum_poly[absorption_mask]
        # (Optional) Add some random noise to make the filled area look natural, not like a rigid line
        # clean_flux[absorption_mask] += np.random.normal(0, noise_level, np.sum(absorption_mask))
        return clean_flux

    elif method == "extract":
        # Option B: Extraction method (Pure Emission)
        # 1. Subtract the continuum
        pure_emission = flux - continuum_poly

        # 2. Set all values less than 0 (or less than -1sigma) to 0
        # This way, absorption lines (negative values) and background noise will disappear
        pure_emission[pure_emission < 0] = 0

        return pure_emission


def log_likelihood(theta, x, y, yerr):
    z, Av = theta

    if not (0.0 < Av < 3.0):
        return -np.inf
    if not (-0.5 < z < 7.5):
        return -np.inf

    try:
        shape_model = get_model(theta, x)
        if not np.all(np.isfinite(shape_model)):
            return -np.inf

    except Exception:
        return -np.inf

    ivar = 1.0 / yerr**2
    numerator = np.sum(y * shape_model * ivar)
    denominator = np.sum(shape_model**2 * ivar)
    if denominator <= 0:
        return -np.inf
    amp = numerator / denominator
    if amp <= 0:
        return -np.inf

    model = amp * shape_model
    chi2 = np.sum((y - model) ** 2 * ivar)

    if not np.isfinite(chi2):
        return -np.inf

    return -0.5 * chi2


def init_guess(spec):
    wave = spec["wavelength"].values
    flux = spec["flux"].values
    err = spec["err_flux"].values

    z_range = np.arange(0.0, 5.0, 1.0)
    chi2_list = []
    best_z = None

    test_Av = 0.0

    for z in z_range:
        try:
            S = get_model([z, test_Av], wave)
            mask = (err > 0) & np.isfinite(flux) & np.isfinite(S)
            if np.sum(mask) < 10:  # Too few valid points
                chi2_list.append(np.inf)
                continue

            y = flux[mask]
            err = err[mask]
            S_clean = S[mask]
            ivar = 1.0 / err**2

            # Analytically solve for the amplitude
            num = np.sum(y * S_clean * ivar)
            den = np.sum(S_clean**2 * ivar)
            if den <= 0:
                chi2_list.append(np.inf)
                continue

            A_hat = num / den
            if A_hat < 0:  # Negative amplitude, non-physical
                chi2_list.append(np.inf)
                continue

            # Calculate Chi2
            model = A_hat * S_clean
            chi2 = np.sum((y - model) ** 2 * ivar)
            chi2_list.append(chi2)

        except Exception:
            # Catch any interpolation out-of-bounds errors
            chi2_list.append(np.inf)

    best_idx = np.argmin(chi2_list)
    best_z = z_range[best_idx]

    return [best_z, test_Av]


def fit_spectrum(spec, init_theta, nwalkers=64, nsteps=1000, pool=None):
    """
    Interactive fitting of the quasar spectrum using sliders for redshift and scaling.

    Parameters
    ----------
    spec : DataFrame
        DataFrame containing 'wavelength', 'flux', and 'err_flux' columns of the observed spectrum.
    init_theta : list
        List containing initial guesses for [z, Av, log_amp].
    nwalkers : int, optional
        Number of walkers for MCMC. Default is 128.
    nsteps : int, optional
        Number of steps for MCMC. Default is 1000.

    Returns
    -------
    output : OptimizeResult
        The optimization result containing fitted parameters and statistics.
    """
    new_wl = spec["wavelength"].values
    binned_flux = spec["flux"].values
    binned_err = spec["err_flux"].values

    rng = np.random.default_rng()

    ndim = 2  # z, Av
    pos = np.zeros((nwalkers, ndim))
    pos[:, 0] = init_theta[0] + rng.normal(loc=0, scale=3, size=nwalkers)  # z
    pos[:, 1] = init_theta[1] + 0.5 * rng.uniform(size=nwalkers)  # Av

    pos[:, 0] = np.clip(pos[:, 0], 0.0, 6.0)
    pos[:, 1] = np.clip(pos[:, 1], 0.0, 3.0)

    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_likelihood,
        args=(new_wl, binned_flux, binned_err),
        pool=pool,
    )
    sampler.run_mcmc(pos, nsteps, progress=True)

    samples = sampler.get_chain(discard=int(nsteps * 0.3), thin=15, flat=True)
    return samples


def get_best_amp(theta, x, y, yerr):
    shape_model = get_model(theta, x)

    ivar = 1.0 / yerr**2
    numerator = np.sum(y * shape_model * ivar)
    denominator = np.sum(shape_model**2 * ivar)
    if denominator <= 0:
        return ValueError("Denominator in amplitude calculation is non-positive.")

    amp = numerator / denominator
    if amp <= 0:
        return ValueError("Calculated amplitude is non-positive.")

    return amp


def plot_fit(
    spec, params, perrs, save_path=None, object_name=None, hide_plot=False, gaia_id=None
):
    """
    Plots the observed spectrum and the fitted model.

    Parameters
    ----------
    spec : DataFrame
        DataFrame containing 'wavelength', 'flux', and 'err_flux' columns of the observed spectrum.
    params : list
        List containing fitted parameters [z, Av].
    save_path : str, optional
        Path to save the plot. If None, the plot is not saved.
    object_name : str, optional
        Name of the object for plot title. If None, uses 'Quasar Spectrum'.
    """

    plt.close("all")

    if object_name is None:
        object_name = "Quasar Spectrum"

    wavelength = spec["wavelength"].values
    flux = spec["flux"].values
    err_flux = spec["err_flux"].values
    spec_filter = spec[(spec["wavelength"] > 3750) & (spec["wavelength"] < 7000)]

    shape_model = get_model(params, wavelength)
    amp = get_best_amp(params, wavelength, flux, err_flux)
    model = amp * shape_model

    z_fit, Av_fit = params
    z_err, Av_err = perrs

    fig, ax = plt.subplots()
    ax.plot(wavelength, flux, label="Observed Spectrum", color="blue", linewidth=0.8)
    ax.fill_between(
        wavelength,
        flux - err_flux,
        flux + err_flux,
        color="lightgray",
        alpha=0.75,
        label="Error",
    )
    ax.plot(
        wavelength,
        model,
        label="Fitted Model",
        color="red",
        linewidth=0.8,
    )

    ax.set_xlabel("Wavelength (Å)")
    ax.set_ylabel("Flux [erg/s/cm²/Å]")
    ax.set_xlim(np.min(wavelength), np.max(wavelength))
    ax.set_ylim(
        0,
        1.1 * np.max([np.nanmax(spec_filter["flux"]), np.nanmax(model)]),
    )
    if gaia_id is not None:
        ax.set_title(
            f"{object_name} (GAIA ID: {gaia_id}) \n Fitted Model ($z={z_fit:.3f}^{{+{z_err[0]:.3f}}}_{{{z_err[1]:.3f}}}, Av={Av_fit:.4f}^{{+{Av_err[0]:.3f}}}_{{{Av_err[1]:.3f}}}$)"
        )
    else:
        ax.set_title(
            f"{object_name} \n Fitted Model ($z={z_fit:.3f}^{{+{z_err[0]:.3f}}}_{{{z_err[1]:.3f}}}, Av={Av_fit:.4f}^{{+{Av_err[0]:.3f}}}_{{{Av_err[1]:.3f}}}$)"
        )
    ax.legend()

    if save_path is not None:
        save_name = re.sub(r"\.\d+", "", object_name)
        tqdm.write(f"Saving fitted plot to {os.path.join(save_path, save_name+'.png')}")
        fig.savefig(
            os.path.join(save_path, save_name + ".png"), dpi=300, transparent=True
        )

    if not hide_plot:
        plt.show()


def main(
    data_path,
    init_theta=None,
    save_path=None,
    object_name=None,
    hide_plot=False,
    ins="alfosc",
    gaia_id=None,
    pool=None,
):

    if ins not in ["alfosc", "osiris"]:
        tqdm.write("Invalid instrument type. Use default 'alfosc'.")
        ins = "alfosc"

    # Set object name
    if object_name is None:
        object_name = os.path.splitext(os.path.basename(data_path))[0]
        object_name = object_name.replace("_combined", "")

    tqdm.write(f"Fitting spectrum for object: {object_name}")
    # Load spectrum
    spec, binned_spec = read_spectrum(data_path, ins=ins)

    """    
    plt.figure(figsize=(10, 6))
    plt.plot(
        spec["wavelength"],
        spec["flux"],
        label="Original Spectrum",
        color="blue",
        linewidth=0.8,
    )
    plt.plot(
        binned_spec["wavelength"],
        binned_spec["flux"],
        label="Binned Spectrum",
        color="red",
        linewidth=0.8,
    )
    plt.xlabel("Wavelength (Å)")
    plt.ylabel("Flux [erg/s/cm²/Å]")
    plt.ylim(0, 2e-16)
    plt.xlim(3800, 8700)
    plt.title(f"{object_name} - Original vs Binned Spectrum")
    plt.legend()
    plt.show()
    """
    # Initial guesses
    if init_theta is not None:
        tqdm.write(f"Using initial guesses from command line: {init_theta}")
    else:
        init_theta = init_guess(binned_spec)  # Default initial guesses for [z, Av]

    tqdm.write(f"Initial guesses: z = {init_theta[0]:.5f}, Av = {init_theta[1]:.5f}")

    # Fit model
    if pool is None:
        cpu_count = 4
        tqdm.write(f"Using {cpu_count} CPU cores for MCMC fitting.")
        with Pool(cpu_count) as pool:
            output = fit_spectrum(
                binned_spec, init_theta=init_theta, nwalkers=64, nsteps=5000, pool=pool
            )
    else:
        output = fit_spectrum(
            binned_spec, init_theta=init_theta, nwalkers=64, nsteps=5000, pool=pool
        )

    tqdm.write("MCMC fitting completed.")

    tqdm.write("Calculating best-fit parameters and uncertainties...")
    tqdm.write(f"Total samples shape: {output.shape}")

    labels = ["Redshift (z)", "Extinction (Av)"]
    params = []
    perr = []
    for i in range(output.shape[1]):  # Iterate over each column (each parameter)
        mcmc = np.percentile(output[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        tqdm.write(f"{labels[i]}: {mcmc[1]:.5f} (+{q[1]:.5f} / -{q[0]:.5f})")
        params.append(mcmc[1])
        perr.append([q[1], -q[0]])

    # Save Results
    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        corner_path = os.path.join(save_path, "mcmc_corners")
        if not os.path.exists(corner_path):
            os.makedirs(corner_path)
        results_file = os.path.join(save_path, f"fit_results.txt")
        if not os.path.isfile(results_file):
            with open(results_file, "w", encoding="utf-8") as f:
                f.write("Fitting Results\n")
                f.write("====================\n")
                f.write(
                    f"Gaia ID,Object,Redshift (z),Redshift Error (σz),Extinction (Av),Extinction Error (σAv)\n"
                )

        with open(results_file, "a", encoding="utf-8") as f:
            f.write(
                f"{gaia_id},{object_name},{params[0]:.8f},+{perr[0][0]:.8f}/{perr[0][1]:.8f},{params[1]:.8f},+{perr[1][0]:.8f}/{perr[1][1]:.8f}\n"
            )
        tqdm.write(f"Fit results saved to {results_file}")

    # Plot fit
    fig = corner.corner(
        output,
        labels=["Redshift $z$", "Extinction $A_V$"],
        show_titles=True,
        title_fmt=".3f",
        quantiles=[0.16, 0.5, 0.84],
    )
    if save_path is not None:
        save_name = re.sub(r"\.\d+", "", object_name)
        fig.savefig(
            os.path.join(corner_path, save_name + "_corner.png"),
            dpi=300,
            transparent=True,
        )
    if not hide_plot:
        plt.show()

    plot_fit(spec, params, perr, save_path, object_name, hide_plot, gaia_id)


def setup_logger(log_file="processing.log"):
    logger = logging.getLogger("SNAQS_Logger")
    logger.setLevel(logging.INFO)

    # 防止重复添加 Handler (Jupyter 或多次运行时)
    if logger.hasHandlers():
        logger.handlers.clear()

    # 格式：时间 - 级别 - 消息
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Handler A: 写入文件 (processing.log)
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


if __name__ == "__main__":
    base_path = os.path.dirname(os.path.abspath(__file__))
    spec_path = os.path.join(base_path, "auto_fit_results")

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
    
    spec_list = spec_list_all[~spec_list_all["source_id"].isin(auto_results["source_id"])]

    log = setup_logger("processing.log")
    cpu_count = 4
    print(f"Using {cpu_count} CPU cores for MCMC fitting.")

    for row in tqdm(
        spec_list.itertuples(),
        total=len(spec_list),
        position=0,
        desc="Processing spectra",
        leave=True,
    ):

        gaia_id = row.source_id
        file_path = os.path.join(base_path, row.specfile.strip())
        ins = row.specfile.split("/")[0]

        if ins == "not":
            ins = "alfosc"

        try:
            with Pool(cpu_count) as pool:
                main(
                    data_path=file_path,
                    object_name=str(row.sdss_name),
                    save_path=spec_path,
                    hide_plot=True,
                    gaia_id=gaia_id,
                    ins=ins,
                    pool=pool,
                )

        except Exception as e:
            tqdm.write(f"Error processing spectrum for {row.sdss_name}: {e}")
            log.error(f"Error processing spectrum for {row.sdss_name}: {e}")
            continue

        gc.collect()
        time.sleep(1)  # Optional: Add a small delay to prevent overwhelming the system

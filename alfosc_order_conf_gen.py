# This Script make ALFOSC files in order and generates configuration files of PyLongSlit for reduction of ALFOSC long-slit spectra
import shutil
from pathlib import Path
import simplejson as json
from decimal import Decimal
from dateutil.parser import parse
import dfitspy
import numpy as np
from collections import defaultdict

import questionary
from questionary import Choice, Style

"""
This script is designed to organize ALFOSC long-slit spectral data and generate configuration files for PyLongSlit reduction. It performs the following tasks:
1. Reads FITS files from a specified directory and classifies them into categories (bias, flat, arc, science, and standard star) based on their headers.
2. Copies the classified files into organized folders for easier access during reduction.
3. Provides an interactive selection menu for choosing the standard star file and its corresponding flux file.
4. Generates a JSON configuration file for PyLongSlit, containing paths to the data folders, exposure times, airmass values, and grouping information for combining science frames.
5. Handles user input for observation date and allows skipping configuration generation if desired.

Usage:
1. Ensure you have the required dependencies installed (e.g., dfitspy, questionary, simplejson).
2. Place the template configuration file (alfosc_template.json) in the same directory as this script or update the path accordingly.
3. Modify the configuration template as needed to match the expected setting for PyLongSlit. (Such as setting of the path to extiction curve file, etc.)
4. Run the script with appropriate command-line arguments for base path, original data path, output path, and standard star path.

Example:
python alfosc_order_conf_gen.py --base_path ../data \
    --ori_path ../data/original/2024-06-01/alfosc \
    --output_path ../data/reduction/alfosc/2024-06-01 \
    --std_path ../data/database/stds \
    --date 2024-06-01
    
Note:
- The script assumes a specific structure for the FITS headers to classify files. Ensure that your FITS files contain the necessary header keywords (e.g., OBJECT, EXPTIME, AIRMASS) before running the script.
- The generated configuration file will be saved in the specified config path with the name format "alfosc/YYYY-MM-DD.json". Make sure the config directory exists or can be created by the script.
- The standard flux file default path is set to "${base_path}/database/stds". You can change this path by providing the --std_path argument. Ensure that this directory contains the appropriate standard star flux files for selection during the interactive menu.
"""


detwin = "[801:1300, 1:2102]"  # ALFOSC Grism#4's detector window size, can be modified if needed


def select_observation_interactive(data_input):
    """
    Generates an interactive table-like selection menu.

    Args:
        data_input (list[dict]): A list of dictionaries containing FITS header info.

    Returns:
        dict: The selected dictionary (row). Returns None if cancelled or empty.
    """
    if not data_input:
        print("Error: The data list is empty.")
        return None

    data_list = []
    if isinstance(data_input, dict):
        for filename, info in data_input.items():
            # Create a copy to avoid modifying original data
            row = info.copy()
            # Inject the filename (key) into the dictionary as a value
            row["FILENAME"] = filename
            data_list.append(row)

    # 1. Define the columns to display (Order matters)
    columns = ["FILENAME", "DETWIN1", "EXPTIME", "OBJECT"]

    # 2. Calculate dynamic column widths
    # We iterate through headers and data to find the max width for each column
    col_widths = {}
    for col in columns:
        max_len = len(col)  # Start with header length
        for row in data_list:
            val = row.get(col, "")

            # Pre-format floats for width calculation to avoid over-estimation
            if isinstance(val, float):
                val_str = f"{val:.5f}"
            else:
                val_str = str(val)

            max_len = max(max_len, len(val_str))

        # Add padding (2 spaces)
        col_widths[col] = max_len + 2

    # 3. Create the format string (e.g., "{:<20} {:<10} ...")
    # This creates the template for aligning text
    fmt_str = " ".join([f"{{:<{col_widths[col]}}}" for col in columns])

    # 4. Generate Header and Separator
    header_text = fmt_str.format(*columns)
    separator = "-" * len(header_text)

    # 5. Build the Choices list
    choices = []
    for row in data_list:
        display_values = []
        for col in columns:
            val = row.get(col, "")

            # Truncate long floats for display purposes
            if isinstance(val, float):
                val = f"{val:.5f}"

            display_values.append(str(val))

        # Format the row string
        display_text = fmt_str.format(*display_values)

        # Create Choice: 'title' is what user sees, 'value' is what is returned
        choices.append(Choice(title=display_text, value=row))

    # 6. Define custom style (Optional, makes it look better)
    custom_style = Style(
        [
            ("qmark", "fg:#673ab7 bold"),  # Color for '?'
            ("question", "bold"),  # Color for the prompt text
            ("answer", "fg:#f44336 bold"),  # Color for the selected answer
            ("pointer", "fg:#673ab7 bold"),  # Color for the selection pointer
            ("highlighted", "fg:#673ab7 bold"),  # Color for the highlighted row
        ]
    )

    # 7. Show the interactive menu
    # We embed the header in the message so it stays fixed at the top
    result = questionary.select(
        f"Please select an standard observation file:\n\n{header_text}\n{separator}",
        choices=choices,
        qmark="?",
        pointer=">",
        use_indicator=True,
        style=custom_style,
    ).ask()

    return result


def select_standard_file(std_name, std_path):
    """
    Interactive selection of standard star file.
    """
    if not std_path.exists() or not std_path.is_dir():
        print("No standard star files found.")
        return None

    std_path_files = list(std_path.glob("*"))
    if not std_path_files:
        print("No standard star files found.")
        return None

    choice = []
    for file in std_path_files:
        choice.append(Choice(title=file.name, value=file))

    selected_std_file = questionary.select(
        f"Please select a standard star file for {std_name}:",
        choices=choice,
        qmark="?",
        pointer=">",
        use_indicator=True,
    ).ask()

    return selected_std_file


def make_data_folders(base_path):
    """
    Create necessary data folders if they do not exist.
    """
    folders = ["bias", "flat", "arc", "sci", "std"]
    for folder in folders:
        folder_path = base_path / folder
        folder_path.mkdir(parents=True, exist_ok=True)

    return {folder: base_path / folder for folder in folders}


def read_and_classify_files(ori_path, data_folders):
    """
    Read FITS files and classify them into appropriate folders.
    """

    science_files = dfitspy.get_files(["all"], dire=ori_path)
    calib_path = ori_path / "calib"
    if calib_path.exists():
        calib_files = dfitspy.get_files(["all"], dire=calib_path)
    else:
        calib_files = science_files
        calib_path = ori_path

    science_list = dfitspy.dfitsort(
        science_files,
        ["OBJECT", "EXPTIME", "AIRMASS", "IMAGETYP", "DETWIN1"],
        HDU=0,
        grepping=["OBJECT", f"{detwin}"],
    )
    bias_list = dfitspy.dfitsort(
        calib_files,
        ["OBJECT", "EXPTIME", "AIRMASS", "IMAGETYP", "DETWIN1"],
        HDU=0,
        grepping=["BIAS", f"{detwin}"],
    )
    flat_list = dfitspy.dfitsort(
        science_files,
        ["OBJECT", "EXPTIME", "AIRMASS", "IMAGETYP", "DETWIN1"],
        HDU=0,
        grepping=["FLAT", f"{detwin}"],
    )
    arc_list = dfitspy.dfitsort(
        science_files,
        ["OBJECT", "EXPTIME", "AIRMASS", "IMAGETYP", "DETWIN1"],
        HDU=0,
        grepping=["WAVE", f"{detwin}"],
    )
    std_list = dfitspy.dfitsort(
        calib_files,
        ["OBJECT", "EXPTIME", "AIRMASS", "IMAGETYP", "DETWIN1"],
        HDU=0,
        grepping=["STD", f"{detwin}"],
    )

    dfitspy.dfitsort_view(science_list)
    dfitspy.dfitsort_view(bias_list)
    dfitspy.dfitsort_view(flat_list)
    dfitspy.dfitsort_view(arc_list)

    for file in science_list.keys():
        src_p = Path(ori_path) / file
        dst_folder_p = Path(data_folders["sci"])
        dst_p = dst_folder_p / file
        if not dst_p.exists():
            shutil.copy(src_p, dst_p)
    for file in bias_list.keys():
        src_p = Path(calib_path) / file
        dst_folder_p = Path(data_folders["bias"])
        dst_p = dst_folder_p / file
        if not dst_p.exists():
            shutil.copy(src_p, dst_p)
    for file in flat_list.keys():
        src_p = Path(ori_path) / file
        dst_folder_p = Path(data_folders["flat"])
        dst_p = dst_folder_p / file
        if not dst_p.exists():
            shutil.copy(src_p, dst_p)
    for file in arc_list.keys():
        src_p = Path(ori_path) / file
        dst_folder_p = Path(data_folders["arc"])
        dst_p = dst_folder_p / file
        if not dst_p.exists():
            shutil.copy(src_p, dst_p)

    if std_list:
        dfitspy.dfitsort_view(std_list)
        std_file = select_observation_interactive(std_list)
        if std_file:
            src_p = calib_path / std_file["FILENAME"]
            dst_folder_p = Path(data_folders["std"])
            dst_p = dst_folder_p / std_file["FILENAME"]
            if not dst_p.exists():
                shutil.copy(src_p, dst_p)
    else:
        std_file = None

    print("File classification and copying completed.")

    return science_list, std_file


def generate_config_file(
    data_folders, config_path, output_path, science_list, std_file, std_path
):
    """
    Generate configuration file for PyLongSlit.
    """

    with open("alfosc_template.json", "r", encoding="utf-8") as f:
        temp = json.load(f)

    temp["bias"]["bias_dir"] = str(data_folders["bias"].resolve())
    temp["flat"]["flat_dir"] = str(data_folders["flat"].resolve())
    temp["arc"]["arc_dir"] = str(data_folders["arc"].resolve())
    temp["science"]["science_dir"] = str(data_folders["sci"].resolve())
    temp["output"]["out_dir"] = str(output_path.resolve())
    temp["standard"]["standard_dir"] = str(data_folders["std"].resolve())

    temp["standard"]["exptime"] = 1.0
    temp["standard"]["airmass"] = 1.0

    temp["science"]["files"] = {file: {} for file in science_list.keys()}

    for file in science_list.keys():
        temp["science"]["files"][file] = {
            "exptime": Decimal(science_list[file]["EXPTIME"]),
            "airmass": Decimal(science_list[file]["AIRMASS"]),
        }

    combine_group = defaultdict(list)
    for filename, header_info in science_list.items():
        obj_name = header_info.get("OBJECT", "Unknown")  # 获取 OBJECT 值，防止缺失
        combine_group[obj_name].append(filename)

    for obj_name, files in combine_group.items():
        temp["combine"][obj_name] = files

    if std_file:
        temp["standard"]["file"] = std_file["FILENAME"]
        temp["standard"]["exptime"] = Decimal(std_file["EXPTIME"])
        temp["standard"]["airmass"] = Decimal(std_file["AIRMASS"])
        std_mag = select_standard_file(std_file["OBJECT"], std_path=std_path)
        if std_mag:
            temp["standard"]["flux_file_path"] = str(std_mag.resolve())
            temp["standard"]["starname"] = str(std_file["OBJECT"])
    else:
        temp["standard"]["skip_standard"] = True

    if config_path.exists():
        config_path.unlink()

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(temp, f, indent=4)

    return config_path


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate ALFOSC ordered configuration for PyLongSlit."
    )
    parser.add_argument(
        "--base_path",
        type=str,
        required=False,
        default="../data",
        help="Base path for data directories.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=False,
        default="../config",
        help="Path to save the generated configuration file.",
    )
    parser.add_argument(
        "--ori_path",
        type=str,
        required=False,
        default="",
        help="Path to original data files.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=False,
        default="",
        help="Path to save the reduced data.",
    )
    parser.add_argument(
        "--std_path",
        type=str,
        required=False,
        default="",
        help="Path to standard star flux files.",
    )
    parser.add_argument("--date", type=str, required=True, help="Observation date")
    parser.add_argument(
        "--skip_config", action="store_true", help="Skip configuration generation"
    )

    args = parser.parse_args()

    try:
        obsdate = parse(args.date).strftime("%Y-%m-%d")
    except ValueError:
        print("Error: Invalid date format. Treat as raw string.")
        obsdate = args.date

    base_path = Path(args.base_path) / obsdate
    ori_path = (
        Path(args.ori_path)
        if args.ori_path
        else Path(args.base_path) / "original" / obsdate / "alfosc"
    )
    output_path = (
        Path(args.output_path)
        if args.output_path
        else Path(args.base_path) / "reduction" / "alfosc" / obsdate
    )
    config_path = Path(args.config_path) / "alfosc" / f"{obsdate}.json"
    std_path = (
        Path(args.std_path)
        if args.std_path
        else Path(args.base_path) / "database" / "stds"
    )

    data_folders = make_data_folders(base_path)
    science_list, std_list = read_and_classify_files(ori_path, data_folders)
    print(f"Flies classified into folders under {base_path}/")

    if not args.skip_config:
        config_file = generate_config_file(
            data_folders, config_path, output_path, science_list, std_list, std_path
        )
        print(f"Configuration file generated at: {config_file}")


if __name__ == "__main__":
    main()

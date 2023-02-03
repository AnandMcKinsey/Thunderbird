"""Module contains functions for reading data catalog and parameters."""
import os
import yaml


def load_catalog_params(
        base: str,
        path: str
) -> dict:
    """
    Function to load parameters file in yaml format into the environment.

    Args:

        base: str
            Base location to project where the notebooks, src and the conf folders are present.

        path: str
            Path to directory where catalogs/parameters are present.

    Returns:

        final_params: dict
            Dictionary of all parameter files present within the conf/parameters
            folder ending with .yaml.

    """
    final_dict = {}
    file_paths = [i for i in os.listdir(f"{base}/{path}") if i.endswith("yaml")]

    for f in file_paths:

        with open(f"{base}/{path}/{f}", "r", encoding="utf-8") as f_obj:
            my_dict = yaml.load(f_obj, Loader=yaml.FullLoader)

        final_dict.update(my_dict)

    return final_dict

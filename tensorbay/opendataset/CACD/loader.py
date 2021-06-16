#
# Copyright 2021 Graviti. Licensed under MIT License.
#
# pylint: disable=invalid-name
# pylint: disable=missing-module-docstring
# pylint: disable=line-too-long

import os
from typing import Any, Dict, Tuple

import numpy as np
from h5py import File

from ...dataset import Data, Dataset
from ...label import Classification

DATASET_NAME = "CACD"


def CACD(path: str) -> Dataset:
    """Dataloader of `Cross-Age Celebrity Dataset (CACD)`_ dataset.

    .. _Cross-Age Celebrity Dataset (CACD): https://bcsiriuschen.github.io/CARC/

    The file structure should be like::

        <path>
            <imagename>.jpg
            ...

    Arguments:
        path: The root directory of the dataset.

    Returns:
        Loaded :class:`~tensorbay.dataset.dataset.Dataset` instance.
    """
    dataset = Dataset(DATASET_NAME)
    # dataset.load_catalog(os.path.join(os.path.dirname(__file__), "catalog.json"))
    segment = dataset.create_segment()
    img_files = sorted(os.listdir(os.path.join(path, "CACD2000")))
    attributes = _get_attribute_map(os.path.join(path, "celebrity2000.mat"))
    for img in img_files:
        category, attribute = attributes[img]
        img_data = Data(os.path.join(path, "CACD2000", img))
        img_data.label.classification = Classification(category, attribute)
        segment.append(img_data)
    return dataset


def _get_attribute_map(path: str) -> Dict[str, Tuple[str, Dict[str, Any]]]:
    """Get data from .mat file.

    Arguments:
        path: The root directory of the dataset.

    Returns:
        A Dict of attributes.
    """
    identity_set = set()
    mat = File(path)
    data = mat["celebrityImageData"]
    name_map = _identity_to_name(mat)  # Creat a hashmap between celebrities' name and identity.
    attributes = {}
    keys = list(data.keys())
    keys.remove("feature")
    datas = {}
    for key in keys:  # Save the needed data into the datas.
        datas[key] = np.array(data[key][0])
    keys.remove("name")
    for index, value in enumerate(datas["name"]):  # The "name" is not the name of the star
        # but the name of the img file.
        attribute = {}
        for key in keys:
            if key == "identity":
                attribute["name"] = name_map[datas[key][index]]
                name = str(int(datas["identity"][index])).zfill(
                    4
                )  # Turn identity to four digits. (1 to 0001)
                identity_set.add(name)
            else:
                attribute[key] = datas[key][index]
        # attributes -- img_file_name: tuple of information.
        attributes["".join(chr(int(i)) for i in mat[value])] = (name, attribute)

    return attributes


def _identity_to_name(mat: File) -> Dict[str, str]:
    """Creat a hashmap between celebrities' name and identity.

    Arguments:
        mat: .mat file.

    Returns:
        A hashmap between celebrities' name and identity.
    """
    data = mat["celebrityData"]
    name_map = {}
    for index, name in enumerate(
        data["name"][0]
    ):  # Name is a h5r object which can be searched in .mat file.
        obj = mat[name]  # Search real name of celebrity in .mat file.
        name_map[data["identity"][0][index]] = "".join(chr(int(i)) for i in obj)
    return name_map

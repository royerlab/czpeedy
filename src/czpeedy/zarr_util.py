import functools
import json
import logging
import operator
from logging import Logger
from pathlib import Path
from typing import Literal, Optional

LOG = logging.getLogger(__name__)
METADATA_FILES_BY_VERSION = {
    2: [".zarray", ".zattrs", ".zgroup"],
    3: ["zarr.json"],
}
ALL_METADATA_FILES = set(functools.reduce(operator.iadd, METADATA_FILES_BY_VERSION.values(), []))
KNOWN_VERSIONS = set(METADATA_FILES_BY_VERSION.keys())


def identify_zarr_format(archive_path: Path, log: Logger = LOG) -> Optional[Literal[2, 3]]:
    """
    Identify the zarr version of the archive by identifying a metadata file and reading its zarr_format key.
    If the metadata file is missing, the zarr_format key is missing, or the specified version is not "2" or "3",
    returns None.
    """

    for candidate_file in ALL_METADATA_FILES:
        metadata_file = archive_path / candidate_file

        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
            zarr_format = metadata.get("zarr_format")
            if zarr_format in KNOWN_VERSIONS:
                log.debug(f"Identified zarr version {zarr_format} from metadata file {metadata_file}")
                return zarr_format
            else:
                log.debug(f"Invalid zarr version {zarr_format} in metadata file {metadata_file}")
                return None

    log.debug(f"Could not identify zarr version from metadata files in archive folder {archive_path}")
    return None

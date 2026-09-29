"""Constants and data classes for the NASA OCO2 adapter."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

COLLECTION_SHORT_NAME: str = "OCO2_GEOS_L3CO2_DAY"
VERSION: str = "10r"
PAGE_SIZE: int = 200

CMR_GRANULES_URL: str = "https://cmr.earthdata.nasa.gov/search/granules.json"

# OCO-2 GEOS L3 is one file per day, global grid (no bbox search needed).
FILL_VALUE: float = -9999.0

# Candidate HDF5 paths for the XCO2 variable (tried in order). Different file
# versions use slightly different group names.
XCO2_PATHS = [
    "HDFEOS/GRIDS/OCO-2 Level 3 Gridded XCO2/Data Fields/XCO2",
    "HDFEOS/GRIDS/OCO-2 Level 3 Daily, 0.5x0.625 deg Grid/Data Fields/XCO2",
    "XCO2",
]
XCO2PREC_SUFFIXES = ["/XCO2PREC", "/XAPRIORI", "/XCO2_RETRIEVAL_ERROR"]

LICENSE_INFO: MappingProxyType[str, str] = MappingProxyType(
    {
        "license": "NASA data policy - free and open (attribution requested)",
        "license_url": (
            "https://www.earthdata.nasa.gov/engage/open-data-services-software/"
            "data-rights-related-issues"
        ),
        "citation": (
            "Brad Weir, Lesley Ott and OCO-2 Science Team (2022), OCO-2 GEOS Level 3 daily, "
            "0.5x0.625 assimilated CO2 V10r, Greenbelt, MD, USA, Goddard Earth Sciences Data "
            "and Information Services Center (GES DISC), doi: 10.5067/Y9M4NM9MPCGH"
        ),
    }
)

VARIABLE_INFO: MappingProxyType[str, dict[str, Any]] = MappingProxyType(
    {
        "xco2": {
            "description": "Column-averaged dry-air mole fraction of CO2",
            "units": "ppm",
        },
        "xco2_uncertainty": {
            "description": "1-sigma retrieval uncertainty",
            "units": "ppm",
        },
    }
)


@dataclass(frozen=True)
class Granule:
    id: str
    date: str
    url: str

"""Constants and data classes for the NASA EMIT adapter."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

COLLECTION_SHORT_NAME: str = "EMITL2BMIN"
COLLECTION_CONCEPT_ID: str = "C2408034484-LPCLOUD"
VERSION: str = "001"
PAGE_SIZE: int = 200

CMR_GANULES_URL: str = "https://cmr.earthdata.nasa.gov/search/granules.json"

QUERY_ARGS = "/location/lat,/location/lon,/mineral_metadata/mineral_name"
LON_PATHS = ["/location/lon", "location/lon", "lon"]
LAT_PATHS = ["/location/lat", "location/lat", "lat"]
MINERAL_NAME_PATHS = [
    "/mineral_metadata/mineral_name",
    "mineral_metadata/mineral_name",
    "mineral_name",
]
ABUNDANCE_PATHS = ["/spectral_abundance", "spectral_abundance"]

LICENSE_INFO: MappingProxyType[str, str] = MappingProxyType(
    {
        "license": "NASA data policy — free and open (attribution requested)",
        "license_url": (
            "https://www.earthdata.nasa.gov/engage/open-data-services-software/"
            "data-rights-related-issues"
        ),
        "citation": (
            "Green, R. (2023). <i>EMIT L2B Estimated Mineral Identification and Band Depth and "
            "Uncertainty 60 m V001</i> [Dataset]. NASA Land Processes Distributed Active Archive "
            "Center. https://doi.org/10.5067/EMIT/EMITL2BMIN.001"
        ),
    }
)

VARIABLE_INFO: MappingProxyType[str, dict[str, Any]] = MappingProxyType({"fill in vars": {}})


@dataclass(frozen=True)
class Granule:
    id: str
    date: str
    nc4_link: str

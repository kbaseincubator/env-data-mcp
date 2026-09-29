"""Constants and data classes for the NASA OCO2 adapter."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

LICENSE_INFO: MappingProxyType[str, str] = MappingProxyType({
    "license": "NASA data policy - free and open (attribution requested)",
    "license_url": (
        "https://www.earthdata.nasa.gov/engage/open-data-services-software/"
            "data-rights-related-issues"
    ),
    "citation": (
        "Brad Weir, Lesley Ott and OCO-2 Science Team (2022), OCO-2 GEOS Level 3 daily, "
        "0.5x0.625 assimilated CO2 V10r, Greenbelt, MD, USA, Goddard Earth Sciences Data "
        "and Information Services Center (GES DISC), doi: 10.5067/Y9M4NM9MPCGH"
    )
})

VARIABLE_INFO: MappingProxyType[str, Any] = MappingProxyType({
    
})
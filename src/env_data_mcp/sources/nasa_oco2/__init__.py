"""NASA OCO-2 - Column CO2 (XCO2), 2015-01-01 to 2022-02-28.

Data source: ``https://cmr.earthdata.nasa.gov`` (GES DISC)
Coverage: Global, 0.5 x 0.625 degree daily grid, 2015-01-01 to 2022-02-28.
Auth required: Yes - a free EARTHDATA token is required. See
``https://urs.earthdata.nasa.gov/documentation/for_users/user_token``
License: NASA data policy - free and open.
``https://www.earthdata.nasa.gov/engage/open-data-services-software/``
Citation: Weir, B., Ott, L., et al. (2021). OCO-2 GEOS Level 3 daily,
          0.5x0.625 deg assimilation, V10r. Goddard Earth Sciences Data and
          Information Services Center (GES DISC). https://doi.org/10.5067/Y9M4NM9MPCGH
"""

from .tools import nasa_oco2_bbox_query, nasa_oco2_point_query

__all__ = [
    "nasa_oco2_bbox_query",
    "nasa_oco2_point_query",
]

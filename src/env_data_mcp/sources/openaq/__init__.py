"""OpenAQ data adapter.

Data source: ``https://api.openaq.org/v3/``
Coverage: Global, 2016-present (sensor network)
Auth required: Yes - free API key from https://explore.openaq.org/register
               Set ``OPENAQ_API_KEY`` environment variable.
License: CC BY 4.0

Note on authentication
----------------------
OpenAQ v2 (unauthenticated) was retired in 2024.  v3 requires a free API key
obtained at https://explore.openaq.org/register.
"""

from .tools import (
    openaq_available_variables,
    openaq_bbox_query,
    openaq_point_query,
)

__all__ = [
    "openaq_available_variables",
    "openaq_bbox_query",
    "openaq_point_query",
]

"""MCP tool functions for the OpenAQ adapter."""

from __future__ import annotations

import os
import time
from typing import Any

from pydantic import ValidationError

from env_data_mcp.helpers import (
    bbox_area_deg2,
    build_meta,
    check_runtime,
    date_range_days,
    point_to_bbox,
)
from env_data_mcp.models import (
    AvailableVariablesResponse,
    BboxInput,
    DateRange,
    GroupedGeometryResponse,
    PointInput,
)
from env_data_mcp.server import mcp

from ._constants import DEFAULT_VARIABLES, LICENSE_INFO
from ._query import get_variable_info, query_bbox, query_point


def _validate_available_variables_response(response: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize available variables tool responses."""
    return AvailableVariablesResponse.model_validate(response).model_dump(by_alias=True)


def _validate_grouped_geometry_results(response: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize grouped geometry responses."""
    return GroupedGeometryResponse.model_validate(response).model_dump(by_alias=True)


def _get_api_key(
    query_params: dict[str, Any],
    variables: list[str],
) -> tuple[str, dict[str, Any] | None]:
    """Get the OpenAQ api key from the environment.

    If the environment variable doesn't exist, an error response is returned
    as the second argument.
    """
    api_key = os.environ.get("OPENAQ_API_KEY", "")
    if not api_key:
        return "", {
            "data": [],
            "_meta": build_meta(
                source="openaq",
                query_params=query_params,
                geometries_returned=0,
                total_records_returned=0,
                latency_s=0,
                license_info=LICENSE_INFO,
                auth_required=True,
                auth_present=False,
                success=False,
                error=(
                    "OPENAQ_API_KEY environment variable is not set. "
                    "Register for a free API key at https://explore.openaq.org/register "
                    "and set it in your environment or .env file."
                ),
                variables=variables,
                variable_info={},
            ),
        }
    return api_key, None


@mcp.tool()
def openaq_available_variables() -> dict[str, Any]:
    """Return a list of available OpenAQ variables with descriptions and units."""
    t0 = time.perf_counter()
    api_key, error = _get_api_key({}, [])
    if error:
        return error
    try:
        variable_info = get_variable_info(api_key)
    except Exception as exc:
        return _validate_available_variables_response(
            {
                "data": {},
                "_meta": build_meta(
                    source="openaq",
                    query_params={},
                    geometries_returned=0,
                    total_records_returned=0,
                    latency_s=0,
                    license_info=LICENSE_INFO,
                    auth_required=True,
                    auth_present=True,
                    success=False,
                    error=str(exc),
                ),
            }
        )
    latency = time.perf_counter() - t0
    return _validate_available_variables_response(
        {
            "data": variable_info,
            "_meta": build_meta(
                source="openaq",
                query_params={},
                geometries_returned=0,
                total_records_returned=len(variable_info),
                latency_s=latency,
                license_info=LICENSE_INFO,
                auth_required=True,
                auth_present=True,
            ),
        }
    )


@mcp.tool()
def openaq_point_query(
    *,
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    radius_km: float = 5.0,
    variables: frozenset[str] | list[str] = DEFAULT_VARIABLES,
    max_runtime_s: float = 30.0,
) -> dict[str, Any]:
    """Query OpenAQ air quality data for a point location.

    Returns air quality data collected within the provided radius around
    a point location.

    ### Args
    * __latitude__: Decimal degrees, WGS84 (-90 to 90).
    * __longitude__: Decimal degrees, WGS84 (-180 to 180).
    * __start_date__: Inclusive start date, ISO 8601 date string, e.g., "2019-08-15".
    * __end_date__: Inclusive end date, ISO 8601 date string, e.g., "2019-08-16".
    * __radius_km__: Search radius in kilometers. Defaults to 5 km. Maximum 25.0 km.
    * __variables__: OpenAQ variable names. Use the ``openaq_available_variables()``
          tool to get a list of valid variable names. Defaults to a standard set of
          commonly used variables.
    * __max_runtime_s__: Optional maximum runtime in seconds. If the query is estimated to
          exceed this, a warning is returned instead of data. If not provided, it is
          assumed to be 30 s.
    """
    variables = list(variables)
    query_params: dict[str, Any] = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "variables": variables,
        "radius_km": radius_km,
        "max_runtime_s": max_runtime_s,
    }
    t0 = time.perf_counter()
    api_key, error = _get_api_key(query_params, variables)
    if error:
        return error
    if radius_km > 25.0:
        return _validate_grouped_geometry_results(
            {
                "data": [],
                "_meta": build_meta(
                    source="openaq",
                    query_params=query_params,
                    geometries_returned=0,
                    total_records_returned=0,
                    latency_s=0,
                    license_info=LICENSE_INFO,
                    auth_required=True,
                    auth_present=True,
                    success=False,
                    error="Radius exceeds maximum of 25.0 km.",
                    variables=variables,
                ),
            }
        )
    var_info: dict[str, dict[str, str]] = {}
    try:
        point = PointInput(latitude=latitude, longitude=longitude)
        date_range = DateRange(start_date=start_date, end_date=end_date)

        full_var_info = get_variable_info(api_key)
        var_info = {k: full_var_info[k] for k in variables if k in full_var_info}

        n_days = date_range_days(start_date, end_date)
        area_deg2 = bbox_area_deg2(point_to_bbox(latitude, longitude, radius_km))
        if warn := check_runtime(
            source="openaq",
            n_days=n_days,
            area_deg2=area_deg2,
            max_runtime_s=max_runtime_s,
            scale_factor=len(variables),
        ):
            return _validate_grouped_geometry_results(warn)
        data, unavailable = query_point(
            point.latitude,
            point.longitude,
            date_range.start_date,
            date_range.end_date,
            radius_km,
            variables,
            api_key,
        )
        latency = time.perf_counter() - t0
        return _validate_grouped_geometry_results(
            {
                "data": data,
                "_meta": build_meta(
                    source="openaq",
                    query_params=query_params,
                    geometries_returned=len(data),
                    total_records_returned=sum(len(r["records"]) for r in data),
                    latency_s=latency,
                    license_info=LICENSE_INFO,
                    auth_required=True,
                    auth_present=True,
                    variables=variables,
                    variable_info=var_info,
                    unavailable_variables=unavailable if unavailable else None,
                ),
            }
        )
    except (ValidationError, ValueError, Exception) as exc:
        latency = time.perf_counter() - t0
        return _validate_grouped_geometry_results(
            {
                "data": [],
                "_meta": build_meta(
                    source="openaq",
                    query_params=query_params,
                    geometries_returned=0,
                    total_records_returned=0,
                    latency_s=latency,
                    license_info=LICENSE_INFO,
                    auth_required=True,
                    auth_present=True,
                    success=False,
                    error=str(exc),
                    variables=variables,
                    variable_info=var_info,
                ),
            }
        )


@mcp.tool()
def openaq_bbox_query(
    *,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    start_date: str,
    end_date: str,
    variables: frozenset[str] | list[str] = DEFAULT_VARIABLES,
    max_runtime_s: float = 30.0,
) -> dict[str, Any]:
    """Query OpenAQ air quality data for a bounding-box area.

    Returns air quality data collected within the bounding-box.

    ### Args
    * __min_lat__: South boundard, decimal degrees, WGS84 (-90 to 90).
    * __max_lat__: North boundard, decimal degrees, WGS84 (-90 to 90).
    * __min_lon__: West boundary, decimal degrees, WGS84 (-180 to 180).
    * __max_lon__: East boundary, decimal degrees, WGS84 (-180 to 180).
    * __start_date__: Inclusive start date, ISO 8601 date string, e.g., "2019-08-15".
    * __end_date__: Inclusive end date, ISO 8601 date string, e.g., "2019-08-16".
    * __variables__: OpenAQ variable names. Use the ``openaq_available_variables()``
          tool to get a list of valid variable names. Defaults to a standard set of
          commonly used variables.
    * __max_runtime_s__: Optional maximum runtime in seconds. If the query is estimated to
          exceed this, a warning is returned instead of data. If not provided, it is
          assumed to be 30 s.
    """
    variables = list(variables)
    query_params: dict[str, Any] = {
        "min_lat": min_lat,
        "max_lat": max_lat,
        "min_lon": min_lon,
        "max_lon": max_lon,
        "start_date": start_date,
        "end_date": end_date,
        "variables": variables,
        "max_runtime_s": max_runtime_s,
    }
    t0 = time.perf_counter()
    api_key, error = _get_api_key(query_params, variables)
    if error:
        return error
    var_info: dict[str, dict[str, str]] = {}
    try:
        bbox = BboxInput(min_lat=min_lat, max_lat=max_lat, min_lon=min_lon, max_lon=max_lon)
        date_range = DateRange(start_date=start_date, end_date=end_date)

        full_var_info = get_variable_info(api_key)
        var_info = {k: full_var_info[k] for k in variables if k in full_var_info}

        n_days = date_range_days(start_date, end_date)
        area_deg2 = bbox_area_deg2(bbox.model_dump())
        if warn := check_runtime(
            source="openaq",
            n_days=n_days,
            area_deg2=area_deg2,
            max_runtime_s=max_runtime_s,
            scale_factor=len(variables),
        ):
            return _validate_grouped_geometry_results(warn)
        data, unavailable = query_bbox(
            bbox.min_lat,
            bbox.max_lat,
            bbox.min_lon,
            bbox.max_lon,
            date_range.start_date,
            date_range.end_date,
            variables,
            api_key,
        )
        latency = time.perf_counter() - t0
        return _validate_grouped_geometry_results(
            {
                "data": data,
                "_meta": build_meta(
                    source="openaq",
                    query_params=query_params,
                    geometries_returned=len(data),
                    total_records_returned=sum(len(r["records"]) for r in data),
                    latency_s=latency,
                    license_info=LICENSE_INFO,
                    auth_required=True,
                    auth_present=True,
                    variables=variables,
                    variable_info=var_info,
                    unavailable_variables=unavailable if unavailable else None,
                ),
            }
        )
    except (ValidationError, ValueError, Exception) as exc:
        latency = time.perf_counter() - t0
        return _validate_grouped_geometry_results(
            {
                "data": [],
                "_meta": build_meta(
                    source="openaq",
                    query_params=query_params,
                    geometries_returned=0,
                    total_records_returned=0,
                    latency_s=latency,
                    license_info=LICENSE_INFO,
                    auth_required=True,
                    auth_present=True,
                    success=False,
                    error=str(exc),
                    variables=variables,
                    variable_info=var_info,
                ),
            }
        )

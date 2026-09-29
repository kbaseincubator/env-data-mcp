"""Integration tests for the NASA OCO2 source adapter (live CMR + GES DISC).

Marked ``@pytest.mark.integration`` - not run in CI unit-test jobs.
These tests call the real NASA CMR granule search API and GES DISC downloads.

Requires the ``EARTHDATA_TOKEN`` environment variable (free registration at
https://urs.earthdata.nasa.gov/). When the token is absent, all tests are
skipped gracefully.

OCO-2 GEOS L3 coverage is 2015-01-01 to 2022-02-28 (global, assimilated
model output - unlike EMIT it has continuous global coverage, land and
ocean, for any day in that window).
"""

from __future__ import annotations

import os
from dataclasses import replace
from http import HTTPStatus
from typing import Any

import httpx
import pytest

from env_data_mcp.sources.nasa_oco2._constants import (
    CMR_GRANULES_URL,
    COLLECTION_SHORT_NAME,
    VERSION,
)
from env_data_mcp.sources.nasa_oco2.tools import (
    nasa_oco2_bbox_query,
    nasa_oco2_point_query,
)

from .common import (
    STANDARD_BBOXES,
    STANDARD_LOCATIONS,
    AdapterSpec,
    assert_grouped_geometry_response_valid,
)

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Adapter-specific test window. The shared STANDARD_LOCATIONS/STANDARD_BBOXES
# dates (2022-06-01 onward) postdate OCO-2 GEOS L3 coverage (ends 2022-02-28),
# so a verified-good window within coverage is used instead. Bbox sizes are
# unchanged from the shared defaults - OCO-2 is a coarse 0.5x0.625-degree
# grid, so a 4-degree bbox is only ~60 cells (fast; verified live).
# ---------------------------------------------------------------------------

_OCO2_WINDOW_START = "2019-08-01"
_OCO2_WINDOW_END = "2019-08-07"

_NASA_OCO2_LOCATIONS = {
    loc.label: replace(loc, start_date=_OCO2_WINDOW_START, end_date=_OCO2_WINDOW_END)
    for loc in STANDARD_LOCATIONS
}
_NASA_OCO2_BBOXES = {
    bbox.label: replace(bbox, start_date=_OCO2_WINDOW_START, end_date=_OCO2_WINDOW_END)
    for bbox in STANDARD_BBOXES
}


# ---------------------------------------------------------------------------
# Availability guard
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _require_nasa_oco2_available():
    """Skip all tests if EARTHDATA_TOKEN is absent or CMR is unreachable.

    CMR granule *search* does not require a valid token (only the follow-up
    file download does), so token validity is checked per-test via
    ``_skip_if_auth_rejected`` instead of here.
    """
    if not os.environ.get("EARTHDATA_TOKEN", ""):
        pytest.skip("EARTHDATA_TOKEN not set. Skipping NASA OCO2 integration tests")
    try:
        r = httpx.get(
            CMR_GRANULES_URL,
            params={"short_name": COLLECTION_SHORT_NAME, "version": VERSION, "page_size": 1},
            timeout=10,
        )
        if r.status_code >= HTTPStatus.INTERNAL_SERVER_ERROR:
            pytest.skip(f"NASA CMR returned HTTP {r.status_code}")
    except Exception as exc:
        pytest.skip(f"NASA CMR not reachable: {exc}")


def _skip_if_auth_rejected(meta: dict[str, Any]) -> None:
    """Skip a test if EARTHDATA_TOKEN is missing or was rejected by NASA.

    ``auth_present`` is True whenever a token was found in the environment,
    even if that token turns out to be expired/invalid - the actual rejection
    only surfaces later as an error message, so it must be checked here too.
    """
    if not meta["auth_present"]:
        pytest.skip("EARTHDATA_TOKEN not set")
    error = meta.get("error") or ""
    if not meta["success"] and "token" in error.lower():
        pytest.skip(f"EarthData token rejected or expired: {error}")


# ---------------------------------------------------------------------------
# Adapter-specific validate hooks - called by test_common_live.py after
# common assertions, and directly by adapter-specific tests below.
# ---------------------------------------------------------------------------


def _assert_valid_xco2_record(rec: dict[str, Any]) -> None:
    assert rec["date"] != ""
    assert rec["granule_id"] != ""
    assert 300.0 <= rec["xco2"] <= 450.0, f"xco2 {rec['xco2']} outside plausible range"
    assert rec["xco2_units"] == "ppm"
    if "xco2_uncertainty" in rec:
        assert rec["xco2_uncertainty"] >= 0.0
        assert rec["xco2_uncertainty_units"] == "ppm"


def _validate_nasa_oco2_point_result(result: dict) -> None:
    """NASA OCO2-specific assertions for a point query result."""
    assert_grouped_geometry_response_valid(result)
    assert result["_meta"]["source"] == "nasa_oco2"
    assert result["_meta"]["auth_required"] is True
    assert result["_meta"]["auth_present"] is True
    for group in result["data"]:
        assert len(group["records"]) >= 1
        for rec in group["records"]:
            _assert_valid_xco2_record(rec)


def _validate_nasa_oco2_bbox_result(result: dict) -> None:
    """NASA OCO2-specific assertions for a bbox query result."""
    assert_grouped_geometry_response_valid(result)
    assert result["_meta"]["source"] == "nasa_oco2"
    assert result["_meta"]["auth_required"] is True
    assert result["_meta"]["auth_present"] is True
    for group in result["data"]:
        assert len(group["records"]) >= 1


# ---------------------------------------------------------------------------
# NASA OCO2 AdapterSpec - exported for test_common_live.py
# ---------------------------------------------------------------------------

NASA_OCO2_SPEC = AdapterSpec(
    name="nasa_oco2",
    available_variables=None,
    point_query=nasa_oco2_point_query,
    bbox_query=nasa_oco2_bbox_query,
    supports_date_range=True,
    primary_variable=None,
    default_variables=None,
    max_runtime_s=60.0,
    custom_locations=_NASA_OCO2_LOCATIONS,
    custom_bboxes=_NASA_OCO2_BBOXES,
    validate_point_result=_validate_nasa_oco2_point_result,
    validate_bbox_result=_validate_nasa_oco2_bbox_result,
)


# ---------------------------------------------------------------------------
# Test coordinates - Yakima River Valley, WA; verified real XCO2 coverage.
# ---------------------------------------------------------------------------

_LAT = 46.2531882
_LON = -119.4768203
_START = "2019-08-01"
_END = "2019-08-31"


# ---------------------------------------------------------------------------
# Point Queries
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def point_query_results() -> dict[str, Any]:
    return nasa_oco2_point_query(
        latitude=_LAT,
        longitude=_LON,
        start_date=_START,
        end_date=_END,
        max_runtime_s=9999,
    )


@pytest.mark.integration
def test_nasa_oco2_point_query_live_success(point_query_results):
    """Success is defined as no exception and _meta.success = True.

    Data may be empty for sparse-coverage regions.
    """
    meta = point_query_results["_meta"]
    _skip_if_auth_rejected(meta)
    assert meta["success"] is True, f"Query failed: {meta.get('error')}"
    assert meta["error"] is None
    assert meta["source"] == "nasa_oco2"


@pytest.mark.integration
def test_nasa_oco2_point_query_live_meta_fields(point_query_results):
    meta = point_query_results["_meta"]
    _skip_if_auth_rejected(meta)
    assert meta["auth_required"] is True
    assert meta["auth_present"] is True
    assert meta["latency_s"] > 0
    assert meta["license"] != ""
    assert meta["license_url"] != ""
    assert "latitude" in meta["query_params"]


@pytest.mark.integration
def test_nasa_oco2_point_query_live_record_schema(point_query_results):
    meta = point_query_results["_meta"]
    _skip_if_auth_rejected(meta)
    if not point_query_results["data"]:
        pytest.skip("No OCO-2 data for this location+period - sparse coverage expected")
    entry = point_query_results["data"][0]
    assert entry["geometry"]["type"] == "Point"
    assert entry["longitude"] == entry["geometry"]["coordinates"][0]
    assert entry["latitude"] == entry["geometry"]["coordinates"][1]
    assert _LAT - 1.0 < entry["latitude"] < _LAT + 1.0
    assert _LON - 1.0 < entry["longitude"] < _LON + 1.0
    assert len(entry["records"]) > 0
    for rec in entry["records"]:
        _assert_valid_xco2_record(rec)


@pytest.mark.integration
def test_nasa_oco2_point_query_live_no_key_returns_auth_error(monkeypatch):
    monkeypatch.delenv("EARTHDATA_TOKEN", raising=False)
    result = nasa_oco2_point_query(
        latitude=_LAT,
        longitude=_LON,
        start_date=_START,
        end_date=_END,
    )
    assert result["_meta"]["success"] is False
    meta = result["_meta"]
    assert meta["auth_required"] is True
    assert meta["auth_present"] is False


# ---------------------------------------------------------------------------
# Bounding-Box Queries
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bbox_query_results() -> dict[str, Any]:
    return nasa_oco2_bbox_query(
        min_lat=_LAT - 2.0,
        max_lat=_LAT + 2.0,
        min_lon=_LON - 2.0,
        max_lon=_LON + 2.0,
        start_date=_START,
        end_date=_END,
        max_runtime_s=9999,
    )


@pytest.mark.integration
def test_nasa_oco2_bbox_query_live_success(bbox_query_results):
    """Success is defined as no exception and _meta.success = True.

    Data may be empty for sparse-coverage regions.
    """
    meta = bbox_query_results["_meta"]
    _skip_if_auth_rejected(meta)
    assert meta["success"] is True, f"Query failed: {meta.get('error')}"
    assert meta["error"] is None
    assert meta["source"] == "nasa_oco2"


@pytest.mark.integration
def test_nasa_oco2_bbox_query_live_meta_fields(bbox_query_results):
    meta = bbox_query_results["_meta"]
    _skip_if_auth_rejected(meta)
    assert meta["auth_required"] is True
    assert meta["auth_present"] is True
    assert meta["latency_s"] > 0
    assert meta["license"] != ""
    assert meta["license_url"] != ""


@pytest.mark.integration
def test_nasa_oco2_bbox_query_live_record_schema(bbox_query_results):
    meta = bbox_query_results["_meta"]
    _skip_if_auth_rejected(meta)
    if not bbox_query_results["data"]:
        pytest.skip("No OCO-2 data for this bbox+period - sparse coverage expected")
    entry = bbox_query_results["data"][0]
    assert entry["geometry"]["type"] == "Point"
    assert entry["longitude"] == entry["geometry"]["coordinates"][0]
    assert entry["latitude"] == entry["geometry"]["coordinates"][1]
    assert len(entry["records"]) > 0
    for rec in entry["records"]:
        _assert_valid_xco2_record(rec)


@pytest.mark.integration
def test_nasa_oco2_bbox_query_live_no_key_returns_auth_error(monkeypatch):
    monkeypatch.delenv("EARTHDATA_TOKEN", raising=False)
    result = nasa_oco2_bbox_query(
        min_lat=_LAT - 2.0,
        max_lat=_LAT + 2.0,
        min_lon=_LON - 2.0,
        max_lon=_LON + 2.0,
        start_date=_START,
        end_date=_END,
    )
    assert result["_meta"]["success"] is False
    meta = result["_meta"]
    assert meta["auth_required"] is True
    assert meta["auth_present"] is False

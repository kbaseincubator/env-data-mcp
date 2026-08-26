"""Integration tests for the OpenAQ source adapter (live API).

Marked ``@pytest.mark.integration`` — not run in CI unit-test jobs.
These tests call the real OpenAQ v3 REST API.

Requires ``OPENAQ_API_KEY`` environment variable (free registration at
https://explore.openaq.org/register).  When the key is absent, all tests
are skipped gracefully.
"""

from __future__ import annotations

import os
from http import HTTPStatus
from typing import Any

import httpx
import pytest

from env_data_mcp.sources.openaq import (
    openaq_available_variables,
    openaq_bbox_query,
    openaq_point_query,
)
from env_data_mcp.sources.openaq._constants import DEFAULT_VARIABLES

from .common import (
    AdapterSpec,
    DataExpectation,
    assert_grouped_geometry_response_valid,
)

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Availability guard
# ---------------------------------------------------------------------------

_OPENAQ_HEALTH = "https://api.openaq.org/v3/locations"


@pytest.fixture(scope="module", autouse=True)
def _require_openaq_available():
    """Skip all tests if OPENAQ_API_KEY is absent or the API is unreachable."""
    api_key = os.environ.get("OPENAQ_API_KEY", "")
    if not api_key:
        pytest.skip("OPENAQ_API_KEY not set. Skipping OpenAQ integration tests")
    try:
        r = httpx.get(
            _OPENAQ_HEALTH,
            params={"limit": 1},
            headers={"X-API-Key": api_key},
            timeout=10,
        )
        if r.status_code >= HTTPStatus.INTERNAL_SERVER_ERROR:
            pytest.skip(f"OpenAQ API returned HTTP {r.status_code}")
        if r.status_code == HTTPStatus.UNAUTHORIZED:
            pytest.skip("OPENAQ_API_KEY is invalid (HTTP 401)")
    except Exception as exc:
        pytest.skip(f"OpenAQ API not reachable: {exc}")


# ---------------------------------------------------------------------------
# Adapter-specific validate hooks - called by test_common_live.py after
# common assertions, and directly by adapter-specific tests below.
# ---------------------------------------------------------------------------


def _validate_openaq_point_result(result: dict) -> None:
    """OpenAQ-specific assertions for a point query result."""
    assert_grouped_geometry_response_valid(result)
    assert result["_meta"]["source"] == "openaq"
    assert result["_meta"]["auth_required"] is True
    assert result["_meta"]["auth_present"] is True
    for group in result["data"]:
        assert len(group["records"]) >= 1


def _validate_openaq_bbox_result(result: dict) -> None:
    """OpenAQ-specific assertions for a bbox query result."""
    assert_grouped_geometry_response_valid(result)
    assert result["_meta"]["source"] == "openaq"
    assert result["_meta"]["auth_required"] is True
    assert result["_meta"]["auth_present"] is True
    for group in result["data"]:
        assert len(group["records"]) >= 1


# ---------------------------------------------------------------------------
# OpenAQ AdapterSpec — exported for test_common_live.py
# ---------------------------------------------------------------------------

OPENAQ_SPEC = AdapterSpec(
    name="openaq",
    available_variables=openaq_available_variables,
    point_query=openaq_point_query,
    bbox_query=openaq_bbox_query,
    supports_date_range=True,
    primary_variable="pm25",
    default_variables=list(DEFAULT_VARIABLES),
    max_runtime_s=60.0,
    data_expectations={
        "nh_rural": DataExpectation(
            has_data=False,
            notes="No air quality measurements in rural area",
        ),
        "sh_rural": DataExpectation(
            has_data=False,
            notes="No air quality measurements in rural area",
        ),
        "nh_polar": DataExpectation(
            has_data=False,
            notes="No air quality measurements at the poles",
        ),
        "sh_polar": DataExpectation(
            has_data=False,
            notes="No air quality measurements at the poles",
        ),
        "ocean": DataExpectation(
            has_data=False,
            notes="No air quality measurements over the ocean",
        ),
    },
    extra_point_kwargs={"radius_km": 5.0},
    extra_bbox_kwargs={},
    use_small_bboxes=False,
    supports_bbox_bounds_test=True,
    supports_bbox_union_test=True,
    validate_point_result=_validate_openaq_point_result,
    validate_bbox_result=_validate_openaq_bbox_result,
)


# ---------------------------------------------------------------------------
# Available variables
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_openaq_available_variables():
    result = openaq_available_variables()
    assert len(result["data"]) > len(DEFAULT_VARIABLES)
    for var in DEFAULT_VARIABLES:
        assert var in result["data"]


# ---------------------------------------------------------------------------
# Test coordinates — Yakima River, Aug 2019
# ---------------------------------------------------------------------------

_LAT = 46.2531882
_LON = -119.4768203


# ---------------------------------------------------------------------------
# Point Queries
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def point_query_results() -> dict[str, Any]:
    return openaq_point_query(
        latitude=_LAT,
        longitude=_LON,
        radius_km=25.0,
        start_date="2019-08-01",
        end_date="2019-08-30",
    )


@pytest.mark.integration
def test_openaq_point_query_live_success(point_query_results):
    """Success is defined as no exception and _meta.success = True.

    Data may be empty for sparse-coverage regions.
    """
    assert point_query_results["_meta"]["success"] is True
    assert point_query_results["_meta"]["error"] is None
    assert point_query_results["_meta"]["source"] == "openaq"


@pytest.mark.integration
def test_openaq_point_query_live_meta_fields(point_query_results):
    meta = point_query_results["_meta"]
    assert meta["auth_required"] is True
    assert meta["auth_present"] is True
    assert meta["latency_s"] > 0
    assert meta["license"] != ""
    assert sorted(meta["variables"]) == sorted(DEFAULT_VARIABLES)
    for var in meta["unavailable_variables"]:
        assert var in DEFAULT_VARIABLES
    assert len(meta["variable_info"]) > 0
    for key, val in meta["variable_info"].items():
        assert key in DEFAULT_VARIABLES
        assert val["description"] != ""
        assert val["units"] != ""


@pytest.mark.integration
def test_openaq_point_query_live_record_schema(point_query_results):
    assert "data" in point_query_results
    assert len(point_query_results["data"]) > 0
    entry = point_query_results["data"][0]
    assert "geometry" in entry
    assert "latitude" in entry
    assert "longitude" in entry
    assert entry["geometry"]["type"] == "Point"
    assert entry["longitude"] == entry["geometry"]["coordinates"][0]
    assert entry["latitude"] == entry["geometry"]["coordinates"][1]
    assert _LAT - 1.0 < entry["latitude"] < _LAT + 1.0
    assert _LON - 1.0 < entry["longitude"] < _LON + 1.0
    assert "records" in entry
    assert len(entry["records"]) > 0
    for rec in entry["records"]:
        assert rec["datetime"] != ""
        matches = [var for var in DEFAULT_VARIABLES if var in rec]
        assert len(matches) == 1
        assert f"{matches[0]}_units" in rec
        assert rec[f"{matches[0]}_units"] != ""


@pytest.mark.integration
def test_openaq_point_query_live_no_key_returns_auth_error(monkeypatch):
    monkeypatch.delenv("OPENAQ_API_KEY", raising=False)
    result = openaq_point_query(
        latitude=_LAT,
        longitude=_LON,
        radius_km=50.0,
        start_date="2019-08-19",
        end_date="2019-08-19",
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
    return openaq_bbox_query(
        min_lat=_LAT,
        max_lat=_LAT + 1.0,
        min_lon=_LON,
        max_lon=_LON + 1.0,
        start_date="2019-08-01",
        end_date="2019-08-30",
    )


@pytest.mark.integration
def test_openaq_bbox_query_live_success(bbox_query_results):
    """Success is defined as no exception and _meta.success = True.

    Data may be empty for sparse-coverage regions.
    """
    assert bbox_query_results["_meta"]["success"] is True
    assert bbox_query_results["_meta"]["error"] is None
    assert bbox_query_results["_meta"]["source"] == "openaq"


@pytest.mark.integration
def test_openaq_bbox_query_live_meta_fields(bbox_query_results):
    meta = bbox_query_results["_meta"]
    assert meta["auth_required"] is True
    assert meta["auth_present"] is True
    assert meta["latency_s"] > 0
    assert meta["license"] != ""
    assert sorted(meta["variables"]) == sorted(DEFAULT_VARIABLES)
    for var in meta["unavailable_variables"]:
        assert var in DEFAULT_VARIABLES
    assert len(meta["variable_info"]) > 0
    for key, val in meta["variable_info"].items():
        assert key in DEFAULT_VARIABLES
        assert val["description"] != ""
        assert val["units"] != ""


@pytest.mark.integration
def test_openaq_bbox_query_live_record_schema(bbox_query_results):
    assert "data" in bbox_query_results
    assert len(bbox_query_results["data"]) > 0
    entry = bbox_query_results["data"][0]
    assert "geometry" in entry
    assert "latitude" in entry
    assert "longitude" in entry
    assert entry["geometry"]["type"] == "Point"
    assert entry["longitude"] == entry["geometry"]["coordinates"][0]
    assert entry["latitude"] == entry["geometry"]["coordinates"][1]
    assert _LAT - 1.0 < entry["latitude"] < _LAT + 1.0
    assert _LON - 1.0 < entry["longitude"] < _LON + 1.0
    assert "records" in entry
    assert len(entry["records"]) > 0
    for rec in entry["records"]:
        assert rec["datetime"] != ""
        matches = [var for var in DEFAULT_VARIABLES if var in rec]
        assert len(matches) == 1
        assert f"{matches[0]}_units" in rec
        assert rec[f"{matches[0]}_units"] != ""


@pytest.mark.integration
def test_openaq_bbox_query_live_no_key_returns_auth_error(monkeypatch):
    monkeypatch.delenv("OPENAQ_API_KEY", raising=False)
    result = openaq_bbox_query(
        min_lat=_LAT,
        max_lat=_LAT + 1.0,
        min_lon=_LON,
        max_lon=_LON + 1.0,
        start_date="2019-08-19",
        end_date="2019-08-19",
    )
    assert result["_meta"]["success"] is False
    meta = result["_meta"]
    assert meta["auth_required"] is True
    assert meta["auth_present"] is False

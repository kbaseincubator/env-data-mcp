"""Integration tests for the NASA EMIT source adapter (live CMR + OPeNDAP).

Marked ``@pytest.mark.integration`` — not run in CI unit-test jobs.
These tests call the real NASA CMR granule search API and LP DAAC OPeNDAP.

Requires the ``EARTHDATA_TOKEN`` environment variable (free registration at
https://urs.earthdata.nasa.gov/). When the token is absent, all tests are
skipped gracefully.

EMIT launched 2022-08-09 and flies on the ISS (~51.6-degree orbital
inclination), so it never observes latitudes beyond roughly +/-52 degrees and
has no data before its launch date.
"""

from __future__ import annotations

import os
from http import HTTPStatus
from typing import Any

import httpx
import pytest

from env_data_mcp.sources.nasa_emit._constants import (
    CMR_GANULES_URL,
    COLLECTION_SHORT_NAME,
    VERSION,
)
from env_data_mcp.sources.nasa_emit.tools import (
    nasa_emit_bbox_query,
    nasa_emit_point_query,
)

from .common import (
    AdapterSpec,
    DataExpectation,
    assert_grouped_geometry_response_valid,
)

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Availability guard
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _require_nasa_emit_available():
    """Skip all tests if EARTHDATA_TOKEN is absent or CMR is unreachable.

    CMR granule *search* does not require a valid token (only the follow-up
    OPeNDAP data download does), so token validity is checked per-test via
    ``_skip_if_auth_rejected`` instead of here.
    """
    if not os.environ.get("EARTHDATA_TOKEN", ""):
        pytest.skip("EARTHDATA_TOKEN not set. Skipping NASA EMIT integration tests")
    try:
        r = httpx.get(
            CMR_GANULES_URL,
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
    even if that token turns out to be expired/invalid — the actual rejection
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


def _assert_valid_mineral_record(rec: dict[str, Any]) -> None:
    """A record must have at least one real group_N mineral_name/band_depth pair."""
    assert rec["datetime"] != ""
    assert rec["granule_id"] != ""
    has_group = False
    for prefix in ("group_1", "group_2"):
        if f"{prefix}_mineral_name" in rec:
            has_group = True
            assert rec[f"{prefix}_mineral_name"] != ""
            assert rec[f"{prefix}_band_depth"] > 0.0
            assert rec[f"{prefix}_band_depth_units"] == "unitless"
    assert has_group, f"record has no group_1/group_2 mineral match: {rec}"


def _validate_nasa_emit_point_result(result: dict) -> None:
    """NASA EMIT-specific assertions for a point query result."""
    assert_grouped_geometry_response_valid(result)
    assert result["_meta"]["source"] == "nasa_emit"
    assert result["_meta"]["auth_required"] is True
    assert result["_meta"]["auth_present"] is True
    for group in result["data"]:
        assert len(group["records"]) >= 1
        for rec in group["records"]:
            _assert_valid_mineral_record(rec)


def _validate_nasa_emit_bbox_result(result: dict) -> None:
    """NASA EMIT-specific assertions for a bbox query result."""
    assert_grouped_geometry_response_valid(result)
    assert result["_meta"]["source"] == "nasa_emit"
    assert result["_meta"]["auth_required"] is True
    assert result["_meta"]["auth_present"] is True
    for group in result["data"]:
        assert len(group["records"]) >= 1


# ---------------------------------------------------------------------------
# NASA EMIT AdapterSpec — exported for test_common_live.py
# ---------------------------------------------------------------------------

NASA_EMIT_SPEC = AdapterSpec(
    name="nasa_emit",
    available_variables=None,
    point_query=nasa_emit_point_query,
    bbox_query=nasa_emit_bbox_query,
    supports_date_range=True,
    primary_variable=None,
    default_variables=None,
    max_runtime_s=120.0,
    # Standard 7-day window predates EMIT's 2022-08-09 launch; extend to 2025.
    longer_date_range=True,
    data_expectations={
        "nh_urban": DataExpectation(
            has_data=False,
            notes="Dense urban/vegetated area — no exposed mineral surfaces for EMIT to detect.",
        ),
        "sh_rural": DataExpectation(
            has_data=False,
            notes="Not a verified EMIT target region; no confirmed coverage.",
        ),
        "sh_urban": DataExpectation(
            has_data=False,
            notes="Dense urban/vegetated area — no exposed mineral surfaces for EMIT to detect.",
        ),
        "nh_polar": DataExpectation(
            has_data=False,
            notes="Outside EMIT's ISS orbital coverage (observes only ~+/-52 degrees latitude).",
        ),
        "sh_polar": DataExpectation(
            has_data=False,
            notes="Outside EMIT's ISS orbital coverage (observes only ~+/-52 degrees latitude).",
        ),
        "ocean": DataExpectation(
            has_data=False,
            notes="EMIT targets arid land dust-source regions, not open ocean.",
        ),
        "sh_midlat": DataExpectation(
            has_data=False,
            notes="Not a verified EMIT target region; no confirmed coverage.",
        ),
        "equatorial": DataExpectation(
            has_data=False,
            notes="EMIT targets arid land dust-source regions, not open ocean.",
        ),
    },
    supports_bbox_union_test=False,  # granule swath edges don't split cleanly at split_lon
    validate_point_result=_validate_nasa_emit_point_result,
    validate_bbox_result=_validate_nasa_emit_bbox_result,
)


# ---------------------------------------------------------------------------
# Test coordinates — Yakima River Valley, WA; arid/semi-arid land with
# detectable mineral signatures, well within EMIT's operational period.
# ---------------------------------------------------------------------------

_LAT = 46.2531882
_LON = -119.4768203
_START = "2022-09-01"
_END = "2023-08-31"


# ---------------------------------------------------------------------------
# Point Queries
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def point_query_results() -> dict[str, Any]:
    return nasa_emit_point_query(
        latitude=_LAT,
        longitude=_LON,
        start_date=_START,
        end_date=_END,
        max_runtime_s=9999,
    )


@pytest.mark.integration
def test_nasa_emit_point_query_live_success(point_query_results):
    """Success is defined as no exception and _meta.success = True.

    Data may be empty for sparse-coverage regions.
    """
    meta = point_query_results["_meta"]
    _skip_if_auth_rejected(meta)
    assert meta["success"] is True, f"Query failed: {meta.get('error')}"
    assert meta["error"] is None
    assert meta["source"] == "nasa_emit"


@pytest.mark.integration
def test_nasa_emit_point_query_live_meta_fields(point_query_results):
    meta = point_query_results["_meta"]
    _skip_if_auth_rejected(meta)
    assert meta["auth_required"] is True
    assert meta["auth_present"] is True
    assert meta["latency_s"] > 0
    assert meta["license"] != ""
    assert meta["license_url"] != ""
    assert "latitude" in meta["query_params"]


@pytest.mark.integration
def test_nasa_emit_point_query_live_record_schema(point_query_results):
    meta = point_query_results["_meta"]
    _skip_if_auth_rejected(meta)
    if not point_query_results["data"]:
        pytest.skip("No EMIT granules cover this location+period — sparse coverage expected")
    entry = point_query_results["data"][0]
    assert entry["geometry"]["type"] == "Point"
    assert entry["longitude"] == entry["geometry"]["coordinates"][0]
    assert entry["latitude"] == entry["geometry"]["coordinates"][1]
    assert _LAT - 1.0 < entry["latitude"] < _LAT + 1.0
    assert _LON - 1.0 < entry["longitude"] < _LON + 1.0
    assert len(entry["records"]) > 0
    for rec in entry["records"]:
        _assert_valid_mineral_record(rec)


@pytest.mark.integration
def test_nasa_emit_point_query_live_no_key_returns_auth_error(monkeypatch):
    monkeypatch.delenv("EARTHDATA_TOKEN", raising=False)
    result = nasa_emit_point_query(
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
    # A tiny box: EMIT is ~60 m/pixel, so even 0.004 degrees spans dozens of
    # pixels, and this adapter returns one geometry group per pixel.
    return nasa_emit_bbox_query(
        min_lat=_LAT - 0.002,
        max_lat=_LAT + 0.002,
        min_lon=_LON - 0.002,
        max_lon=_LON + 0.002,
        start_date=_START,
        end_date=_END,
        max_runtime_s=9999,
    )


@pytest.mark.integration
def test_nasa_emit_bbox_query_live_success(bbox_query_results):
    """Success is defined as no exception and _meta.success = True.

    Data may be empty for sparse-coverage regions.
    """
    meta = bbox_query_results["_meta"]
    _skip_if_auth_rejected(meta)
    assert meta["success"] is True, f"Query failed: {meta.get('error')}"
    assert meta["error"] is None
    assert meta["source"] == "nasa_emit"


@pytest.mark.integration
def test_nasa_emit_bbox_query_live_meta_fields(bbox_query_results):
    meta = bbox_query_results["_meta"]
    _skip_if_auth_rejected(meta)
    assert meta["auth_required"] is True
    assert meta["auth_present"] is True
    assert meta["latency_s"] > 0
    assert meta["license"] != ""
    assert meta["license_url"] != ""


@pytest.mark.integration
def test_nasa_emit_bbox_query_live_record_schema(bbox_query_results):
    meta = bbox_query_results["_meta"]
    _skip_if_auth_rejected(meta)
    if not bbox_query_results["data"]:
        pytest.skip("No EMIT granules cover this bbox+period — sparse coverage expected")
    entry = bbox_query_results["data"][0]
    assert entry["geometry"]["type"] == "Point"
    assert entry["longitude"] == entry["geometry"]["coordinates"][0]
    assert entry["latitude"] == entry["geometry"]["coordinates"][1]
    assert _LAT - 1.0 < entry["latitude"] < _LAT + 1.0
    assert _LON - 1.0 < entry["longitude"] < _LON + 1.0
    assert len(entry["records"]) > 0
    for rec in entry["records"]:
        _assert_valid_mineral_record(rec)


@pytest.mark.integration
def test_nasa_emit_bbox_query_live_no_key_returns_auth_error(monkeypatch):
    monkeypatch.delenv("EARTHDATA_TOKEN", raising=False)
    result = nasa_emit_bbox_query(
        min_lat=_LAT - 0.002,
        max_lat=_LAT + 0.002,
        min_lon=_LON - 0.002,
        max_lon=_LON + 0.002,
        start_date=_START,
        end_date=_END,
    )
    assert result["_meta"]["success"] is False
    meta = result["_meta"]
    assert meta["auth_required"] is True
    assert meta["auth_present"] is False

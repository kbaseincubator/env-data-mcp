"""Unit tests for env_data_mcp.sources.openaq.tools."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from env_data_mcp.sources.openaq._constants import OPENAQ_BASE_URL
from env_data_mcp.sources.openaq.tools import (
    _get_api_key,
    openaq_available_variables,
    openaq_bbox_query,
    openaq_point_query,
)

from .conftest import (
    _API_KEY,
    _LAT,
    _LON,
    _MEASUREMENT_RESPONSE_NO2,
    _MEASUREMENT_RESPONSE_PM25,
    _PARAMETERS_RESPONSE,
)

# ---------------------------------------------------------------------------
# _get_api_key
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_set_api_key")
def test_get_api_key_available():
    key, error = _get_api_key({"foo": 12}, ["foo", "bar"])
    assert key == _API_KEY
    assert error is None


@pytest.mark.usefixtures("_unset_api_key")
def test_get_api_key_unavailable():
    key, error = _get_api_key({"foo": 12}, ["foo", "bar"])
    assert key == ""
    assert error is not None
    assert error["data"] == []
    assert error["_meta"]["query_params"] == {"foo": 12}
    assert error["_meta"]["variables"] == ["foo", "bar"]
    assert error["_meta"]["auth_required"] is True
    assert error["_meta"]["auth_present"] is False
    assert error["_meta"]["success"] is False
    assert "OPENAQ_API_KEY" in error["_meta"]["error"]


# ---------------------------------------------------------------------------
# openaq_available_variables
# ---------------------------------------------------------------------------


def test_openaq_available_variables(httpx_mock, monkeypatch):
    monkeypatch.setattr("env_data_mcp.sources.openaq._query._AVAILABLE_PARAMETERS", None)
    httpx_mock.add_response(
        url=f"{OPENAQ_BASE_URL}/parameters",
        json=_PARAMETERS_RESPONSE,
    )
    result = openaq_available_variables()
    assert len(result["data"]) == 3
    assert result["data"] == {
        "pm25": {"description": "particulate matter <= 2.5 microns", "units": "ug/m3"},
        "no2": {"description": "nitrogen dioxide", "units": "ppb"},
        "o3": {"description": "ozone", "units": "ppb"},
    }
    assert result["_meta"]["total_records_returned"] == 3
    assert result["_meta"]["latency_s"] > 0.0
    assert result["_meta"]["latency_s"] < 1.0


def test_openaq_available_variables_http_error(httpx_mock, monkeypatch):
    monkeypatch.setattr("env_data_mcp.sources.openaq._query._AVAILABLE_PARAMETERS", None)
    httpx_mock.add_response(
        url=f"{OPENAQ_BASE_URL}/parameters",
        status_code=500,
    )
    with patch("env_data_mcp.sources.openaq._query._AVAILABLE_PARAMETERS", None):
        result = openaq_available_variables()
        assert len(result["data"]) == 0
        assert result["_meta"]["total_records_returned"] == 0
        assert result["_meta"]["success"] is False
        assert result["_meta"]["error"] is not None


# ---------------------------------------------------------------------------
# openaq_point_query
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_point_query_mock", "_set_api_key")
def test_openaq_point_query():
    result = openaq_point_query(
        latitude=_LAT,
        longitude=_LON,
        start_date="2019-08-19",
        end_date="2019-08-19",
        radius_km=5.0,
        variables=["o3", "pm25", "no2"],
    )
    assert len(result["data"]) == 1
    assert result["data"][0]["geometry"]["type"] == "Point"
    assert result["data"][0]["geometry"]["coordinates"] == [_LON, _LAT]
    assert result["data"][0]["latitude"] == _LAT
    assert result["data"][0]["longitude"] == _LON
    assert len(result["data"][0]["records"]) == 3
    for rec in result["data"][0]["records"]:
        assert "no2" in rec or "pm25" in rec
        if "no2" in rec:
            assert rec["datetime"] == _MEASUREMENT_RESPONSE_NO2["results"][0]["period"]
            assert rec["no2"] == 5.0
            assert rec["no2_units"] == "ppb"
        if "pm25" in rec:
            assert rec["datetime"] in [r["period"] for r in _MEASUREMENT_RESPONSE_PM25["results"]]
            assert rec["pm25"] in [12.5, 14.1]
            assert rec["pm25_units"] == "ug/m3"
    assert result["_meta"]["source"] == "openaq"
    assert result["_meta"]["geometries_returned"] == 1
    assert result["_meta"]["total_records_returned"] == 3
    assert result["_meta"]["auth_required"] is True
    assert result["_meta"]["auth_present"] is True
    assert sorted(result["_meta"]["variables"]) == sorted(["no2", "pm25", "o3"])
    assert result["_meta"]["unavailable_variables"] == ["o3"]


@pytest.mark.usefixtures("_point_query_mock", "_set_api_key")
def test_openaq_point_query_no_results():
    result = openaq_point_query(
        latitude=_LAT,
        longitude=_LON,
        start_date="2019-08-19",
        end_date="2019-08-19",
        radius_km=5.0,
        variables=["o3", "foo", "bar"],
    )
    assert len(result["data"]) == 0
    assert sorted(result["_meta"]["variables"]) == sorted(["o3", "foo", "bar"])
    assert sorted(result["_meta"]["unavailable_variables"]) == sorted(["o3", "foo", "bar"])


@pytest.mark.usefixtures("_unset_api_key")
def test_openaq_point_query_no_auth(httpx_mock, monkeypatch):
    result = openaq_point_query(
        latitude=_LAT,
        longitude=_LON,
        start_date="2019-08-19",
        end_date="2019-08-19",
        radius_km=5.0,
        variables=["o3", "no2", "pm25"],
    )
    assert len(result["data"]) == 0
    assert result["_meta"]["success"] is False
    assert "OPENAQ_API_KEY" in result["_meta"]["error"]


# ---------------------------------------------------------------------------
# openaq_bbox_query
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_bbox_query_mock", "_set_api_key")
def test_openaq_bbox_query():
    result = openaq_bbox_query(
        min_lat=_LAT,
        max_lat=_LAT + 1.0,
        min_lon=_LON,
        max_lon=_LON + 1.0,
        start_date="2019-08-19",
        end_date="2019-08-19",
        variables=["o3", "pm25", "no2"],
    )
    assert len(result["data"]) == 1
    assert result["data"][0]["geometry"]["type"] == "Point"
    assert result["data"][0]["geometry"]["coordinates"] == [_LON, _LAT]
    assert result["data"][0]["latitude"] == _LAT
    assert result["data"][0]["longitude"] == _LON
    assert len(result["data"][0]["records"]) == 3
    for rec in result["data"][0]["records"]:
        assert "no2" in rec or "pm25" in rec
        if "no2" in rec:
            assert rec["datetime"] == _MEASUREMENT_RESPONSE_NO2["results"][0]["period"]
            assert rec["no2"] == 5.0
            assert rec["no2_units"] == "ppb"
        if "pm25" in rec:
            assert rec["datetime"] in [r["period"] for r in _MEASUREMENT_RESPONSE_PM25["results"]]
            assert rec["pm25"] in [12.5, 14.1]
            assert rec["pm25_units"] == "ug/m3"
    assert result["_meta"]["source"] == "openaq"
    assert result["_meta"]["geometries_returned"] == 1
    assert result["_meta"]["total_records_returned"] == 3
    assert result["_meta"]["auth_required"] is True
    assert result["_meta"]["auth_present"] is True
    assert sorted(result["_meta"]["variables"]) == sorted(["no2", "pm25", "o3"])
    assert result["_meta"]["unavailable_variables"] == ["o3"]


@pytest.mark.usefixtures("_bbox_query_mock", "_set_api_key")
def test_openaq_bbox_query_no_results():
    result = openaq_bbox_query(
        min_lat=_LAT,
        max_lat=_LAT + 1.0,
        min_lon=_LON,
        max_lon=_LON + 1.0,
        start_date="2019-08-19",
        end_date="2019-08-19",
        variables=["o3", "foo", "bar"],
    )
    assert len(result["data"]) == 0
    assert sorted(result["_meta"]["variables"]) == sorted(["o3", "foo", "bar"])
    assert sorted(result["_meta"]["unavailable_variables"]) == sorted(["o3", "foo", "bar"])


@pytest.mark.usefixtures("_unset_api_key")
def test_openaq_bbox_query_no_auth(httpx_mock, monkeypatch):
    result = openaq_bbox_query(
        min_lat=_LAT,
        max_lat=_LAT + 1.0,
        min_lon=_LON,
        max_lon=_LON + 1.0,
        start_date="2019-08-19",
        end_date="2019-08-19",
        variables=["o3", "no2", "pm25"],
    )
    assert len(result["data"]) == 0
    assert result["_meta"]["success"] is False
    assert "OPENAQ_API_KEY" in result["_meta"]["error"]

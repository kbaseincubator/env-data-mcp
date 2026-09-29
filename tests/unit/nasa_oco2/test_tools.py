"""Unit tests for env_data_mcp.sources.nasa_oco2.tools."""

from __future__ import annotations

import re

import pytest

from env_data_mcp.sources.nasa_oco2._constants import CMR_GRANULES_URL
from env_data_mcp.sources.nasa_oco2.tools import (
    _get_token,
    nasa_oco2_bbox_query,
    nasa_oco2_point_query,
)

from .conftest import _EMPTY_CMR_RESPONSE, _GRANULE_ID_1, _LAT, _LON, _TOKEN

# ---------------------------------------------------------------------------
# _get_token
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_set_token")
def test_get_token_available():
    token, error = _get_token({"foo": 12})
    assert token == _TOKEN
    assert error is None


@pytest.mark.usefixtures("_unset_token")
def test_get_token_unavailable():
    token, error = _get_token({"foo": 12})
    assert token == ""
    assert error is not None
    assert error["data"] == []
    assert error["_meta"]["query_params"] == {"foo": 12}
    assert error["_meta"]["auth_required"] is True
    assert error["_meta"]["auth_present"] is False
    assert error["_meta"]["success"] is False
    assert "EARTHDATA_TOKEN" in error["_meta"]["error"]


# ---------------------------------------------------------------------------
# nasa_oco2_point_query
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_point_query_mock", "_set_token")
def test_nasa_oco2_point_query():
    result = nasa_oco2_point_query(
        latitude=_LAT,
        longitude=_LON,
        start_date="2019-08-01",
        end_date="2019-08-02",
    )
    assert len(result["data"]) == 1
    assert result["data"][0]["geometry"]["type"] == "Point"
    assert result["data"][0]["latitude"] == pytest.approx(_LAT, abs=1e-3)
    assert result["data"][0]["longitude"] == pytest.approx(_LON, abs=1e-3)
    assert len(result["data"][0]["records"]) == 2
    assert result["data"][0]["records"][0]["granule_id"] == _GRANULE_ID_1
    assert result["_meta"]["source"] == "nasa_oco2"
    assert result["_meta"]["geometries_returned"] == 1
    assert result["_meta"]["total_records_returned"] == 2
    assert result["_meta"]["auth_required"] is True
    assert result["_meta"]["auth_present"] is True
    assert result["_meta"]["success"] is True
    assert "xco2" in result["_meta"]["variable_info"]


@pytest.mark.usefixtures("_set_token")
def test_nasa_oco2_point_query_no_granules(httpx_mock):
    httpx_mock.add_response(
        url=re.compile(rf"^{re.escape(CMR_GRANULES_URL)}.*"),
        json=_EMPTY_CMR_RESPONSE,
    )
    result = nasa_oco2_point_query(
        latitude=_LAT,
        longitude=_LON,
        start_date="2019-08-01",
        end_date="2019-08-02",
    )
    assert result["data"] == []
    assert result["_meta"]["success"] is True
    assert result["_meta"]["geometries_returned"] == 0
    assert result["_meta"]["total_records_returned"] == 0


@pytest.mark.usefixtures("_unset_token")
def test_nasa_oco2_point_query_no_auth():
    result = nasa_oco2_point_query(
        latitude=_LAT,
        longitude=_LON,
        start_date="2019-08-01",
        end_date="2019-08-02",
    )
    assert result["data"] == []
    assert result["_meta"]["success"] is False
    assert "EARTHDATA_TOKEN" in result["_meta"]["error"]


# ---------------------------------------------------------------------------
# nasa_oco2_bbox_query
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_bbox_query_mock", "_set_token")
def test_nasa_oco2_bbox_query():
    result = nasa_oco2_bbox_query(
        min_lat=_LAT - 0.5,
        max_lat=_LAT + 1.5,
        min_lon=_LON - 0.5,
        max_lon=_LON + 1.5,
        start_date="2019-08-01",
        end_date="2019-08-02",
    )
    assert len(result["data"]) == 2
    assert result["_meta"]["source"] == "nasa_oco2"
    assert result["_meta"]["geometries_returned"] == 2
    assert result["_meta"]["total_records_returned"] == 3
    assert result["_meta"]["auth_required"] is True
    assert result["_meta"]["auth_present"] is True
    assert result["_meta"]["success"] is True


@pytest.mark.usefixtures("_set_token")
def test_nasa_oco2_bbox_query_no_granules(httpx_mock):
    httpx_mock.add_response(
        url=re.compile(rf"^{re.escape(CMR_GRANULES_URL)}.*"),
        json=_EMPTY_CMR_RESPONSE,
    )
    result = nasa_oco2_bbox_query(
        min_lat=_LAT - 0.5,
        max_lat=_LAT + 1.5,
        min_lon=_LON - 0.5,
        max_lon=_LON + 1.5,
        start_date="2019-08-01",
        end_date="2019-08-02",
    )
    assert result["data"] == []
    assert result["_meta"]["success"] is True
    assert result["_meta"]["geometries_returned"] == 0


@pytest.mark.usefixtures("_unset_token")
def test_nasa_oco2_bbox_query_no_auth():
    result = nasa_oco2_bbox_query(
        min_lat=_LAT - 0.5,
        max_lat=_LAT + 1.5,
        min_lon=_LON - 0.5,
        max_lon=_LON + 1.5,
        start_date="2019-08-01",
        end_date="2019-08-02",
    )
    assert result["data"] == []
    assert result["_meta"]["success"] is False
    assert "EARTHDATA_TOKEN" in result["_meta"]["error"]

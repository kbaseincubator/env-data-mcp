"""Shared fixtures and mock endpoints for OpenAQ unit tests."""

from __future__ import annotations

import sqlite3

import pytest
from hishel import FilterPolicy, SyncSqliteStorage
from hishel.httpx import SyncCacheClient

from env_data_mcp.sources.openaq._constants import OPENAQ_BASE_URL
from env_data_mcp.sources.openaq._query import _SuccessOnlyFilter

_LAT = 46.2531882
_LON = -119.4768203
_API_KEY = "test-api-key-1234"

_PARAMETERS_RESPONSE = {
    "results": [
        {
            "id": 111,
            "name": "pm25",
            "description": "particulate matter <= 2.5 microns",
            "units": "ug/m3",
        },
        {"id": 222, "name": "no2", "description": "nitrogen dioxide", "units": "ppb"},
        {"id": 333, "name": "o3", "description": "ozone", "units": "ppb"},
    ]
}

_LOCATION_RESPONSE = {
    "results": [
        {
            "id": 999,
            "name": "Yakima Monitor",
            "coordinates": {"latitude": _LAT, "longitude": _LON},
            "sensors": [
                {"id": 312, "parameter": {"id": 111, "name": "pm25", "units": "µg/m3"}},
                {"id": 523, "parameter": {"id": 222, "name": "no2", "units": "ppb"}},
            ],
        }
    ]
}

_MEASUREMENT_RESPONSE_PM25 = {
    "results": [
        {
            "value": 12.5,
            "period": {"datetimeFrom": {"local": "2019-08-19T10:00:00-07:00"}},
            "coordinates": {"latitude": _LAT, "longitude": _LON},
        },
        {
            "value": 14.1,
            "period": {"datetimeFrom": {"local": "2019-08-19T11:00:00-07:00"}},
            "coordinates": {"latitude": _LAT, "longitude": _LON},
        },
    ]
}

_MEASUREMENT_RESPONSE_NO2 = {
    "results": [
        {
            "value": 5.0,
            "period": {"datetimeFrom": {"local": "2019-08-19T10:00:00-07:00"}},
            "coordinates": {"latitude": _LAT, "longitude": _LON},
        }
    ]
}

_EMPTY_MEASUREMENTS = {"results": []}


@pytest.fixture(autouse=True)
def _reset_client_cache(monkeypatch):
    monkeypatch.setattr(
        "env_data_mcp.sources.openaq._query._client",
        SyncCacheClient(
            policy=FilterPolicy(response_filters=[_SuccessOnlyFilter()]),
            storage=SyncSqliteStorage(
                connection=sqlite3.connect(":memory:", check_same_thread=False)
            ),
            timeout=30.0,
        ),
    )


@pytest.fixture
def _set_api_key(monkeypatch):
    monkeypatch.setenv("OPENAQ_API_KEY", _API_KEY)


@pytest.fixture
def _unset_api_key(monkeypatch):
    monkeypatch.delenv("OPENAQ_API_KEY", raising=False)


@pytest.fixture
def _point_query_mock(httpx_mock, monkeypatch):
    monkeypatch.setattr("env_data_mcp.sources.openaq._query._AVAILABLE_PARAMETERS", None)
    httpx_mock.add_response(
        url=f"{OPENAQ_BASE_URL}/parameters",
        json=_PARAMETERS_RESPONSE,
        is_reusable=True,
    )
    httpx_mock.add_response(
        url=f"{OPENAQ_BASE_URL}/locations?coordinates={_LAT},{_LON}&radius={5000.0}",
        json=_LOCATION_RESPONSE,
    )
    httpx_mock.add_response(
        url=f"{OPENAQ_BASE_URL}/sensors/523/measurements?date_from=2019-08-19T00:00:00Z&date_to=2019-08-19T23:59:59Z",
        json=_MEASUREMENT_RESPONSE_NO2,
        is_optional=True,
    )
    httpx_mock.add_response(
        url=f"{OPENAQ_BASE_URL}/sensors/312/measurements?date_from=2019-08-19T00:00:00Z&date_to=2019-08-19T23:59:59Z",
        json=_MEASUREMENT_RESPONSE_PM25,
        is_optional=True,
    )


@pytest.fixture
def _bbox_query_mock(httpx_mock, monkeypatch):
    monkeypatch.setattr("env_data_mcp.sources.openaq._query._AVAILABLE_PARAMETERS", None)
    httpx_mock.add_response(
        url=f"{OPENAQ_BASE_URL}/parameters",
        json=_PARAMETERS_RESPONSE,
        is_reusable=True,
    )
    httpx_mock.add_response(
        url=f"{OPENAQ_BASE_URL}/locations?bbox={_LON},{_LAT},{_LON + 1.0},{_LAT + 1.0}",
        json=_LOCATION_RESPONSE,
    )
    httpx_mock.add_response(
        url=f"{OPENAQ_BASE_URL}/sensors/523/measurements?date_from=2019-08-19T00:00:00Z&date_to=2019-08-19T23:59:59Z",
        json=_MEASUREMENT_RESPONSE_NO2,
        is_optional=True,
    )
    httpx_mock.add_response(
        url=f"{OPENAQ_BASE_URL}/sensors/312/measurements?date_from=2019-08-19T00:00:00Z&date_to=2019-08-19T23:59:59Z",
        json=_MEASUREMENT_RESPONSE_PM25,
        is_optional=True,
    )

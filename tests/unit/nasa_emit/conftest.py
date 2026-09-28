"""Shared fixtures and mock endpoints for NASA EMIT unit tests."""

from __future__ import annotations

import io
import re

import h5py
import numpy as np
import pytest

from env_data_mcp.sources.nasa_emit._constants import CMR_GANULES_URL, QUERY_ARGS

_LAT = 46.2531882
_LON = -119.4768203
_TOKEN = "test-earthdata-token"

_GRANULE_ID = "emit20230815t000000_l2b_min_001"
_OPENDAP_BASE_URL = "https://opendap.earthdata.nasa.gov/emit/l2b/emit20230815"
# Resolved data-fetch base: _parse_granule_url() always appends ".nc4" to the CMR href.
_NC4_URL = f"{_OPENDAP_BASE_URL}.nc4"

_MINERAL_NAMES = ["Calcite", "Kaolinite"]

# 2x2 pixel grid; the point-query coordinates sit exactly on the (0, 0) cell
# so the nearest-pixel search is deterministic.
_LATS = np.array([[_LAT, _LAT], [_LAT + 0.01, _LAT + 0.01]])
_LONS = np.array([[_LON, _LON + 0.01], [_LON, _LON + 0.01]])

_POINT_ABUNDANCE = np.array([[[0.3, 0.15]]])  # shape (1, 1, n_minerals)
_BBOX_ABUNDANCE = np.array(
    [
        [[0.3, 0.15], [0.2, 0.05]],
        [[0.4, 0.01], [0.1, 0.02]],
    ]
)  # shape (2, 2, n_minerals)

_CMR_RESPONSE = {
    "feed": {
        "entry": [
            {
                "producer_granule_id": _GRANULE_ID,
                "title": _GRANULE_ID,
                "time_start": "2023-08-15T12:00:00.000Z",
                "links": [
                    {
                        "rel": "http://esipfed.org/ns/fedsearch/1.1/opendap#",
                        "href": f"{_OPENDAP_BASE_URL}.dmr",
                    }
                ],
            }
        ]
    }
}

_EMPTY_CMR_RESPONSE = {"feed": {"entry": []}}


def _make_lat_lon_nc4(
    lats: np.ndarray = _LATS,
    lons: np.ndarray = _LONS,
    names: list[str] = _MINERAL_NAMES,
) -> bytes:
    """Return HDF5 bytes containing lat/lon grids and a mineral name list."""
    buf = io.BytesIO()
    with h5py.File(buf, "w") as hf:
        loc = hf.require_group("location")
        loc.create_dataset("lat", data=lats)
        loc.create_dataset("lon", data=lons)
        mm = hf.require_group("mineral_metadata")
        dt = h5py.special_dtype(vlen=str)
        ds = mm.create_dataset("mineral_name", (len(names),), dtype=dt)
        for k, name in enumerate(names):
            ds[k] = name
    buf.seek(0)
    return buf.read()


def _make_abundance_nc4(abundance: np.ndarray) -> bytes:
    """Return HDF5 bytes containing a spectral_abundance array."""
    buf = io.BytesIO()
    with h5py.File(buf, "w") as hf:
        hf.create_dataset("spectral_abundance", data=abundance)
    buf.seek(0)
    return buf.read()


@pytest.fixture
def _set_token(monkeypatch):
    monkeypatch.setenv("EARTHDATA_TOKEN", _TOKEN)


@pytest.fixture
def _unset_token(monkeypatch):
    monkeypatch.delenv("EARTHDATA_TOKEN", raising=False)


@pytest.fixture
def _cmr_mock(httpx_mock):
    httpx_mock.add_response(
        url=re.compile(rf"^{re.escape(CMR_GANULES_URL)}.*"),
        json=_CMR_RESPONSE,
    )


@pytest.fixture
def _point_query_mock(httpx_mock, _cmr_mock):
    httpx_mock.add_response(
        url=f"{_NC4_URL}?{QUERY_ARGS}",
        content=_make_lat_lon_nc4(),
    )
    httpx_mock.add_response(
        url=f"{_NC4_URL}?/spectral_abundance[0:0][0:0][0:1]",
        content=_make_abundance_nc4(_POINT_ABUNDANCE),
    )


@pytest.fixture
def _bbox_query_mock(httpx_mock, _cmr_mock):
    httpx_mock.add_response(
        url=f"{_NC4_URL}?{QUERY_ARGS}",
        content=_make_lat_lon_nc4(),
    )
    httpx_mock.add_response(
        url=f"{_NC4_URL}?/spectral_abundance[0:1][0:1][0:1]",
        content=_make_abundance_nc4(_BBOX_ABUNDANCE),
    )

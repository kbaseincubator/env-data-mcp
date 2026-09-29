"""Shared fixtures and mock endpoints for NASA OCO2 unit tests."""

from __future__ import annotations

import io
import re

import h5py
import numpy as np
import pytest

from env_data_mcp.sources.nasa_oco2._constants import CMR_GRANULES_URL

_LAT = 46.2531882
_LON = -119.4768203
_TOKEN = "test-earthdata-token"

_GRANULE_ID_1 = "oco2_LtCO2_190801_test001"
_GRANULE_ID_2 = "oco2_LtCO2_190802_test002"
_DOWNLOAD_URL_1 = "https://ges-disc.gsfc.nasa.gov/data/oco2_190801.he5"
_DOWNLOAD_URL_2 = "https://ges-disc.gsfc.nasa.gov/data/oco2_190802.he5"

_FILL = -9999.0

# Small explicit-coordinate grid; the query point sits exactly on cell (0, 0)
# so the nearest-cell search is deterministic.
_NROWS, _NCOLS = 3, 3
_LATS = np.array([_LAT, _LAT + 1.0, _LAT + 2.0], dtype=np.float32)
_LONS = np.array([_LON, _LON + 1.0, _LON + 2.0], dtype=np.float32)

_CMR_RESPONSE = {
    "feed": {
        "entry": [
            {
                "producer_granule_id": _GRANULE_ID_1,
                "title": _GRANULE_ID_1,
                "time_start": "2019-08-01T00:00:00.000Z",
                "links": [
                    {"rel": "http://esipfed.org/ns/fedsearch/1.1/data#", "href": _DOWNLOAD_URL_1}
                ],
            },
            {
                "producer_granule_id": _GRANULE_ID_2,
                "title": _GRANULE_ID_2,
                "time_start": "2019-08-02T00:00:00.000Z",
                "links": [
                    {"rel": "http://esipfed.org/ns/fedsearch/1.1/data#", "href": _DOWNLOAD_URL_2}
                ],
            },
        ]
    }
}

_EMPTY_CMR_RESPONSE = {"feed": {"entry": []}}


def _make_xco2_hdf5(
    values: np.ndarray,
    uncertainties: np.ndarray | None = None,
    lats: np.ndarray = _LATS,
    lons: np.ndarray = _LONS,
) -> bytes:
    """Return HDF5 bytes shaped like the OCO-2 GEOS L3 format.

    3-D XCO2 array (1, nrows, ncols), mol/mol units, plus explicit lat/lon
    coordinate arrays (so the nearest-cell search is deterministic in tests).
    """
    buf = io.BytesIO()
    data = values.reshape(1, *values.shape).astype(np.float32)
    with h5py.File(buf, "w") as hf:
        grp = hf.require_group("HDFEOS/GRIDS/OCO-2 Level 3 Gridded XCO2/Data Fields")
        ds = grp.create_dataset("XCO2", data=data)
        ds.attrs["_FillValue"] = _FILL
        if uncertainties is not None:
            prec = uncertainties.reshape(1, *uncertainties.shape).astype(np.float32)
            prec_ds = grp.create_dataset("XCO2PREC", data=prec)
            prec_ds.attrs["_FillValue"] = _FILL
        hf.create_dataset("lat", data=lats)
        hf.create_dataset("lon", data=lons)
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
        url=re.compile(rf"^{re.escape(CMR_GRANULES_URL)}.*"),
        json=_CMR_RESPONSE,
    )


@pytest.fixture
def _point_query_mock(httpx_mock, _cmr_mock):
    # Both days have a real value at cell (0, 0), which matches (_LAT, _LON).
    day1 = np.full((_NROWS, _NCOLS), _FILL, dtype=np.float32)
    day1[0, 0] = 408.5e-6  # mol/mol -> ~408.5 ppm
    day1_unc = np.full((_NROWS, _NCOLS), _FILL, dtype=np.float32)
    day1_unc[0, 0] = 0.5e-6

    day2 = np.full((_NROWS, _NCOLS), _FILL, dtype=np.float32)
    day2[0, 0] = 409.2e-6
    day2_unc = np.full((_NROWS, _NCOLS), _FILL, dtype=np.float32)
    day2_unc[0, 0] = 0.6e-6

    httpx_mock.add_response(url=_DOWNLOAD_URL_1, content=_make_xco2_hdf5(day1, day1_unc))
    httpx_mock.add_response(url=_DOWNLOAD_URL_2, content=_make_xco2_hdf5(day2, day2_unc))


@pytest.fixture
def _bbox_query_mock(httpx_mock, _cmr_mock):
    # Cell (0, 0) has data on both days; cell (1, 1) only on day 1.
    day1 = np.full((_NROWS, _NCOLS), _FILL, dtype=np.float32)
    day1[0, 0] = 408.5e-6
    day1[1, 1] = 410.0e-6

    day2 = np.full((_NROWS, _NCOLS), _FILL, dtype=np.float32)
    day2[0, 0] = 409.2e-6

    httpx_mock.add_response(url=_DOWNLOAD_URL_1, content=_make_xco2_hdf5(day1))
    httpx_mock.add_response(url=_DOWNLOAD_URL_2, content=_make_xco2_hdf5(day2))

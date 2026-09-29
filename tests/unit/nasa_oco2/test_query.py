"""Unit tests for env_data_mcp.sources.nasa_oco2._query."""

from __future__ import annotations

import io
import re
from http import HTTPStatus

import h5py
import numpy as np
import pytest

from env_data_mcp.sources.nasa_oco2._constants import CMR_GRANULES_URL, Granule
from env_data_mcp.sources.nasa_oco2._query import (
    _add_record,
    _build_headers,
    _cells_to_groups,
    _fetch_granule_bytes,
    _get_download_url,
    _get_fill_value,
    _grid_params,
    _open_xco2_dataset,
    _parse_granule_date,
    _parse_granules,
    _parse_xco2_bbox_cells,
    _parse_xco2_point_cell,
    _point_to_idx,
    _query_granules,
    _to_ppm,
    query_bbox,
    query_point,
)

from .conftest import (
    _CMR_RESPONSE,
    _DOWNLOAD_URL_1,
    _FILL,
    _GRANULE_ID_1,
    _LAT,
    _LON,
    _NCOLS,
    _NROWS,
    _TOKEN,
    _make_xco2_hdf5,
)

# ---------------------------------------------------------------------------
# _build_headers
# ---------------------------------------------------------------------------


def test_build_headers():
    assert _build_headers(_TOKEN) == {"Authorization": f"Bearer {_TOKEN}"}


# ---------------------------------------------------------------------------
# _parse_granule_date
# ---------------------------------------------------------------------------


def test_parse_granule_date():
    assert _parse_granule_date("2019-08-19T15:00:00.000Z") == "2019-08-19"


def test_parse_granule_date_fallback():
    assert _parse_granule_date("not-a-date-at-all") == "not-a-date"


# ---------------------------------------------------------------------------
# _get_download_url
# ---------------------------------------------------------------------------


def test_get_download_url_prefers_opendap():
    g = {
        "links": [
            {"rel": "http://esipfed.org/ns/fedsearch/1.1/data#", "href": "https://x/data.he5"},
            {
                "rel": "http://esipfed.org/ns/fedsearch/1.1/opendap#",
                "href": "https://opendap.earthdata.nasa.gov/data.he5",
            },
        ]
    }
    url = _get_download_url(g)
    assert url is not None
    assert "opendap" in url


def test_get_download_url_falls_back_to_data():
    g = {
        "links": [
            {"rel": "http://esipfed.org/ns/fedsearch/1.1/data#", "href": "https://x/data.he5"},
        ]
    }
    assert _get_download_url(g) == "https://x/data.he5"


def test_get_download_url_returns_none_when_no_links():
    assert _get_download_url({"links": []}) is None


# ---------------------------------------------------------------------------
# _parse_granules
# ---------------------------------------------------------------------------


def test_parse_granules():
    granules = _parse_granules(_CMR_RESPONSE["feed"]["entry"])
    assert len(granules) == 2
    assert granules[0].id == _GRANULE_ID_1
    assert granules[0].date == "2019-08-01"
    assert granules[0].url == _DOWNLOAD_URL_1


def test_parse_granules_skips_entries_without_download_url():
    raw = [{"producer_granule_id": "no-url", "time_start": "2019-08-01T00:00:00.000Z", "links": []}]
    assert _parse_granules(raw) == []


# ---------------------------------------------------------------------------
# _query_granules
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_cmr_mock")
def test_query_granules():
    granules = _query_granules(start_date="2019-08-01", end_date="2019-08-02", token=_TOKEN)
    assert len(granules) == 2
    assert granules[0].id == _GRANULE_ID_1


def test_query_granules_unauthorized(httpx_mock):
    httpx_mock.add_response(
        url=re.compile(rf"^{re.escape(CMR_GRANULES_URL)}.*"),
        status_code=HTTPStatus.UNAUTHORIZED,
    )
    with pytest.raises(ValueError, match="token rejected"):
        _query_granules(start_date="2019-08-01", end_date="2019-08-02", token=_TOKEN)


# ---------------------------------------------------------------------------
# _fetch_granule_bytes
# ---------------------------------------------------------------------------


def test_fetch_granule_bytes(httpx_mock):
    content = b"fake-hdf5-bytes"
    httpx_mock.add_response(url=_DOWNLOAD_URL_1, content=content)
    assert _fetch_granule_bytes(_DOWNLOAD_URL_1, _TOKEN) == content


def test_fetch_granule_bytes_unauthorized(httpx_mock):
    httpx_mock.add_response(url=_DOWNLOAD_URL_1, status_code=HTTPStatus.UNAUTHORIZED)
    with pytest.raises(ValueError, match="token rejected"):
        _fetch_granule_bytes(_DOWNLOAD_URL_1, _TOKEN)


# ---------------------------------------------------------------------------
# _get_fill_value / _open_xco2_dataset
# ---------------------------------------------------------------------------


def test_get_fill_value():
    values = np.full((_NROWS, _NCOLS), _FILL, dtype=np.float32)
    content = _make_xco2_hdf5(values)
    with h5py.File(io.BytesIO(content), "r") as hf:
        ds = _open_xco2_dataset(hf)
        assert ds is not None
        assert _get_fill_value(ds) == pytest.approx(_FILL)


def test_open_xco2_dataset_fallback_path():
    """XCO2 stored at a non-standard path is found via the visititems fallback."""
    buf = io.BytesIO()
    with h5py.File(buf, "w") as hf:
        grp = hf.require_group("CUSTOM/PATH")
        grp.create_dataset("XCO2", data=np.ones((3, 3), dtype=np.float32))
    buf.seek(0)
    with h5py.File(buf, "r") as hf:
        assert _open_xco2_dataset(hf) is not None


def test_open_xco2_dataset_missing_returns_none():
    buf = io.BytesIO()
    with h5py.File(buf, "w") as hf:
        hf.create_dataset("something_else", data=np.ones((3,)))
    buf.seek(0)
    with h5py.File(buf, "r") as hf:
        assert _open_xco2_dataset(hf) is None


# ---------------------------------------------------------------------------
# _grid_params / _point_to_idx
# ---------------------------------------------------------------------------


def test_grid_params_half_deg():
    lat_orig, lon_orig, lat_step, lon_step = _grid_params(360, 576)
    assert lat_step == pytest.approx(0.5)
    assert lon_step == pytest.approx(360.0 / 576)
    assert lat_orig == pytest.approx(-89.75)


def test_point_to_idx_clamped_high():
    lat_orig, lon_orig, lat_step, lon_step = _grid_params(180, 360)
    i, j = _point_to_idx(91.0, 181.0, lat_orig, lon_orig, lat_step, lon_step, 180, 360)
    assert i == 179
    assert j == 359


def test_point_to_idx_clamped_low():
    lat_orig, lon_orig, lat_step, lon_step = _grid_params(180, 360)
    i, j = _point_to_idx(-91.0, -181.0, lat_orig, lon_orig, lat_step, lon_step, 180, 360)
    assert i == 0
    assert j == 0


# ---------------------------------------------------------------------------
# _to_ppm
# ---------------------------------------------------------------------------


def test_to_ppm_converts_mol_fraction():
    assert _to_ppm(4.085e-4) == pytest.approx(408.5)


def test_to_ppm_leaves_ppm_unchanged():
    assert _to_ppm(408.5) == pytest.approx(408.5)


# ---------------------------------------------------------------------------
# _parse_xco2_point_cell
# ---------------------------------------------------------------------------


def test_parse_xco2_point_cell_returns_value():
    values = np.full((_NROWS, _NCOLS), _FILL, dtype=np.float32)
    values[0, 0] = 408.5e-6
    unc = np.full((_NROWS, _NCOLS), _FILL, dtype=np.float32)
    unc[0, 0] = 0.5e-6
    content = _make_xco2_hdf5(values, unc)
    cell = _parse_xco2_point_cell(content, _LAT, _LON)
    assert cell is not None
    lat, lon, xco2, xco2_unc = cell
    assert lat == pytest.approx(_LAT, abs=1e-3)
    assert lon == pytest.approx(_LON, abs=1e-3)
    assert xco2 == pytest.approx(408.5, abs=0.01)
    assert xco2_unc == pytest.approx(0.5, abs=0.01)


def test_parse_xco2_point_cell_fill_returns_none():
    values = np.full((_NROWS, _NCOLS), _FILL, dtype=np.float32)
    content = _make_xco2_hdf5(values)
    assert _parse_xco2_point_cell(content, _LAT, _LON) is None


def test_parse_xco2_point_cell_no_uncertainty():
    values = np.full((_NROWS, _NCOLS), _FILL, dtype=np.float32)
    values[0, 0] = 408.5e-6
    content = _make_xco2_hdf5(values)
    cell = _parse_xco2_point_cell(content, _LAT, _LON)
    assert cell is not None
    assert cell[3] is None


def test_parse_xco2_point_cell_inferred_grid():
    """No explicit lat/lon in the file: falls back to inferred grid math."""
    buf = io.BytesIO()
    nrows, ncols = 180, 360
    lat_orig, lon_orig, lat_step, lon_step = _grid_params(nrows, ncols)
    i, j = _point_to_idx(_LAT, _LON, lat_orig, lon_orig, lat_step, lon_step, nrows, ncols)
    data = np.full((nrows, ncols), _FILL, dtype=np.float32)
    data[i, j] = 408.5
    with h5py.File(buf, "w") as hf:
        grp = hf.require_group("HDFEOS/GRIDS/OCO-2 Level 3 Gridded XCO2/Data Fields")
        ds = grp.create_dataset("XCO2", data=data)
        ds.attrs["_FillValue"] = _FILL
    buf.seek(0)
    cell = _parse_xco2_point_cell(buf.read(), _LAT, _LON)
    assert cell is not None
    assert cell[2] == pytest.approx(408.5, abs=0.01)


# ---------------------------------------------------------------------------
# _parse_xco2_bbox_cells
# ---------------------------------------------------------------------------


def test_parse_xco2_bbox_cells_returns_cells():
    values = np.full((_NROWS, _NCOLS), _FILL, dtype=np.float32)
    values[0, 0] = 408.5e-6
    values[1, 1] = 410.0e-6
    content = _make_xco2_hdf5(values)
    cells = _parse_xco2_bbox_cells(
        content, min_lat=_LAT - 0.5, max_lat=_LAT + 1.5, min_lon=_LON - 0.5, max_lon=_LON + 1.5
    )
    assert len(cells) == 2
    xco2_values = sorted(c[2] for c in cells)
    assert xco2_values[0] == pytest.approx(408.5, abs=0.01)
    assert xco2_values[1] == pytest.approx(410.0, abs=0.01)


def test_parse_xco2_bbox_cells_outside_grid_returns_empty():
    values = np.full((_NROWS, _NCOLS), _FILL, dtype=np.float32)
    values[0, 0] = 408.5e-6
    content = _make_xco2_hdf5(values)
    cells = _parse_xco2_bbox_cells(content, min_lat=80.0, max_lat=85.0, min_lon=0.0, max_lon=5.0)
    assert cells == []


def test_parse_xco2_bbox_cells_all_fill_returns_empty():
    values = np.full((_NROWS, _NCOLS), _FILL, dtype=np.float32)
    content = _make_xco2_hdf5(values)
    cells = _parse_xco2_bbox_cells(
        content, min_lat=_LAT - 0.5, max_lat=_LAT + 2.5, min_lon=_LON - 0.5, max_lon=_LON + 2.5
    )
    assert cells == []


# ---------------------------------------------------------------------------
# _add_record / _cells_to_groups
# ---------------------------------------------------------------------------


def test_add_record_and_cells_to_groups():
    cells: dict[tuple[float, float], list[dict]] = {}
    granule1 = Granule(id="g1", date="2019-08-02", url="https://x/1")
    granule2 = Granule(id="g2", date="2019-08-01", url="https://x/2")
    _add_record(cells, (_LAT, _LON, 408.5, 0.5), granule1)
    _add_record(cells, (_LAT, _LON, 409.2, None), granule2)

    groups = _cells_to_groups(cells)
    assert len(groups) == 1
    group = groups[0]
    assert group["geometry"] == {"type": "Point", "coordinates": [_LON, _LAT]}
    # Sorted by date ascending regardless of insertion order.
    assert [r["date"] for r in group["records"]] == ["2019-08-01", "2019-08-02"]
    rec1 = group["records"][1]
    assert rec1["xco2"] == 408.5
    assert rec1["xco2_units"] == "ppm"
    assert rec1["xco2_uncertainty"] == 0.5
    assert rec1["xco2_uncertainty_units"] == "ppm"
    assert rec1["granule_id"] == "g1"
    rec0 = group["records"][0]
    assert "xco2_uncertainty" not in rec0


# ---------------------------------------------------------------------------
# query_point
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_point_query_mock")
def test_query_point():
    groups = query_point(
        latitude=_LAT, longitude=_LON, start_date="2019-08-01", end_date="2019-08-02", token=_TOKEN
    )
    assert len(groups) == 1
    group = groups[0]
    assert group["latitude"] == pytest.approx(_LAT, abs=1e-3)
    assert group["longitude"] == pytest.approx(_LON, abs=1e-3)
    assert len(group["records"]) == 2
    assert [r["date"] for r in group["records"]] == ["2019-08-01", "2019-08-02"]


# ---------------------------------------------------------------------------
# query_bbox
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_bbox_query_mock")
def test_query_bbox():
    groups = query_bbox(
        min_lat=_LAT - 0.5,
        max_lat=_LAT + 1.5,
        min_lon=_LON - 0.5,
        max_lon=_LON + 1.5,
        start_date="2019-08-01",
        end_date="2019-08-02",
        token=_TOKEN,
    )
    assert len(groups) == 2
    record_counts = sorted(len(g["records"]) for g in groups)
    assert record_counts == [1, 2]
    two_record_group = next(g for g in groups if len(g["records"]) == 2)
    assert two_record_group["latitude"] == pytest.approx(_LAT, abs=1e-3)
    assert two_record_group["longitude"] == pytest.approx(_LON, abs=1e-3)

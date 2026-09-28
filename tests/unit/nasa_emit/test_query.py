"""Unit tests for env_data_mcp.sources.nasa_emit._query."""

from __future__ import annotations

import io
import re
from http import HTTPStatus

import h5py
import numpy as np
import pytest

from env_data_mcp.sources.nasa_emit._constants import (
    CMR_GANULES_URL,
    QUERY_ARGS,
    Granule,
)
from env_data_mcp.sources.nasa_emit._query import (
    _build_headers,
    _decode_mineral_names,
    _extract_pixels_in_bbox,
    _fetch_nc4_file,
    _find_nearest_pixel,
    _get_dataset,
    _mineral_records_for_bbox,
    _mineral_records_for_pixel,
    _parse_granule_date,
    _parse_granule_url,
    _parse_granules,
    _query_granule_bbox,
    _query_granule_point,
    _query_granules,
    query_bbox,
    query_point,
)

from .conftest import (
    _BBOX_ABUNDANCE,
    _CMR_RESPONSE,
    _GRANULE_ID,
    _LAT,
    _LATS,
    _LON,
    _LONS,
    _MINERAL_NAMES,
    _NC4_URL,
    _OPENDAP_BASE_URL,
    _POINT_ABUNDANCE,
    _TOKEN,
    _make_abundance_nc4,
    _make_lat_lon_nc4,
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
    assert _parse_granule_date("2023-08-15T12:00:00.000Z") == "2023-08-15"


def test_parse_granule_date_fallback():
    assert _parse_granule_date("not-a-date-at-all") == "not-a-date"


# ---------------------------------------------------------------------------
# _parse_granule_url
# ---------------------------------------------------------------------------


def test_parse_granule_url_opendap_rel():
    links = [
        {"rel": "http://esipfed.org/ns/fedsearch/1.1/opendap#", "href": f"{_OPENDAP_BASE_URL}.dmr"}
    ]
    assert _parse_granule_url(links) == _NC4_URL


def test_parse_granule_url_opendap_in_href():
    links = [{"rel": "unrelated", "href": f"{_OPENDAP_BASE_URL}/opendap/file.nc4"}]
    assert _parse_granule_url(links) == f"{_OPENDAP_BASE_URL}/opendap/file.nc4"


def test_parse_granule_url_missing_raises():
    with pytest.raises(ValueError, match="Missing URL"):
        _parse_granule_url([{"rel": "data", "href": "https://example.com/file.nc"}])


# ---------------------------------------------------------------------------
# _parse_granules
# ---------------------------------------------------------------------------


def test_parse_granules():
    granules = _parse_granules(_CMR_RESPONSE["feed"]["entry"])
    assert len(granules) == 1
    g = granules[0]
    assert g.id == _GRANULE_ID
    assert g.date == "2023-08-15"
    assert g.nc4_link == _NC4_URL


# ---------------------------------------------------------------------------
# _get_dataset
# ---------------------------------------------------------------------------


def test_get_dataset_found():
    content = _make_lat_lon_nc4()
    with h5py.File(io.BytesIO(content), "r") as hf:
        ds = _get_dataset(hf, "/location/lat", "location/lat", "lat")
        assert ds is not None


def test_get_dataset_fallback_path():
    buf = io.BytesIO()
    with h5py.File(buf, "w") as hf:
        hf.create_dataset("lat", data=np.ones((2, 2)))
    buf.seek(0)
    with h5py.File(buf, "r") as hf:
        ds = _get_dataset(hf, "/location/lat", "lat")
        assert ds is not None


def test_get_dataset_missing_raises():
    buf = io.BytesIO()
    with h5py.File(buf, "w") as hf:
        hf.create_dataset("something_else", data=np.ones((3,)))
    buf.seek(0)
    with h5py.File(buf, "r") as hf, pytest.raises(ValueError, match="No HDF5 dataset found"):
        _get_dataset(hf, "/location/lat", "lat")


# ---------------------------------------------------------------------------
# _decode_mineral_names
# ---------------------------------------------------------------------------


def test_decode_mineral_names():
    content = _make_lat_lon_nc4()
    with h5py.File(io.BytesIO(content), "r") as hf:
        ds = hf["mineral_metadata/mineral_name"]
        assert isinstance(ds, h5py.Dataset)
        assert _decode_mineral_names(ds) == _MINERAL_NAMES


def test_decode_mineral_names_bytes():
    """Mineral names stored as byte strings (older HDF5 files)."""
    buf = io.BytesIO()
    with h5py.File(buf, "w") as hf:
        hf.create_dataset("mineral_name", data=np.array([b"Calcite", b"Kaolinite"], dtype="S20"))
    buf.seek(0)
    with h5py.File(buf, "r") as hf:
        ds = hf["mineral_name"]
        assert isinstance(ds, h5py.Dataset)
        assert _decode_mineral_names(ds) == ["Calcite", "Kaolinite"]


# ---------------------------------------------------------------------------
# _find_nearest_pixel
# ---------------------------------------------------------------------------


def test_find_nearest_pixel_exact_match():
    assert _find_nearest_pixel(_LATS, _LONS, _LAT, _LON) == (0, 0)


def test_find_nearest_pixel_approx_match():
    assert _find_nearest_pixel(_LATS, _LONS, _LAT + 0.009, _LON + 0.001) == (1, 0)


# ---------------------------------------------------------------------------
# _extract_pixels_in_bbox
# ---------------------------------------------------------------------------


def test_extract_pixels_in_bbox_all_inside():
    pixels = _extract_pixels_in_bbox(
        lat_arr=_LATS,
        lon_arr=_LONS,
        min_lat=_LAT - 1.0,
        max_lat=_LAT + 1.0,
        min_lon=_LON - 1.0,
        max_lon=_LON + 1.0,
    )
    assert set(pixels) == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_extract_pixels_in_bbox_partial():
    pixels = _extract_pixels_in_bbox(
        lat_arr=_LATS,
        lon_arr=_LONS,
        min_lat=_LAT - 1.0,
        max_lat=_LAT + 0.001,
        min_lon=_LON - 1.0,
        max_lon=_LON + 1.0,
    )
    assert set(pixels) == {(0, 0), (0, 1)}


def test_extract_pixels_in_bbox_empty():
    pixels = _extract_pixels_in_bbox(
        lat_arr=_LATS, lon_arr=_LONS, min_lat=80.0, max_lat=90.0, min_lon=0.0, max_lon=10.0
    )
    assert pixels == []


# ---------------------------------------------------------------------------
# _fetch_nc4_file
# ---------------------------------------------------------------------------


def test_fetch_nc4_file(httpx_mock):
    content = _make_lat_lon_nc4()
    httpx_mock.add_response(url=f"{_OPENDAP_BASE_URL}?{QUERY_ARGS}", content=content)
    assert _fetch_nc4_file(_OPENDAP_BASE_URL, QUERY_ARGS, _TOKEN) == content


def test_fetch_nc4_file_unauthorized(httpx_mock):
    httpx_mock.add_response(
        url=f"{_OPENDAP_BASE_URL}?{QUERY_ARGS}", status_code=HTTPStatus.UNAUTHORIZED
    )
    with pytest.raises(ValueError, match="token rejected"):
        _fetch_nc4_file(_OPENDAP_BASE_URL, QUERY_ARGS, _TOKEN)


# ---------------------------------------------------------------------------
# _query_granules
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_cmr_mock")
def test_query_granules():
    granules = _query_granules(
        min_lat=_LAT - 0.1,
        max_lat=_LAT + 0.1,
        min_lon=_LON - 0.1,
        max_lon=_LON + 0.1,
        start_date="2023-08-01",
        end_date="2023-08-31",
        token=_TOKEN,
    )
    assert len(granules) == 1
    assert granules[0].id == _GRANULE_ID
    assert granules[0].nc4_link == _NC4_URL


def test_query_granules_unauthorized(httpx_mock):
    httpx_mock.add_response(
        url=re.compile(rf"^{re.escape(CMR_GANULES_URL)}.*"),
        status_code=HTTPStatus.UNAUTHORIZED,
    )
    with pytest.raises(ValueError, match="token rejected"):
        _query_granules(
            min_lat=_LAT - 0.1,
            max_lat=_LAT + 0.1,
            min_lon=_LON - 0.1,
            max_lon=_LON + 0.1,
            start_date="2023-08-01",
            end_date="2023-08-31",
            token=_TOKEN,
        )


# ---------------------------------------------------------------------------
# _mineral_records_for_pixel
# ---------------------------------------------------------------------------


def test_mineral_records_for_pixel(httpx_mock):
    granule = Granule(id=_GRANULE_ID, date="2023-08-15", nc4_link=_NC4_URL)
    httpx_mock.add_response(
        url=f"{_NC4_URL}?/spectral_abundance[0:0][0:0][0:1]",
        content=_make_abundance_nc4(_POINT_ABUNDANCE),
    )
    records = _mineral_records_for_pixel(
        granule=granule,
        i=0,
        j=0,
        mineral_names=_MINERAL_NAMES,
        pixel_lat=_LAT,
        pixel_lon=_LON,
        token=_TOKEN,
    )
    assert len(records) == 1
    group = records[0]
    assert group["geometry"] == {"type": "Point", "coordinates": [_LON, _LAT]}
    assert group["latitude"] == _LAT
    assert group["longitude"] == _LON
    assert len(group["records"]) == 2
    for rec, name, val in zip(group["records"], _MINERAL_NAMES, [0.3, 0.15], strict=True):
        assert rec["mineral_name"] == name
        assert rec["abundance"] == pytest.approx(val)
        assert rec["units"] == "fractional (0-1)"
        assert rec["aquisition_date"] == "2023-08-15"
        assert rec["granule_id"] == _GRANULE_ID


# ---------------------------------------------------------------------------
# _mineral_records_for_bbox
# ---------------------------------------------------------------------------


def test_mineral_records_for_bbox(httpx_mock):
    granule = Granule(id=_GRANULE_ID, date="2023-08-15", nc4_link=_NC4_URL)
    httpx_mock.add_response(
        url=f"{_NC4_URL}?/spectral_abundance[0:1][0:1][0:1]",
        content=_make_abundance_nc4(_BBOX_ABUNDANCE),
    )
    pixels = [(0, 0), (0, 1), (1, 0), (1, 1)]
    records = _mineral_records_for_bbox(
        granule=granule,
        pixels=pixels,
        mineral_names=_MINERAL_NAMES,
        lat_arr=_LATS,
        lon_arr=_LONS,
        token=_TOKEN,
    )
    assert len(records) == 4
    by_coords = {tuple(g["geometry"]["coordinates"]): g for g in records}
    assert by_coords[(_LON, _LAT)]["records"][0]["abundance"] == pytest.approx(0.3)
    assert by_coords[(_LON, _LAT)]["records"][1]["abundance"] == pytest.approx(0.15)
    assert by_coords[(_LON + 0.01, _LAT + 0.01)]["records"][1]["abundance"] == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# _query_granule_point / _query_granule_bbox
# ---------------------------------------------------------------------------


def test_query_granule_point(httpx_mock):
    granule = Granule(id=_GRANULE_ID, date="2023-08-15", nc4_link=_NC4_URL)
    httpx_mock.add_response(url=f"{_NC4_URL}?{QUERY_ARGS}", content=_make_lat_lon_nc4())
    httpx_mock.add_response(
        url=f"{_NC4_URL}?/spectral_abundance[0:0][0:0][0:1]",
        content=_make_abundance_nc4(_POINT_ABUNDANCE),
    )
    records = _query_granule_point(granule=granule, latitude=_LAT, longitude=_LON, token=_TOKEN)
    assert len(records) == 1
    assert records[0]["latitude"] == _LAT
    assert records[0]["longitude"] == _LON
    assert len(records[0]["records"]) == 2


def test_query_granule_bbox(httpx_mock):
    granule = Granule(id=_GRANULE_ID, date="2023-08-15", nc4_link=_NC4_URL)
    httpx_mock.add_response(url=f"{_NC4_URL}?{QUERY_ARGS}", content=_make_lat_lon_nc4())
    httpx_mock.add_response(
        url=f"{_NC4_URL}?/spectral_abundance[0:1][0:1][0:1]",
        content=_make_abundance_nc4(_BBOX_ABUNDANCE),
    )
    records = _query_granule_bbox(
        granule=granule,
        min_lat=_LAT - 1.0,
        max_lat=_LAT + 1.0,
        min_lon=_LON - 1.0,
        max_lon=_LON + 1.0,
        token=_TOKEN,
    )
    assert len(records) == 4


def test_query_granule_bbox_no_pixels(httpx_mock):
    granule = Granule(id=_GRANULE_ID, date="2023-08-15", nc4_link=_NC4_URL)
    httpx_mock.add_response(url=f"{_NC4_URL}?{QUERY_ARGS}", content=_make_lat_lon_nc4())
    records = _query_granule_bbox(
        granule=granule, min_lat=80.0, max_lat=90.0, min_lon=0.0, max_lon=10.0, token=_TOKEN
    )
    assert records == []


# ---------------------------------------------------------------------------
# query_point
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_point_query_mock")
def test_query_point():
    records = query_point(
        latitude=_LAT,
        longitude=_LON,
        start_date="2023-08-01",
        end_date="2023-08-31",
        token=_TOKEN,
    )
    assert len(records) == 1
    assert records[0]["latitude"] == _LAT
    assert records[0]["longitude"] == _LON
    assert len(records[0]["records"]) == 2
    assert records[0]["records"][0]["granule_id"] == _GRANULE_ID


# ---------------------------------------------------------------------------
# query_bbox
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_bbox_query_mock")
def test_query_bbox():
    records = query_bbox(
        min_lat=_LAT - 1.0,
        max_lat=_LAT + 1.0,
        min_lon=_LON - 1.0,
        max_lon=_LON + 1.0,
        start_date="2023-08-01",
        end_date="2023-08-31",
        token=_TOKEN,
    )
    assert len(records) == 4

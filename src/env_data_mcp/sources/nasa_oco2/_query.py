"""Query functions for the NASA OCO2 adapter."""

from __future__ import annotations

import io
import re
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from typing import Any, cast

import h5py
import httpx
import numpy as np

from ._constants import (
    CMR_GRANULES_URL,
    COLLECTION_SHORT_NAME,
    FILL_VALUE,
    PAGE_SIZE,
    VERSION,
    XCO2_PATHS,
    XCO2PREC_SUFFIXES,
    Granule,
)

_MAX_WORKERS = 10


def query_point(
    *,
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    token: str,
) -> list[dict[str, Any]]:
    granules = _query_granules(start_date=start_date, end_date=end_date, token=token)
    cells: dict[tuple[float, float], list[dict[str, Any]]] = {}

    def _fetch(granule: Granule) -> tuple[Granule, _Cell | None] | None:
        try:
            content = _fetch_granule_bytes(granule.url, token)
        except httpx.HTTPStatusError:
            # Some CMR-indexed granules aren't resolvable via download link (reprocessed/removed).
            return None
        return granule, _parse_xco2_point_cell(content, latitude, longitude)

    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        results = list(pool.map(_fetch, granules))

    for result in results:
        if result is None:
            continue
        granule, cell = result
        if cell is None:
            continue
        _add_record(cells, cell, granule)

    return _cells_to_groups(cells)


def query_bbox(
    *,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    start_date: str,
    end_date: str,
    token: str,
) -> list[dict[str, Any]]:
    granules = _query_granules(start_date=start_date, end_date=end_date, token=token)
    cells: dict[tuple[float, float], list[dict[str, Any]]] = {}

    def _fetch(granule: Granule) -> tuple[Granule, list[_Cell]] | None:
        try:
            content = _fetch_granule_bytes(granule.url, token)
        except httpx.HTTPStatusError:
            # Some CMR-indexed granules aren't resolvable via download link (reprocessed/removed).
            return None
        return granule, _parse_xco2_bbox_cells(content, min_lat, max_lat, min_lon, max_lon)

    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        results = list(pool.map(_fetch, granules))

    for result in results:
        if result is None:
            continue
        granule, day_cells = result
        for cell in day_cells:
            _add_record(cells, cell, granule)

    return _cells_to_groups(cells)


# ---------------------------------------------------------------------------
# auth helper
# ---------------------------------------------------------------------------


def _build_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# granule helpers
# ---------------------------------------------------------------------------


def _query_granules(*, start_date: str, end_date: str, token: str) -> list[Granule]:
    """Return all OCO-2 L3 granules whose day falls within [start_date, end_date].

    OCO-2 GEOS L3 is one file per day, global coverage - the CMR search is
    temporal-only (no bounding_box param).
    """
    params = {
        "short_name": COLLECTION_SHORT_NAME,
        "version": VERSION,
        "temporal[]": f"{start_date}T00:00:00Z,{end_date}T23:59:59Z",
        "page_size": PAGE_SIZE,
        "sort_key": "start_date",
    }
    header = _build_headers(token=token)
    all_granules: list[Granule] = []
    search_after: str | None = None

    while True:
        req_headers = dict(header)
        if search_after is not None:
            req_headers["CMR-Search-After"] = search_after

        resp = httpx.get(
            url=CMR_GRANULES_URL,
            params=params,
            headers=req_headers,
            timeout=30.0,
            follow_redirects=True,
        )
        if resp.status_code == HTTPStatus.UNAUTHORIZED:
            raise ValueError(
                "EarthData token rejected (HTTP 401). Token may be expired. "
                "Regenerate at https://urs.earthdata.nasa.gov/ > Profile > Generate Token "
                "and update EARTHDATA_TOKEN environment variable."
            )
        resp.raise_for_status()
        entries = resp.json().get("feed", {}).get("entry", [])
        all_granules.extend(_parse_granules(entries))

        search_after = resp.headers.get("CMR-Search-After")
        if not search_after or len(entries) < PAGE_SIZE:
            break

    return all_granules


def _parse_granules(raw: list[dict[str, Any]]) -> list[Granule]:
    """Build a Granule per entry, skipping entries with no usable download link."""
    granules: list[Granule] = []
    for g in raw:
        url = _get_download_url(g)
        if not url:
            continue
        granules.append(
            Granule(
                id=g.get("producer_granule_id") or g.get("title", ""),
                date=_parse_granule_date(g.get("time_start", "")),
                url=url,
            )
        )
    return granules


def _parse_granule_date(raw: str) -> str:
    m = re.match(r"(\d{4}-\d{2}-\d{2})", raw)
    return m.group(1) if m else raw[:10]


_DOWNLOAD_FILE_EXTS = (".he5", ".nc", ".nc4", ".h5")


def _get_download_url(granule: dict[str, Any]) -> str | None:
    """Return the best download URL from a CMR granule entry.

    Prefers OPeNDAP file links (rel contains 'opendap', direct file), then
    direct data-download links, then any non-HTML link.
    """
    links = granule.get("links", [])
    for link in links:
        rel = link.get("rel", "")
        href = link.get("href", "")
        if "opendap" in rel.lower() and any(href.endswith(ext) for ext in _DOWNLOAD_FILE_EXTS):
            return href
    for link in links:
        rel = link.get("rel", "")
        href = link.get("href", "")
        if "data#" in rel and any(href.endswith(ext) for ext in _DOWNLOAD_FILE_EXTS):
            return href
    for link in links:
        href = link.get("href", "")
        if href.startswith("https://") and not href.endswith(".html"):
            return href
    return None


def _fetch_granule_bytes(url: str, token: str) -> bytes:
    resp = httpx.get(url, headers=_build_headers(token), timeout=120.0, follow_redirects=True)
    if resp.status_code == HTTPStatus.UNAUTHORIZED:
        raise ValueError(
            "EarthData token rejected (HTTP 401). Token may be expired. "
            "Regenerate at https://urs.earthdata.nasa.gov/ > Profile > Generate Token "
            "and update EARTHDATA_TOKEN environment variable."
        )
    resp.raise_for_status()
    return resp.content


# ---------------------------------------------------------------------------
# HDF5 parsing helpers
# ---------------------------------------------------------------------------


def _get_fill_value(ds: h5py.Dataset) -> float:
    """Extract the fill value, handling numpy 0-d or 1-d array attributes."""
    fill = ds.attrs.get("_FillValue", FILL_VALUE)
    if hasattr(fill, "flat"):
        return float(next(iter(fill.flat)))
    return float(fill)


def _open_xco2_dataset(hf: h5py.File) -> h5py.Dataset | None:
    """Return the XCO2 dataset from an open h5py file, trying known paths."""
    for path in XCO2_PATHS:
        try:
            ds = hf[path]
            if isinstance(ds, h5py.Dataset):
                return ds
        except KeyError:
            continue
    # Fallback: walk all datasets and find one named XCO2.
    result: list[h5py.Dataset] = []

    def _visit(name: str, obj: Any) -> None:
        if (
            isinstance(obj, h5py.Dataset)
            and "XCO2" in name.upper()
            and not any(s in name.upper() for s in ("PREC", "ERROR", "APRIORI", "OBS"))
        ):
            result.append(obj)

    hf.visititems(_visit)
    return result[0] if result else None


def _grid_params(nrows: int, ncols: int) -> tuple[float, float, float, float]:
    """Infer (lat_origin, lon_origin, lat_step, lon_step) from grid dimensions.

    OCO-2 L3 grids are cell-centred with the first cell at half-step from the
    coordinate origin.
    """
    lat_step = 180.0 / nrows
    lon_step = 360.0 / ncols
    lat_origin = -90.0 + lat_step / 2.0
    lon_origin = -180.0 + lon_step / 2.0
    return lat_origin, lon_origin, lat_step, lon_step


def _point_to_idx(
    lat: float,
    lon: float,
    lat_origin: float,
    lon_origin: float,
    lat_step: float,
    lon_step: float,
    nrows: int,
    ncols: int,
) -> tuple[int, int]:
    """Convert WGS84 (lat, lon) to (row, col) indices on the OCO-2 L3 grid."""
    i = int(round((lat - lat_origin) / lat_step))
    j = int(round((lon - lon_origin) / lon_step))
    return max(0, min(nrows - 1, i)), max(0, min(ncols - 1, j))


def _find_prec_dataset(hf: h5py.File, xco2_ds: h5py.Dataset) -> h5py.Dataset | None:
    base = (xco2_ds.name or "").rsplit("/", 1)[0]
    for suf in XCO2PREC_SUFFIXES:
        candidate = base + suf
        if candidate in hf:
            obj = hf[candidate]
            if isinstance(obj, h5py.Dataset):
                return obj
    return None


def _to_ppm(val: float) -> float:
    """GEOS L3 stores XCO2 as mol/mol; legacy files already use ppm."""
    return val * 1e6 if val < 1.0 else val


# A parsed grid cell: (lat, lon, xco2_ppm, xco2_uncertainty_ppm | None).
_Cell = tuple[float, float, float, float | None]


def _parse_xco2_point_cell(content: bytes, lat: float, lon: float) -> _Cell | None:
    """Open HDF5 content and extract XCO2 at the nearest grid cell to (lat, lon).

    Returns None if the cell is fill-valued (no data at this location/date).
    """
    with h5py.File(io.BytesIO(content), "r") as hf:
        xco2_ds = _open_xco2_dataset(hf)
        if xco2_ds is None:
            return None
        data = xco2_ds[:]
        if data.ndim == 3:
            data = data[0]  # squeeze daily time dimension (GEOS L3 format)
        nrows, ncols = data.shape
        fill = _get_fill_value(xco2_ds)

        if "lat" in hf and "lon" in hf:
            lat_arr: np.ndarray = cast(h5py.Dataset, hf["lat"])[:]
            lon_arr: np.ndarray = cast(h5py.Dataset, hf["lon"])[:]
            i = int(np.argmin(np.abs(lat_arr - lat)))
            j = int(np.argmin(np.abs(lon_arr - lon)))
            actual_lat = round(float(lat_arr[i]), 4)
            actual_lon = round(float(lon_arr[j]), 4)
        else:
            lat_orig, lon_orig, lat_step, lon_step = _grid_params(nrows, ncols)
            i, j = _point_to_idx(lat, lon, lat_orig, lon_orig, lat_step, lon_step, nrows, ncols)
            actual_lat = round(lat_orig + i * lat_step, 4)
            actual_lon = round(lon_orig + j * lon_step, 4)

        val = float(data[i, j])
        if val <= fill or np.isnan(val):
            return None
        xco2 = round(_to_ppm(val), 3)

        prec: float | None = None
        prec_ds = _find_prec_dataset(hf, xco2_ds)
        if prec_ds is not None:
            p_data = prec_ds[:]
            if p_data.ndim == 3:
                p_data = p_data[0]
            p = float(p_data[i, j])
            if p > fill and not np.isnan(p):
                prec = round(_to_ppm(p), 3)

        return (actual_lat, actual_lon, xco2, prec)


def _parse_xco2_bbox_cells(
    content: bytes,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
) -> list[_Cell]:
    """Extract all non-fill XCO2 cells within the bounding box."""
    cells: list[_Cell] = []
    with h5py.File(io.BytesIO(content), "r") as hf:
        xco2_ds = _open_xco2_dataset(hf)
        if xco2_ds is None:
            return cells
        data = xco2_ds[:]
        if data.ndim == 3:
            data = data[0]
        nrows, ncols = data.shape
        fill = _get_fill_value(xco2_ds)

        if "lat" in hf and "lon" in hf:
            lat_arr: np.ndarray = cast(h5py.Dataset, hf["lat"])[:]
            lon_arr: np.ndarray = cast(h5py.Dataset, hf["lon"])[:]
            i_idx = np.where((lat_arr >= min_lat) & (lat_arr <= max_lat))[0]
            j_idx = np.where((lon_arr >= min_lon) & (lon_arr <= max_lon))[0]
            if len(i_idx) == 0 or len(j_idx) == 0:
                return cells
            i_lo, i_hi = int(i_idx[0]), int(i_idx[-1]) + 1
            j_lo, j_hi = int(j_idx[0]), int(j_idx[-1]) + 1
            row_lats = [round(float(lat_arr[i_lo + di]), 4) for di in range(i_hi - i_lo)]
            col_lons = [round(float(lon_arr[j_lo + dj]), 4) for dj in range(j_hi - j_lo)]
        else:
            lat_orig, lon_orig, lat_step, lon_step = _grid_params(nrows, ncols)
            args = (lat_orig, lon_orig, lat_step, lon_step, nrows, ncols)
            i_lo, _ = _point_to_idx(min_lat, min_lon, *args)
            i_hi, _ = _point_to_idx(max_lat, max_lon, *args)
            _, j_lo = _point_to_idx(min_lat, min_lon, *args)
            _, j_hi = _point_to_idx(max_lat, max_lon, *args)
            i_lo, i_hi = min(i_lo, i_hi), max(i_lo, i_hi) + 1
            j_lo, j_hi = min(j_lo, j_hi), max(j_lo, j_hi) + 1
            row_lats = [round(lat_orig + (i_lo + di) * lat_step, 4) for di in range(i_hi - i_lo)]
            col_lons = [round(lon_orig + (j_lo + dj) * lon_step, 4) for dj in range(j_hi - j_lo)]

        prec_ds = _find_prec_dataset(hf, xco2_ds)
        prec_data: np.ndarray | None = None
        if prec_ds is not None:
            prec_raw = prec_ds[:]
            if prec_raw.ndim == 3:
                prec_raw = prec_raw[0]
            prec_data = prec_raw[i_lo:i_hi, j_lo:j_hi]

        slice_data = data[i_lo:i_hi, j_lo:j_hi]
        for di in range(slice_data.shape[0]):
            for dj in range(slice_data.shape[1]):
                val = float(slice_data[di, dj])
                if val <= fill or np.isnan(val):
                    continue
                xco2 = round(_to_ppm(val), 3)
                prec: float | None = None
                if prec_data is not None:
                    p = float(prec_data[di, dj])
                    if p > fill and not np.isnan(p):
                        prec = round(_to_ppm(p), 3)
                cells.append((row_lats[di], col_lons[dj], xco2, prec))
    return cells


# ---------------------------------------------------------------------------
# geometry-group assembly
# ---------------------------------------------------------------------------


def _add_record(
    cells: dict[tuple[float, float], list[dict[str, Any]]], cell: _Cell, granule: Granule
) -> None:
    lat, lon, xco2, xco2_uncertainty = cell
    record: dict[str, Any] = {
        "date": granule.date,
        "xco2": xco2,
        "xco2_units": "ppm",
        "granule_id": granule.id,
    }
    if xco2_uncertainty is not None:
        record["xco2_uncertainty"] = xco2_uncertainty
        record["xco2_uncertainty_units"] = "ppm"
    cells.setdefault((lat, lon), []).append(record)


def _cells_to_groups(
    cells: dict[tuple[float, float], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for (lat, lon), records in cells.items():
        records.sort(key=lambda r: r["date"])
        groups.append(
            {
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "latitude": lat,
                "longitude": lon,
                "records": records,
            }
        )
    return groups

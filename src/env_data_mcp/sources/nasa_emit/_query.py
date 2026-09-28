"""Query functions for the NASA EMIT adapter."""

from __future__ import annotations

import io
import re
from http import HTTPStatus
from typing import Any

import h5py
import httpx
import numpy as np

from ._constants import (
    ABUNDANCE_PATHS,
    CMR_GANULES_URL,
    COLLECTION_SHORT_NAME,
    LAT_PATHS,
    LON_PATHS,
    MINERAL_NAME_PATHS,
    PAGE_SIZE,
    QUERY_ARGS,
    VERSION,
    Granule,
)


def query_point(
    *,
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    token: str,
) -> list[dict[str, Any]]:
    # Use a small bbox around the point for CMR spatial search
    pad = 0.01  # deg
    granules = _query_granules(
        min_lat=latitude - pad,
        max_lat=latitude + pad,
        min_lon=longitude - pad,
        max_lon=longitude + pad,
        start_date=start_date,
        end_date=end_date,
        token=token,
    )
    records: list[dict[str, Any]] = []
    for g in granules:
        if not g.nc4_link:
            raise ValueError(f"Missing NetCDF URL for granule {g.id}")
        batch = _query_granule_point(
            granule=g,
            latitude=latitude,
            longitude=longitude,
            token=token,
        )
        records.extend(batch)
    return records


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
    granules = _query_granules(
        min_lat=min_lat,
        max_lat=max_lat,
        min_lon=min_lon,
        max_lon=max_lon,
        start_date=start_date,
        end_date=end_date,
        token=token,
    )
    records: list[dict[str, Any]] = []
    for g in granules:
        if not g.nc4_link:
            raise ValueError(f"Missing NetCDF URL for granule {g.id}")
        batch = _query_granule_bbox(
            granule=g,
            min_lat=min_lat,
            max_lat=max_lat,
            min_lon=min_lon,
            max_lon=max_lon,
            token=token,
        )
        records.extend(batch)
    return records


# ---------------------------------------------------------------------------
# auth helper
# ---------------------------------------------------------------------------


def _build_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# granule helpers
# ---------------------------------------------------------------------------


def _query_granules(
    *,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    start_date: str,
    end_date: str,
    token: str,
) -> list[Granule]:
    """Return all EMIT L2B granules whose spatial footprint overlaps the bbox."""
    params = {
        "short_name": COLLECTION_SHORT_NAME,
        "version": VERSION,
        "bounding_box": f"{min_lon},{min_lat},{max_lon},{max_lat}",
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
            url=CMR_GANULES_URL,
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
        granules = _parse_granules(entries)
        all_granules.extend(granules)

        search_after = resp.headers.get("CMR-Search-After")
        if not search_after or len(entries) < PAGE_SIZE:
            break

    return all_granules


def _parse_granules(raw: list[dict[str, Any]]) -> list[Granule]:
    return [
        Granule(
            id=g.get("producer_granule_id") or g.get("title", ""),
            date=_parse_granule_date(g.get("time_start", "")),
            nc4_link=_parse_granule_url(g.get("links", [])),
        )
        for g in raw
    ]


def _parse_granule_date(raw: str) -> str:
    m = re.match(r"(\d{4}-\d{2}-\d{2})", raw)
    return m.group(1) if m else raw[:10]


def _parse_granule_url(raw: list[dict[str, Any]]) -> str:
    for link in raw:
        rel = link.get("rel", "")
        href = link.get("href", "")
        if "opendap" in rel.lower() or "opendap" in href.lower():
            # Hyrax requires an explicit ".nc4" suffix to return a subsettable netCDF4/HDF5
            # response; the bare granule href returns an HTML landing page instead.
            base = re.sub(r"\.(dap|nc4|dmr|dds|das)$", "", href)
            return f"{base}.nc4"
    raise ValueError("Missing URL for granule data file")


def _fetch_nc4_file(base_url: str, var_expr: str, token: str) -> bytes:
    url = f"{base_url}?{var_expr}"
    resp = httpx.get(url, headers=_build_headers(token), timeout=120.0, follow_redirects=True)
    if resp.status_code == HTTPStatus.UNAUTHORIZED:
        raise ValueError(
            "EarthData token rejected (HTTP 401). Token may be expired. "
            "Regenerate at https://urs.earthdata.nasa.gov/ > Profile > Generate Token "
            "and update EARTHDATA_TOKEN environment variable."
        )
    resp.raise_for_status()
    return resp.content


def _get_dataset(hf: h5py.File, *paths: str) -> h5py.Dataset:
    """Try a sequence of HDF5 paths and return the first one that resolves."""
    for p in paths:
        try:
            obj = hf[p]
            if isinstance(obj, h5py.Dataset):
                return obj
        except KeyError:
            continue
    raise ValueError(f"No HDF5 dataset found in {paths}")


def _decode_mineral_names(ds: h5py.Dataset) -> list[str]:
    raw = ds[:]
    names: list[str] = []
    for item in raw.flat:
        if isinstance(item, bytes):
            names.append(item.decode("utf-8", errors="replace").strip())
        else:
            names.append(str(item).strip())
    return names


def _find_nearest_pixel(
    lat_arr: np.ndarray,
    lon_arr: np.ndarray,
    query_lat: float,
    query_lon: float,
) -> tuple[int, int]:
    dist2 = (lat_arr - query_lat) ** 2 + (lon_arr - query_lon) ** 2
    flat_idx = int(np.argmin(dist2))
    idx = np.unravel_index(flat_idx, lat_arr.shape)
    return (int(idx[0]), int(idx[1]))


def _extract_pixels_in_bbox(
    *,
    lat_arr: np.ndarray,
    lon_arr: np.ndarray,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
) -> list[tuple[int, int]]:
    mask = (lat_arr >= min_lat) & (lat_arr <= max_lat) & (lon_arr >= min_lon) & (lon_arr <= max_lon)
    (
        rows,
        cols,
    ) = np.where(mask)
    return list(zip(rows.tolist(), cols.tolist(), strict=True))


def _mineral_records_for_pixel(
    *,
    granule: Granule,
    i: int,
    j: int,
    mineral_names: list[str],
    pixel_lat: float,
    pixel_lon: float,
    token: str,
) -> list[dict[str, Any]]:
    n = len(mineral_names)
    var_expr = f"/spectral_abundance[{i}:{i}][{j}:{j}][0:{n - 1}]"
    content = _fetch_nc4_file(granule.nc4_link, var_expr, token)

    with h5py.File(io.BytesIO(content), "r") as hf:
        abund_ds = _get_dataset(hf, *ABUNDANCE_PATHS)
        abundance = abund_ds[:].ravel()

    return [
        {
            "geometry": {"type": "Point", "coordinates": [pixel_lon, pixel_lat]},
            "latitude": pixel_lat,
            "longitude": pixel_lon,
            "records": [
                {
                    "mineral_name": name,
                    "abundance": float(val),
                    "units": "fractional (0-1)",
                    "aquisition_date": granule.date,
                    "granule_id": granule.id,
                }
                for name, val in zip(mineral_names, abundance, strict=False)
            ],
        }
    ]


def _mineral_records_for_bbox(
    *,
    granule: Granule,
    pixels: list[tuple[int, int]],
    mineral_names: list[str],
    lat_arr: np.ndarray,
    lon_arr: np.ndarray,
    token: str,
) -> list[dict[str, Any]]:
    n = len(mineral_names)
    rows = [p[0] for p in pixels]
    cols = [p[1] for p in pixels]
    i_lo, i_hi = min(rows), max(rows)
    j_lo, j_hi = min(cols), max(cols)
    var_expr = f"/spectral_abundance[{i_lo}:{i_hi}][{j_lo}:{j_hi}][0:{n - 1}]"
    content = _fetch_nc4_file(granule.nc4_link, var_expr, token)

    with h5py.File(io.BytesIO(content), "r") as hf:
        abund_ds = _get_dataset(hf, *ABUNDANCE_PATHS)
        abundance = abund_ds[:]

    return [
        {
            "geometry": {
                "type": "Point",
                "coordinates": [float(lon_arr[i, j]), float(lat_arr[i, j])],
            },
            "latitude": float(lat_arr[i, j]),
            "longitude": float(lon_arr[i, j]),
            "records": [
                {
                    "mineral_name": name,
                    "abundance": float(val),
                    "units": "fractional (0-1)",
                    "aquisition_date": granule.date,
                    "granule_id": granule.id,
                }
                for name, val in zip(mineral_names, abundance[i - i_lo, j - j_lo, :], strict=False)
            ],
        }
        for i, j in pixels
    ]


def _query_granule_point(
    *,
    granule: Granule,
    latitude: float,
    longitude: float,
    token: str,
) -> list[dict[str, Any]]:
    nc4_bytes = _fetch_nc4_file(granule.nc4_link, QUERY_ARGS, token)

    with h5py.File(io.BytesIO(nc4_bytes), "r") as hf:
        lat_ds = _get_dataset(hf, *LAT_PATHS)
        lon_ds = _get_dataset(hf, *LON_PATHS)
        mineral_ds = _get_dataset(hf, *MINERAL_NAME_PATHS)
        lat_arr = lat_ds[:].astype(float)
        lon_arr = lon_ds[:].astype(float)
        mineral_names = _decode_mineral_names(mineral_ds)

    i, j = _find_nearest_pixel(lat_arr, lon_arr, latitude, longitude)
    pixel_lat = float(lat_arr[i, j])
    pixel_lon = float(lon_arr[i, j])

    return _mineral_records_for_pixel(
        granule=granule,
        i=i,
        j=j,
        mineral_names=mineral_names,
        pixel_lat=pixel_lat,
        pixel_lon=pixel_lon,
        token=token,
    )


def _query_granule_bbox(
    *,
    granule: Granule,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    token: str,
) -> list[dict[str, Any]]:
    nc4_bytes = _fetch_nc4_file(granule.nc4_link, QUERY_ARGS, token)

    with h5py.File(io.BytesIO(nc4_bytes), "r") as hf:
        lat_ds = _get_dataset(hf, *LAT_PATHS)
        lon_ds = _get_dataset(hf, *LON_PATHS)
        mineral_ds = _get_dataset(hf, *MINERAL_NAME_PATHS)
        lat_arr = lat_ds[:].astype(float)
        lon_arr = lon_ds[:].astype(float)
        mineral_names = _decode_mineral_names(mineral_ds)

    pixels = _extract_pixels_in_bbox(
        lat_arr=lat_arr,
        lon_arr=lon_arr,
        min_lat=min_lat,
        max_lat=max_lat,
        min_lon=min_lon,
        max_lon=max_lon,
    )
    if not pixels:
        return []

    return _mineral_records_for_bbox(
        granule=granule,
        pixels=pixels,
        mineral_names=mineral_names,
        lat_arr=lat_arr,
        lon_arr=lon_arr,
        token=token,
    )

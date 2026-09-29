"""Query functions for the NASA OCO2 adapter."""

from __future__ import annotations

from typing import Any


def query_point(
    *,
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    token: str,
) -> list[dict[str, Any]]:
    return []


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
    return []
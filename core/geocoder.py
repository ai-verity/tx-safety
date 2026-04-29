"""Geocoding helpers for Texas cities — Nominatim (no API key)."""
from __future__ import annotations
import asyncio
import httpx
from typing import Optional

# Hardcoded fallback coords for major TX cities (for offline / rate-limit resilience)
TX_CITY_COORDS: dict[str, tuple[float, float]] = {
    "houston":        (29.7604, -95.3698),
    "dallas":         (32.7767, -96.7970),
    "san antonio":    (29.4241, -98.4936),
    "austin":         (30.2672, -97.7431),
    "fort worth":     (32.7555, -97.3308),
    "el paso":        (31.7619, -106.4850),
    "arlington":      (32.7357, -97.1081),
    "corpus christi": (27.8006, -97.3964),
    "plano":          (33.0198, -96.6989),
    "laredo":         (27.5064, -99.5075),
    "lubbock":        (33.5779, -101.8552),
    "garland":        (32.9126, -96.6389),
    "irving":         (32.8140, -96.9489),
    "amarillo":       (35.2220, -101.8313),
    "grand prairie":  (32.7460, -96.9978),
    "brownsville":    (25.9017, -97.4975),
    "mckinney":       (33.1972, -96.6397),
    "frisco":         (33.1507, -96.8236),
    "pasadena":       (29.6911, -95.2091),
    "mcallen":        (26.2034, -98.2300),
    "killeen":        (31.1171, -97.7278),
    "mesquite":       (32.7668, -96.5992),
    "midland":        (31.9973, -102.0779),
    "waco":           (31.5493, -97.1467),
    "abilene":        (32.4487, -99.7331),
    "beaumont":       (30.0802, -94.1266),
    "odessa":         (31.8457, -102.3676),
    "tyler":          (32.3513, -95.3011),
    "carrollton":     (32.9537, -96.8903),
    "round rock":     (30.5083, -97.6789),
    "denton":         (33.2148, -97.1331),
    "lewisville":     (33.0462, -96.9942),
    "sugar land":     (29.6197, -95.6349),
    "cedar park":     (30.5052, -97.8203),
    "longview":       (32.5007, -94.7405),
    "edinburg":       (26.3017, -98.1633),
    "wichita falls":  (33.9137, -98.4934),
    "san angelo":     (31.4638, -100.4370),
    "temple":         (31.0982, -97.3428),
    "mission":        (26.2159, -98.3253),
    "pearland":       (29.5635, -95.2860),
    "college station":(30.6280, -96.3344),
    "conroe":         (30.3119, -95.4561),
    "new braunfels":  (29.7030, -98.1245),
    "harlingen":      (26.1906, -97.6961),
    "league city":    (29.5075, -95.0949),
    "allen":          (33.1032, -96.6705),
    "richardson":     (32.9483, -96.7299),
    "el paso":        (31.7619, -106.4850),
    "san marcos":     (29.8833, -97.9414),
    "pharr":          (26.1948, -98.1836),
    "baytown":        (29.7355, -94.9774),
    "rowlett":        (32.9029, -96.5636),
    "atascocita":     (29.9980, -95.1769),
    "flower mound":   (33.0146, -97.0964),
    "missouri city":  (29.6185, -95.5377),
    "north richland hills": (32.8343, -97.2289),
    "mansfield":      (32.5632, -97.1417),
    "wylie":          (33.0151, -96.5388),
    "burleson":       (32.5418, -97.3208),
    "victoria":       (28.8053, -97.0036),
    "waxahachie":     (32.3868, -96.8489),
    "lufkin":         (31.3382, -94.7291),
    "nacogdoches":    (31.6035, -94.6557),
    "texarkana":      (33.4251, -94.0477),
    "galveston":      (29.3013, -94.7977),
    "amarillo":       (35.2220, -101.8313),
    "lubbock":        (33.5779, -101.8552),
}

_geo_cache: dict[str, tuple[float, float]] = {}


async def geocode_city(city: str, county: str = "") -> tuple[Optional[float], Optional[float]]:
    """Return (lat, lon) for a Texas city. Uses cache → hardcoded → Nominatim."""
    key = city.lower().strip()
    if key in _geo_cache:
        return _geo_cache[key]
    if key in TX_CITY_COORDS:
        coords = TX_CITY_COORDS[key]
        _geo_cache[key] = coords
        return coords
    # Partial match
    for known, coords in TX_CITY_COORDS.items():
        if known in key or key in known:
            _geo_cache[key] = coords
            return coords
    # Try Nominatim as last resort
    try:
        query = f"{city}, Texas, USA"
        async with httpx.AsyncClient(timeout=8) as client:
            r = await client.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": query, "format": "json", "limit": 1},
                headers={"User-Agent": "TX-Safety-Dashboard/1.0"},
            )
            results = r.json()
            if results:
                lat = float(results[0]["lat"])
                lon = float(results[0]["lon"])
                _geo_cache[key] = (lat, lon)
                return lat, lon
    except Exception:
        pass
    return None, None


def map_coords(lat: float, lon: float) -> tuple[float, float]:
    """Convert lat/lon to SVG x/y coords for the Texas map viewport (520×420)."""
    # Texas bounding box approx: lat 25.8–36.5, lon -106.6 to -93.5
    lat_min, lat_max = 25.8, 36.5
    lon_min, lon_max = -106.6, -93.5
    # Map viewport offsets (match SVG path)
    x = ((lon - lon_min) / (lon_max - lon_min)) * 420 + 40
    y = ((lat_max - lat) / (lat_max - lat_min)) * 370 + 30
    return round(x, 1), round(y, 1)

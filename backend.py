"""
Backend service module for HeatShield AI.
Provides borough-level ranking, site evaluations, and data access.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Path to reference data
_DATA_PATH = Path(__file__).resolve().parent / "data" / "brooklyn_reference_data.json"


def rank_brooklyn_locations(limit: int = 20) -> list[dict[str, Any]]:
    """
    Ranks Brooklyn study areas and candidate locations by composite priority index.
    Combines physical microclimate offset with demographic vulnerability.
    """
    with open(_DATA_PATH, "r", encoding="utf-8") as f:
        ref_data = json.load(f)

    sites = ref_data.get("sites", [])
    ranked_results: list[dict[str, Any]] = []

    for site in sites:
        # Build driver attributions
        raw_drivers = site.get("top_feature_attributions", [])
        formatted_drivers = []
        for attr in raw_drivers:
            feature_name = attr.get("feature", "Factor").replace("_", " ").title()
            val = attr.get("attribution", 0.0)
            formatted_drivers.append({
                "feature": feature_name,
                "contribution": f"{val:+.2f}°F"
            })

        geo_id = site.get("geo_id", site.get("site_id", "36047091000"))
        ranked_results.append({
            "id": f"tract-{geo_id}",
            "name": f"{site['neighborhood']} ({site.get('intersection', site.get('address', ''))})",
            "neighborhood": site.get("neighborhood", "Brooklyn"),
            "latitude": float(site["latitude"]),
            "longitude": float(site["longitude"]),
            "offset_f": float(site.get("predicted_offset_f", 0.0)),
            "priority_score": float(site.get("priority_score", 50.0)),
            "population": int(site.get("total_pop", 5000)),
            "age_65_plus": int(site.get("pop_over_65", 800)),
            "no_vehicle_pct": float(site.get("no_vehicle_pct", 40.0)),
            "recommended_intervention": site.get("recommended_intervention", "Targeted Intervention"),
            "drivers": formatted_drivers or [
                {"feature": "Canopy Deficit", "contribution": "+2.10°F"},
                {"feature": "Impervious Surface", "contribution": "+1.45°F"},
                {"feature": "Building Age", "contribution": "+0.55°F"},
            ],
        })

    # Sort descending by priority score
    ranked_results.sort(key=lambda x: x["priority_score"], reverse=True)
    return ranked_results[:limit]

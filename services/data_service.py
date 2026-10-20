"""
Data Service Layer for HeatShield AI
------------------------------------
Abstracts data retrieval from local synthetic stores (in Demo mode)
or remote BigQuery / Google ADK backend APIs (in Live mode).
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def get_intersections(city: str = "brooklyn") -> pd.DataFrame:
    """
    Retrieves intersection microclimate and demographic dataset for specified city.
    """
    if "dallas" in city.lower():
        csv_path = DATA_DIR / "demo_intersections_dallas.csv"
    else:
        csv_path = DATA_DIR / "demo_intersections.csv"
        
    if not csv_path.exists():
        csv_path = DATA_DIR / "demo_intersections.csv"
    
    df = pd.read_csv(csv_path)
    # Ensure proper typing
    df["predicted_offset_f"] = df["predicted_offset_f"].astype(float)
    df["population"] = df["population"].astype(int)
    df["median_age"] = df["median_age"].astype(float)
    df["no_vehicle_rate"] = df["no_vehicle_rate"].astype(float)
    df["vulnerable_population"] = df["vulnerable_population"].astype(int)
    df["priority_score"] = df["priority_score"].astype(int)

    # Derived display strings
    df["offset_str"] = df["predicted_offset_f"].apply(lambda v: f"+{v:.1f}°F" if v >= 0 else f"{v:.1f}°F")
    df["vuln_pop_str"] = df["vulnerable_population"].apply(lambda v: f"{v:,}")
    df["total_pop_str"] = df["population"].apply(lambda v: f"{v:,}")
    df["no_veh_str"] = df["no_vehicle_rate"].apply(lambda v: f"{v:.1f}%")
    
    return df


def get_intersection_details(intersection_id: str, city: str = "brooklyn") -> Optional[Dict[str, Any]]:
    """Retrieves full record for a single intersection by ID and city."""
    df = get_intersections(city)
    matched = df[df["intersection_id"] == intersection_id]
    if matched.empty:
        return None
    return matched.iloc[0].to_dict()
    if matched.empty:
        return None
    return matched.iloc[0].to_dict()


def get_heat_explanations(intersection_id: str) -> List[Dict[str, Any]]:
    """
    Retrieves BigQuery ML ML.EXPLAIN_PREDICT feature attributions for an intersection.
    In Demo mode, reads from data/demo_explanations.json.
    """
    json_path = DATA_DIR / "demo_explanations.json"
    if not json_path.exists():
        return []
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return data.get(intersection_id, [
        { "factor": "Impervious Surface Coverage", "attribution_f": 1.5, "share_pct": 50, "description": "Baseline asphalt fraction." },
        { "factor": "Low Tree Canopy", "attribution_f": 1.0, "share_pct": 33, "description": "Street shade deficit." },
        { "factor": "Building Geometry", "attribution_f": 0.5, "share_pct": 17, "description": "Urban canyon radiation." }
    ])


def get_agent_recommendations(city: str = "brooklyn") -> List[Dict[str, Any]]:
    """
    Retrieves AI planning agent cooling resource recommendations.
    In Demo mode, reads from data/demo_recommendations.json.
    """
    json_path = DATA_DIR / "demo_recommendations.json"
    if not json_path.exists():
        return []
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return data.get("recommendations", [])


def get_historical_heat_data(city: str = "brooklyn") -> pd.DataFrame:
    """
    Retrieves station 7-year temperature frequency analysis (GHCN-Daily).
    Synthetic placeholder ready for BigQuery a4i_climate.station_daily_temps integration.
    """
    years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
    days_over_90 = [14, 18, 20, 17, 24, 22, 29]
    days_over_95 = [2, 4, 6, 3, 8, 7, 11]
    return pd.DataFrame({
        "year": years,
        "days_over_90f": days_over_90,
        "days_over_95f": days_over_95
    })


def calculate_priority_score(offset: float, vuln_pop: int, no_veh_rate: float) -> int:
    """
    Transparent illustrative planning heuristic combining:
      - Positive predicted heat offset (40% weight)
      - Vulnerability indicator (35% weight)
      - No-vehicle rate (25% weight)
    Note: Illustrative planning score, not a validated medical risk model.
    """
    norm_offset = max(0.0, min(10.0, offset + 3.1)) / 10.0  # normalize between -3.1 and +6.9
    norm_vuln = min(1.0, vuln_pop / 12000.0)
    norm_veh = min(1.0, no_veh_rate / 40.0)
    
    score = (norm_offset * 40.0) + (norm_vuln * 35.0) + (norm_veh * 25.0)
    return int(round(score))

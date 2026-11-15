"""Public accounting of measured inputs, model assumptions and missing evidence."""

from .config import RAW
from .gravity import available as gravity_available
from .horizon import N_AZ, OBSERVER_HEIGHT_M
from .ml.simulate import DEFAULT


def data_provenance() -> dict:
    return {
        "all_outputs_observational": False,
        "sources": {
            "terrain": "https://pds-geosciences.wustl.edu/missions/lro/lola.htm",
            "imagery": "https://svs.gsfc.nasa.gov/4720/",
            "ephemeris": "https://ssd.jpl.nasa.gov/planets/eph_export.html",
            "orientation": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/",
            "craters": "https://doi.org/10.1029/2018JE005592",
        },
        "ephemeris": {
            "method": "Local JPL DE421 + NAIF MOON_ME_DE421; not a live Horizons response",
            "kernels": {name: (RAW / name).is_file() for name in (
                "de421.bsp", "moon_080317.tf", "pck00011.tpc", "moon_pa_de421_1900-2050.bpc",
            )},
        },
        "gravity_grid_available": gravity_available(),
        "assumptions": {
            "touchdown_labels": "Monte-Carlo simulator on LOLA 20 m; no observed landing-outcome labels",
            "vehicle": "Generic HLS-class parameters; not verified CLPS vehicle specifications",
            "simulator": DEFAULT.to_dict(),
            "observer_height_m": OBSERVER_HEIGHT_M,
            "bulk_horizon_azimuth_step_deg": 360.0 / N_AZ,
            "conditions_horizon_azimuth_step_deg": 0.5,
            "scoring": "Application-defined thresholds and weights; not NASA-certified suitability scores",
            "regions": "Approximate centres and application-defined square search areas; not official boundary polygons",
            "divert": {"lateral_accel_mps2": 1.5, "hover_reserve_s": 75.0, "retarget_dv_budget_mps": 60.0},
        },
        "missing_evidence": [
            "Verified mission-specific CLPS geometry, navigation, propulsion, solar-panel, load and battery specifications",
            "Fine-resolution local terrain, rocks and regolith hazards below the 20/80 m DEM cell size",
            "Official candidate-region boundary polygons and coordinate provenance",
            "Integrated cold-trap, ice-abundance and science-target layers",
            "Mission-specific antenna pattern, ground-station availability, link budget and relay coverage",
            "Observed landing outcomes for validating success probabilities",
        ],
    }

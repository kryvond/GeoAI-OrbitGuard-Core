from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel, Field
import torch
from kryvond.models.terravision_vit import TerraVisionGeoModel
from kryvond.physics.orbitguard_pinn import OrbitGuardPINN

app = FastAPI(
    title="Kryvond Space — Geospatial & Orbital AI Inference API",
    description="Production GPU satellite semantic segmentation and physics-informed orbital conjunction engine.",
    version="1.0.0",
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_DIR = Path(__file__).resolve().parent.parent
GEO_CKPT = ROOT_DIR / "artifacts" / "checkpoints" / "terravision_vit_best.pt"
PINN_CKPT = ROOT_DIR / "artifacts" / "checkpoints" / "orbitguard_pinn_best.pt"

geo_model = TerraVisionGeoModel(in_bands=12, num_classes=5, base_dim=32).to(DEVICE)
if GEO_CKPT.exists():
    geo_model.load_state_dict(torch.load(GEO_CKPT, map_location=DEVICE, weights_only=True))
geo_model.eval()

orbit_model = OrbitGuardPINN(hidden_dim=128).to(DEVICE)
if PINN_CKPT.exists():
    orbit_model.load_state_dict(torch.load(PINN_CKPT, map_location=DEVICE, weights_only=True))
orbit_model.eval()


class OrbitalConjunctionRequest(BaseModel):
    sat_position_km: list[float] = Field(default=[6878.0, 0.0, 0.0], description="ECI [x, y, z] in km")
    sat_velocity_kms: list[float] = Field(default=[0.0, 5.4, 5.3], description="ECI [vx, vy, vz] in km/s")
    debris_position_km: list[float] = Field(default=[6880.5, 12.0, -8.0], description="Debris ECI [x, y, z] in km")
    debris_velocity_kms: list[float] = Field(default=[0.05, 5.38, 5.31], description="Debris ECI [vx, vy, vz] in km/s")
    horizon_minutes: float = Field(default=45.0, ge=5.0, le=180.0)


class TileSegmentationRequest(BaseModel):
    height: int = Field(default=64, ge=32, le=256)
    width: int = Field(default=64, ge=32, le=256)
    band_means: list[float] = Field(
        default=[0.12, 0.14, 0.28, 0.11, 0.35, 0.48, 0.55, 0.68, 0.22, 0.16, 0.42, 0.25],
        min_length=12,
        max_length=12,
    )


@app.get("/health")
async def health_check():
    return {
        "status": "operational",
        "platform": "Kryvond GeoAI & OrbitGuard Core",
        "version": "1.0.0",
        "cuda_available": torch.cuda.is_available(),
        "device": str(DEVICE),
        "geo_checkpoint_loaded": GEO_CKPT.exists(),
        "pinn_checkpoint_loaded": PINN_CKPT.exists(),
    }


@app.post("/v1/geo/segment-tile")
async def segment_multispectral_tile(req: TileSegmentationRequest):
    """Runs trained TerraVisionGeoModel on a 12-band tensor and returns class distribution."""
    base = torch.tensor(req.band_means, device=DEVICE, dtype=torch.float32).view(1, 12, 1, 1)
    tile = torch.clamp(base.expand(1, 12, req.height, req.width), 0.01, 1.0)
    with torch.no_grad():
        out = geo_model(tile)
        preds = out["probs"].argmax(dim=1)[0]

    class_names = {
        0: "barren_pasture",
        1: "critical_infrastructure_urban",
        2: "surface_water_flood",
        3: "vegetation_canopy",
        4: "industrial_energy_assets",
    }
    total_px = float(req.height * req.width)
    counts = torch.bincount(preds.flatten(), minlength=5)
    fractions = {
        name: round(float(counts[cid].item()) / total_px, 4)
        for cid, name in class_names.items()
    }
    dominant_cid = int(counts.argmax().item()) if preds.numel() > 0 else 0
    return {
        "dominant_class_id": dominant_cid,
        "dominant_class_label": class_names.get(dominant_cid, "unknown"),
        "class_coverage_fractions": fractions,
    }


@app.post("/v1/orbital/conjunction-screen")
async def screen_orbital_conjunction(req: OrbitalConjunctionRequest):
    """Propagates satellite & debris states via trained OrbitGuardPINN and flags miss distance."""
    R_E = 6378.137
    V_C = 7.905365

    t_steps = torch.linspace(0.0, (req.horizon_minutes * 60.0) / 806.811, 200, device=DEVICE).unsqueeze(-1)
    sat_s0 = torch.tensor(
        [p / R_E for p in req.sat_position_km] + [v / V_C for v in req.sat_velocity_kms],
        device=DEVICE, dtype=torch.float32
    ).unsqueeze(0).repeat(200, 1)

    deb_s0 = torch.tensor(
        [p / R_E for p in req.debris_position_km] + [v / V_C for v in req.debris_velocity_kms],
        device=DEVICE, dtype=torch.float32
    ).unsqueeze(0).repeat(200, 1)

    with torch.no_grad():
        sat_traj_km = orbit_model(t_steps, sat_s0) * R_E
        deb_traj_km = orbit_model(t_steps, deb_s0) * R_E
        dists_km = torch.norm(sat_traj_km - deb_traj_km, dim=-1)
        min_dist_km, min_idx = torch.min(dists_km, dim=0)

    tca_seconds = float(t_steps[min_idx].item() * 806.811)
    miss_km = float(min_dist_km.item())
    collision_prob = float(torch.exp(torch.tensor(-miss_km / 2.5)).item())

    return {
        "time_of_closest_approach_sec": round(tca_seconds, 2),
        "minimum_miss_distance_km": round(miss_km, 4),
        "conjunction_probability_score": round(collision_prob, 6),
        "maneuver_recommended": miss_km < 5.0,
    }

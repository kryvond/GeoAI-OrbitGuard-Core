from pathlib import Path
import torch
from fastapi.testclient import TestClient

from kryvond.pipeline.spectral_indices import GPUSpectralProcessor
from kryvond.models.terravision_vit import TerraVisionGeoModel
from kryvond.physics.orbitguard_pinn import OrbitGuardPINN
from api.server import app


def test_gpu_spectral_processor_bounds():
    proc = GPUSpectralProcessor()
    x = torch.rand(2, 12, 32, 32)
    indices = proc.compute_indices(x)
    for key in ["ndvi", "ndwi", "ndbi", "sar_ratio"]:
        assert indices[key].shape == (2, 1, 32, 32)
        assert torch.all(indices[key] >= -1.0) and torch.all(indices[key] <= 1.0)
    fused = proc(x)
    assert fused.shape == (2, 16, 32, 32)


def test_terravision_vit_forward_and_probabilities():
    model = TerraVisionGeoModel(in_bands=12, num_classes=5, base_dim=16).eval()
    x = torch.rand(2, 12, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out["logits"].shape == (2, 5, 64, 64)
    assert out["probs"].shape == (2, 5, 64, 64)
    prob_sums = out["probs"].sum(dim=1)
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5)


def test_orbitguard_pinn_exact_initial_conditions():
    pinn = OrbitGuardPINN(hidden_dim=64).eval()
    state0 = torch.tensor([[1.08, -0.12, 0.05, 0.02, 0.71, 0.65]], dtype=torch.float32)
    t_zero = torch.zeros((1, 1), dtype=torch.float32)
    r_at_zero, v_at_zero, _ = pinn.predict_state_and_acceleration(t_zero, state0)
    assert torch.allclose(r_at_zero, state0[:, :3], atol=1e-6)
    assert torch.allclose(v_at_zero, state0[:, 3:6], atol=1e-6)


def test_production_artifacts_and_fastapi_endpoints():
    root = Path(__file__).resolve().parent.parent
    assert (root / "artifacts" / "checkpoints" / "terravision_vit_best.pt").exists()
    assert (root / "artifacts" / "checkpoints" / "orbitguard_pinn_best.pt").exists()
    assert (root / "artifacts" / "onnx" / "terravision_vit.onnx").exists()
    assert (root / "artifacts" / "onnx" / "orbitguard_pinn.onnx").exists()
    assert (root / "artifacts" / "metrics_report.json").exists()

    client = TestClient(app)
    r_health = client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "operational"
    assert r_health.json()["geo_checkpoint_loaded"] is True
    assert r_health.json()["pinn_checkpoint_loaded"] is True

    r_seg = client.post("/v1/geo/segment-tile", json={"height": 64, "width": 64})
    assert r_seg.status_code == 200
    seg_data = r_seg.json()
    assert "class_coverage_fractions" in seg_data
    assert 0 <= seg_data["dominant_class_id"] <= 4

    r_orb = client.post("/v1/orbital/conjunction-screen", json={"horizon_minutes": 30.0})
    assert r_orb.status_code == 200
    assert "minimum_miss_distance_km" in r_orb.json()

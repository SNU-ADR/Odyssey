"""The hand-drawn SD edges come from the repository; no environment variable moves them."""
import importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_manual_edges_are_read_from_the_repository_whatever_the_environment(monkeypatch):
    monkeypatch.setenv("SDMAP_OUT", "/nonexistent")
    from odyssey_bridge import sd_route
    sd_route = importlib.reload(sd_route)
    assert Path(sd_route.SDMAP) == ROOT / "OdysseyBenchmark/data/sd_roadblock_map"
    edges = sd_route.manual_edges("sg-one-north")
    assert edges and all(i < 0 for i in edges)
    assert sd_route.replaced_sd_edges("sg-one-north")
    assert sd_route.manual_edges("us-ma-boston") == {} and sd_route.replaced_sd_edges("us-ma-boston") == set()

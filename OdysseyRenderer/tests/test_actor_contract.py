import pytest

from odyssey_renderer.omnire.actor_contract import validate_actor_manifest as validate_actor_manifest


def _asset(*, enabled=True, gaussian_count=10):
    return {"background": {"config": {"omnire_actor_manifest": {
        "schema_version": 1,
        "population_authority": "checkpoint_instance_id_map",
        "activation_policy": "finite_positive_gaussian_dynamic_actors",
        "actors": {"car": {
            "token": "car",
            "simulation_enabled": enabled,
            "gaussian_count": gaussian_count,
        }},
    }}}}


def test_v8_manifest_accepts_positive_gaussian_simulated_actor():
    assert validate_actor_manifest(_asset())["actors"]["car"]["gaussian_count"] == 10


def test_v8_manifest_allows_disabled_zero_gaussian_provenance_actor():
    assert validate_actor_manifest(
        _asset(enabled=False, gaussian_count=0)
    )["actors"]["car"]["simulation_enabled"] is False


@pytest.mark.parametrize("mutation", ["missing", "authority", "activation", "enabled_zero"])
def test_actor_manifest_fails_closed(mutation):
    asset = _asset()
    manifest = asset["background"]["config"]["omnire_actor_manifest"]
    if mutation == "missing":
        del asset["background"]["config"]["omnire_actor_manifest"]
    elif mutation == "authority":
        manifest["population_authority"] = "nuplan_gt"
    elif mutation == "activation":
        manifest["activation_policy"] = "all_nodes"
    else:
        manifest["actors"]["car"]["gaussian_count"] = 0
    with pytest.raises(ValueError, match="scene checkpoint|enabled actor"):
        validate_actor_manifest(asset)

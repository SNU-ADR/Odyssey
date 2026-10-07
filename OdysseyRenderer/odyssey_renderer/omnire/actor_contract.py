"""Validation of the checkpoint-pinned OmniRe actor population."""


def validate_actor_manifest(asset):
    """Return the actor manifest, or raise when it is missing or ambiguous."""
    config = asset.get("background", {}).get("config", {})
    manifest = config.get("omnire_actor_manifest")
    if not isinstance(manifest, dict):
        raise ValueError(
            "scene checkpoint requires background.config.omnire_actor_manifest"
        )
    if manifest.get("schema_version") != 1:
        raise ValueError(
            "scene checkpoint requires actor-manifest schema_version=1"
        )
    if manifest.get("population_authority") != "checkpoint_instance_id_map":
        raise ValueError(
            "scene checkpoint requires checkpoint_instance_id_map population authority"
        )
    if manifest.get("activation_policy") != "finite_positive_gaussian_dynamic_actors":
        raise ValueError(
            "scene checkpoint requires finite-positive-Gaussian actor activation"
        )
    actors = manifest.get("actors")
    if not isinstance(actors, dict):
        raise ValueError("actor manifest requires an actors mapping")
    for token, actor in actors.items():
        if not isinstance(actor, dict) or str(actor.get("token")) != str(token):
            raise ValueError(f"malformed actor manifest entry: {token}")
        enabled = bool(actor.get("simulation_enabled", False))
        gaussian_count = actor.get("gaussian_count")
        if enabled and (not isinstance(gaussian_count, int) or gaussian_count <= 0):
            raise ValueError(
                f"enabled actor {token} has no positive Gaussian population"
            )
    return manifest

"""Additive registration of OmniRe node types into MTGS MODEL_MAPPING.

Adding a key is additive; changing a function body is invasive. This
module only ever adds keys, and only ever keys that begin with "OmniRe".

setdefault alone is not enough. If a key already exists, setdefault returns the
INCUMBENT and says nothing -- so a name collision would surface later as a
wrong model class, which is worse than a crash. Every registration is therefore
verified afterwards.
"""
import logging

from odyssey_renderer.mtgs.mtgs import MODEL_MAPPING

logger = logging.getLogger(__name__)

PREFIX = "OmniRe"


def register_model(type_string, cls):
    """Add `type_string -> cls` if and only if it is free and it is ours."""
    if not type_string.startswith(PREFIX):
        raise ValueError(
            "refusing to register %r: OmniRe node types must be namespaced so "
            "they cannot collide with an MTGS asset's config.type" % type_string
        )
    incumbent = MODEL_MAPPING.setdefault(type_string, cls)
    if incumbent is not cls:
        logger.warning(
            "MODEL_MAPPING already had %s -> %r; keeping the incumbent",
            type_string, incumbent,
        )
    return MODEL_MAPPING[type_string]


def register_all():
    """Register every OmniRe node type. Called from OmniReRenderEngine.__init__.

    Background maps to the existing VanillaGaussianSplattingModel, rigid actors
    to RigidSubModel, and the sky is baked into a skybox node MTGS already
    understands. Only the deformable node needs a type of its own -- MTGS maps
    "DeformableSubModel" to None in every deployed variant, so that string can
    never be emitted.
    """
    from odyssey_renderer.omnire.blended_deformable import OmniReBlendedDeformableSubModel
    from odyssey_renderer.omnire.deformable_object import OmniReDeformableSubModel
    from odyssey_renderer.omnire.smpl_object import OmniReSMPLSubModel
    from odyssey_renderer.omnire.ground import OmniReGroundSubModel

    return {
        "OmniReDeformableSubModel":
            register_model("OmniReDeformableSubModel", OmniReDeformableSubModel),
        # A chunk-bank checkpoint stores one deformable actor as SEVERAL local
        # deformation networks plus per-frame blend weights, rather than the one
        # network OmniReDeformableSubModel holds. Distinct type because the
        # state_dict shape differs; the single-network assets keep the key above.
        "OmniReBlendedDeformableSubModel":
            register_model("OmniReBlendedDeformableSubModel", OmniReBlendedDeformableSubModel),
        # MTGS has no SMPL type at ALL -- not a poisoned one like
        # DeformableSubModel, simply absent -- so this is a new key rather than
        # a shadowed one. Still namespaced, for the same reason.
        "OmniReSMPLSubModel":
            register_model("OmniReSMPLSubModel", OmniReSMPLSubModel),
        # The road surface. Static in the world frame like the background, but it
        # cannot BE the background: drivestudio constrains its scale (frozen
        # thickness, clamped planar extent) and its rotation (yaw plus the LiDAR
        # surface normal), and VanillaModel would apply plain exp()/normalise().
        "OmniReGroundSubModel":
            register_model("OmniReGroundSubModel", OmniReGroundSubModel),
    }

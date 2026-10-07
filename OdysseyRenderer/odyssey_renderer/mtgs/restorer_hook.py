"""The render path's one image-restoration hook.

mtgs.py asks ``get_restorer()`` for the process restorer once per render and, if one is
installed, hands it the undistorted camera batch: ``restore_batch(images)`` takes a list of
HxWx3 uint8 BGR arrays and returns them restored, in the same order and at the same size.
``odyssey_renderer.omnire.restorer_client`` installs it (the Fixer worker or an HTTP restore
service) before the first render; with nothing installed the images are not restored.
"""

_RESTORER = None


def install(restorer):
    """Make ``restorer`` the process restorer and return it."""
    global _RESTORER
    _RESTORER = restorer
    return restorer


def get_restorer():
    """The installed restorer, or None."""
    return _RESTORER

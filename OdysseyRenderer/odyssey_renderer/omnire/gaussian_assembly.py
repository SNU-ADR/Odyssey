"""Borrowed, inference-only concatenation buffers for sequential camera rendering."""
import weakref

import torch


def _version(tensor):
    # Inference tensors have no counter: copying is the conservative fallback.
    try:
        return tensor._version
    except RuntimeError:
        return None


class GaussianAssembly:
    """Keep row order; skip only explicitly reusable, unmodified source slices.

    Outputs are borrowed until the next assemble for that key. Consumers must finish
    reading on the current CUDA stream before reuse. Stream changes allocate a new
    buffer. Reusable sources must not be mutated via .data or untracked CUDA writes;
    the renderer only opts in its immutable asset caches.
    """
    def __init__(self):
        self._entries = {}

    @torch.no_grad()
    def assemble(self, key, values, reusable=None):
        if reusable is None:
            reusable = [False] * len(values)
        assert len(values) == len(reusable)
        first = values[0]
        if any(t.dtype != first.dtype or t.device != first.device for t in values):
            # Preserve torch.cat's dtype promotion/error behavior.
            self._entries.pop(key, None)
            return torch.cat(values, dim=0)
        stream = torch.cuda.current_stream(first.device).cuda_stream if first.is_cuda else None
        layout = (stream, tuple((tuple(t.shape), t.dtype, t.device) for t in values))
        entry = self._entries.get(key)
        versions = [_version(t) for t in values]
        if entry is None or entry[1] != layout:
            output = torch.empty((sum(t.shape[0] for t in values), *first.shape[1:]),
                                 dtype=first.dtype, device=first.device)
            dirty = [True] * len(values)
        else:
            output, _, references, old_versions, output_version = entry
            intact = output_version is not None and _version(output) == output_version
            dirty = [not (intact and reuse and version is not None
                          and old_version == version and reference() is value)
                     for value, reference, version, old_version, reuse
                     in zip(values, references, versions, old_versions, reusable)]
        # Coalesce adjacent dirty slices into cat calls instead of launching one
        # copy kernel per actor. Static slices between them remain untouched.
        offset, index = 0, 0
        while index < len(values):
            if not dirty[index]:
                offset += values[index].shape[0]
                index += 1
                continue
            start, begin = index, offset
            while index < len(values) and dirty[index]:
                offset += values[index].shape[0]
                index += 1
            torch.cat(values[start:index], dim=0, out=output[begin:offset])
        self._entries[key] = (output, layout, [weakref.ref(t) for t in values],
                              versions, _version(output))
        return output

"""The Torch compute backend lives in `.torch_backend`; this package has no dispatch.

Historically this package selected between a Torch and an MLX backend by name.
MLX/MPS support was removed org-wide (2026-09-14, see brain/decisions.md);
Torch is now the only backend, so there is nothing left to resolve or select --
callers import `TorchBackend` from `.torch_backend` directly.
"""

from __future__ import annotations

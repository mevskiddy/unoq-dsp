"""Public errors shared by all importers and the hardware transport."""

class UnsupportedModel(ValueError):
    """The model cannot execute under the current DSP contract."""


class InvalidBundle(ValueError):
    """A compiled model is malformed, incompatible, or corrupt."""


class DSPError(RuntimeError):
    """Hardware execution failed. No inference is silently replayed on CPU."""

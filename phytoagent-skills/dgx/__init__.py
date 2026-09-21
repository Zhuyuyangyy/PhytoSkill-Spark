"""Access to the remote DGX Spark node: SSH transport and real vision inference."""

from dgx.client import DgxClient, load_credentials

__all__ = ["DgxClient", "load_credentials"]

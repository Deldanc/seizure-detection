"""Public ANFIS module.

The previous file contained a path-hardcoded membership-plus-dense-layer
prototype, not an adaptive fuzzy rule system. The canonical trained Sugeno
ANFIS implementation now lives in :mod:`src.models.ga_anfis` alongside the GA
pipeline and is re-exported here for model-facing imports.
"""

from src.models.ga_anfis import SugenoANFIS

ANFIS = SugenoANFIS

__all__ = ["ANFIS", "SugenoANFIS"]

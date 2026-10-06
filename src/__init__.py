"""OChem Helper — RDKit-based property, functional-group and retrosynthesis helper.

The torch-dependent generative models are optional; importing this package
must work with RDKit alone.
"""

__version__ = "0.2.0"
__author__ = "Christopher MacGregor / ROSE Corp"

from .core import functional_groups  # noqa: F401  (always available)

try:  # optional, torch-backed
    from .core import MolecularGraph
    from .models import MoleculeGenerator, PropertyPredictor
except ImportError:  # pragma: no cover - torch not installed
    MolecularGraph = MoleculeGenerator = PropertyPredictor = None  # type: ignore

__all__ = ["functional_groups", "MolecularGraph", "MoleculeGenerator", "PropertyPredictor"]

"""
Parvus - Quantum-Inspired Data Compression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A sophisticated data compression and similarity search system that uses
quantum-inspired techniques for efficient data storage and retrieval.

Installation:
    pip install parvus          # Basic installation
    pip install parvus[gpu]     # With GPU support
    pip install parvus[dev]     # For development

Basic usage:
    >>> from parvus import ParvusCompressor
    >>> system = ParvusCompressor()
    >>> embeddings = system.load_data_from_json('data.json')
    >>> system.compress(embeddings)
    >>> results, distances = system.query("search query")
"""

from .parvus import ParvusCompressor

__version__ = "0.1.0"
__author__ = "Your Name"
__license__ = "MIT"

__all__ = ["ParvusCompressor"]

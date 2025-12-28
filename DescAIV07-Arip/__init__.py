# __init__.py
"""
DESCAIV07 - CSV Column Description Generator

This package uses OpenRouter's AI models to generate dataset context
and column descriptions from CSV files.

Modules:
    ai_client.py         - AI API client setup
    table_context.py     - Functions for extracting and summarizing table samples
    column_descriptions.py - Functions for generating column-specific descriptions
    main.py              - CLI entry point
"""

from .ai_client import get_client
from .table_context import read_table_context, generate_table_description
from .column_descriptions import generate_column_descriptions
from .exportDescriptions import export_to_greenplum

__all__ = [
    "get_client",
    "read_table_context",
    "generate_table_description",
    "generate_column_descriptions",
    "export_to_greenplum"
]
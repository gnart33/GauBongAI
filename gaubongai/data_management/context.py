from typing import Dict, List, Any, Optional
import pandas as pd
from dataclasses import dataclass


@dataclass
class Context:
    """Container for context data and metadata."""

    data: pd.DataFrame
    metadata: Dict[str, Any]
    notes: List[str]


class DataContextManager:
    """Manages data contexts and their associated metadata and notes."""

    def __init__(self):
        """Initialize an empty DataContextManager."""
        self._contexts: Dict[str, Context] = {}

    def create_context(
        self, name: str, data: pd.DataFrame, metadata: Dict[str, Any]
    ) -> None:
        """Create a new data context."""
        if name in self._contexts:
            raise ValueError(f"Context '{name}' already exists")
        self._contexts[name] = Context(data=data, metadata=metadata, notes=[])

    def get_context(self, name: str) -> Context:
        """Get a context by name."""
        if name not in self._contexts:
            raise KeyError(f"Context '{name}' not found")
        return self._contexts[name]

    def update_context(
        self, name: str, data: pd.DataFrame, metadata: Dict[str, Any]
    ) -> None:
        """Update an existing context."""
        if name not in self._contexts:
            raise KeyError(f"Context '{name}' not found")
        self._contexts[name].data = data
        self._contexts[name].metadata = metadata

    def delete_context(self, name: str) -> None:
        """Delete a context and its associated notes."""
        if name not in self._contexts:
            raise KeyError(f"Context '{name}' not found")
        del self._contexts[name]

    def add_notes(self, name: str, note: str) -> None:
        """Add notes to a context."""
        if name not in self._contexts:
            raise KeyError(f"Context '{name}' not found")
        self._contexts[name].notes.append(note)

    def get_notes(self, name: str) -> List[str]:
        """Get notes for a context."""
        if name not in self._contexts:
            raise KeyError(f"Context '{name}' not found")
        return self._contexts[name].notes

    def list_contexts(self) -> List[str]:
        """List all available context names."""
        return list(self._contexts.keys())

from typing import Dict, List, Any, Optional
import pandas as pd


class DataContextManager:
    """Manager for handling data contexts and their associated metadata and notes."""

    def __init__(self):
        """Initialize an empty DataContextManager."""
        self.contexts: Dict[str, Dict[str, Any]] = {}
        self.notes: Dict[str, List[Dict[str, str]]] = {}

    def create_context(
        self, name: str, data: pd.DataFrame, metadata: Dict[str, Any]
    ) -> None:
        """Create a new data context with the given name, data, and metadata."""
        self.contexts[name] = {
            "data": data,
            "metadata": metadata,
            "notes": [],
        }
        self.notes[name] = []

    def get_context(self, name: str) -> Optional[Dict[str, Any]]:
        """Retrieve a data context by name."""
        return self.contexts.get(name)

    def update_context(
        self, name: str, data: pd.DataFrame, metadata: Dict[str, Any]
    ) -> None:
        """Update an existing data context with new data and metadata."""
        if name in self.contexts:
            self.contexts[name]["data"] = data
            self.contexts[name]["metadata"] = metadata

    def delete_context(self, name: str) -> None:
        """Delete a data context and its associated notes."""
        if name in self.contexts:
            del self.contexts[name]
        if name in self.notes:
            del self.notes[name]

    def add_notes(self, context_name: str, notes: List[Dict[str, str]]) -> None:
        """Add notes to a data context."""
        if context_name in self.contexts:
            self.notes[context_name] = notes

    def get_notes(self, context_name: str) -> Optional[List[Dict[str, str]]]:
        """Retrieve notes for a data context."""
        return self.notes.get(context_name)

    def list_contexts(self) -> List[str]:
        """List all available data context names."""
        return list(self.contexts.keys())

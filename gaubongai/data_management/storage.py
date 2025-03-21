from typing import Dict, Any, List, Optional
from .interfaces import DataContainer, DataCategory


class DataStorage:
    """Store for managing different types of data."""

    def __init__(self):
        # Store data by category and name
        self._storage: Dict[DataCategory, Dict[str, DataContainer]] = {
            category: {} for category in DataCategory
        }

    def store(self, name: str, data_info: DataContainer) -> None:
        """Store data with its category."""
        self._storage[data_info.category][name] = data_info

    def get(
        self, name: str, category: Optional[DataCategory] = None
    ) -> Optional[DataContainer]:
        """
        Retrieve data by name and optionally category.
        If category is not specified, search all categories.
        """
        if category:
            return self._storage[category].get(name)

        # Search all categories
        for category_storage in self._storage.values():
            if name in category_storage:
                return category_storage[name]
        return None

    def list_by_category(self, category: DataCategory) -> List[str]:
        """List all data names in a category."""
        return list(self._storage[category].keys())

    def list_all(self) -> Dict[DataCategory, List[str]]:
        """List all data names by category."""
        return {
            category: list(storage.keys())
            for category, storage in self._storage.items()
        }

    def delete(self, name: str, category: Optional[DataCategory] = None) -> bool:
        """
        Delete data by name and optionally category.
        Returns True if data was found and deleted.
        """
        if category:
            if name in self._storage[category]:
                del self._storage[category][name]
                return True
            return False

        # Search all categories
        for category_storage in self._storage.values():
            if name in category_storage:
                del category_storage[name]
                return True
        return False

    def get_metadata(
        self, name: str, category: Optional[DataCategory] = None
    ) -> Optional[Dict[str, Any]]:
        """Get metadata for stored data."""
        data_info = self.get(name, category)
        return data_info.metadata if data_info else None

    def update_metadata(
        self,
        name: str,
        metadata: Dict[str, Any],
        category: Optional[DataCategory] = None,
    ) -> bool:
        """
        Update metadata for stored data.
        Returns True if data was found and metadata was updated.
        """
        data_info = self.get(name, category)
        if data_info:
            data_info.metadata.update(metadata)
            return True
        return False

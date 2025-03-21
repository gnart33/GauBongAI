import pytest
from gaubongai.data_management.storage import DataStorage
from gaubongai.data_management.interfaces import DataCategory


def test_storage_initialization():
    """
    Test DataStorage initialization.

    This test verifies that:
    1. Storage is created with empty dictionaries for each category
    2. All DataCategory values are represented in storage
    """
    storage = DataStorage()

    # Check all categories are initialized
    for category in DataCategory:
        assert category in storage._storage
        assert isinstance(storage._storage[category], dict)
        assert len(storage._storage[category]) == 0


def test_store_and_get(sample_data_info):
    """
    Test storing and retrieving data.

    This test verifies:
    1. Data can be stored with a name
    2. Data can be retrieved by name
    3. Data can be retrieved by name and category
    4. Correct data is returned
    """
    storage = DataStorage()

    # Store data
    storage.store("test_data", sample_data_info)

    # Retrieve without category
    retrieved = storage.get("test_data")
    assert retrieved is not None
    assert retrieved.data.equals(sample_data_info.data)
    assert retrieved.metadata == sample_data_info.metadata

    # Retrieve with category
    retrieved = storage.get("test_data", DataCategory.TABULAR)
    assert retrieved is not None
    assert retrieved.data.equals(sample_data_info.data)


def test_get_nonexistent_data():
    """
    Test retrieving non-existent data.

    This test verifies:
    1. None is returned for non-existent data name
    2. None is returned for non-existent category
    """
    storage = DataStorage()

    assert storage.get("nonexistent") is None
    assert storage.get("nonexistent", DataCategory.TABULAR) is None


def test_list_by_category(sample_data_info, sample_text_data_info):
    """
    Test listing data by category.

    This test verifies:
    1. Correct data names are listed for each category
    2. Empty list is returned for categories with no data
    """
    storage = DataStorage()

    # Store data in different categories
    storage.store("tabular_data", sample_data_info)
    storage.store("text_data", sample_text_data_info)

    # Check listings
    assert "tabular_data" in storage.list_by_category(DataCategory.TABULAR)
    assert "text_data" in storage.list_by_category(DataCategory.TEXT)
    assert len(storage.list_by_category(DataCategory.IMAGE)) == 0


def test_list_all(sample_data_info, sample_text_data_info):
    """
    Test listing all data.

    This test verifies:
    1. All stored data is listed correctly by category
    2. Empty categories are included in results
    """
    storage = DataStorage()

    # Store data
    storage.store("tabular_data", sample_data_info)
    storage.store("text_data", sample_text_data_info)

    all_data = storage.list_all()

    # Check all categories are present
    assert set(all_data.keys()) == set(DataCategory)

    # Check data is listed in correct categories
    assert "tabular_data" in all_data[DataCategory.TABULAR]
    assert "text_data" in all_data[DataCategory.TEXT]
    assert len(all_data[DataCategory.IMAGE]) == 0


def test_delete(sample_data_info):
    """
    Test deleting data.

    This test verifies:
    1. Data can be deleted by name
    2. Data can be deleted by name and category
    3. Deleting non-existent data returns False
    4. Deleted data cannot be retrieved
    """
    storage = DataStorage()

    # Store data
    storage.store("test_data", sample_data_info)

    # Delete with category
    assert storage.delete("test_data", DataCategory.TABULAR)
    assert storage.get("test_data", DataCategory.TABULAR) is None

    # Store again and delete without category
    storage.store("test_data", sample_data_info)
    assert storage.delete("test_data")
    assert storage.get("test_data") is None

    # Try to delete non-existent data
    assert not storage.delete("nonexistent")


def test_metadata_operations(sample_data_info):
    """
    Test metadata operations.

    This test verifies:
    1. Metadata can be retrieved
    2. Metadata can be updated
    3. Metadata operations on non-existent data return None/False
    """
    storage = DataStorage()

    # Store data
    storage.store("test_data", sample_data_info)

    # Get metadata
    metadata = storage.get_metadata("test_data")
    assert metadata == sample_data_info.metadata

    # Update metadata
    new_metadata = {"new_field": "value"}
    assert storage.update_metadata("test_data", new_metadata)

    # Check updated metadata
    updated_metadata = storage.get_metadata("test_data")
    assert "new_field" in updated_metadata
    assert updated_metadata["new_field"] == "value"

    # Check operations on non-existent data
    assert storage.get_metadata("nonexistent") is None
    assert not storage.update_metadata("nonexistent", {})

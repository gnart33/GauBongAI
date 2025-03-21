import pytest
import pandas as pd
from gaubongai.data_management.context import DataContextManager
from gaubongai.data_management.interfaces import DataCategory


@pytest.fixture
def context_manager():
    """Create a fresh DataContextManager instance for each test."""
    return DataContextManager()


@pytest.fixture
def sample_data():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})


@pytest.fixture
def sample_metadata():
    """Create sample metadata for testing."""
    return {"rows": 3, "columns": ["A", "B"], "description": "Test data"}


def test_create_context(context_manager, sample_data, sample_metadata):
    """Test creating a new context."""
    context_name = "test_context"
    context_manager.create_context(context_name, sample_data, sample_metadata)

    # Verify context was created
    context = context_manager.get_context(context_name)
    assert context is not None
    pd.testing.assert_frame_equal(context.data, sample_data)
    assert context.metadata == sample_metadata


def test_create_duplicate_context(context_manager, sample_data, sample_metadata):
    """Test that creating a duplicate context raises an error."""
    context_name = "test_context"
    context_manager.create_context(context_name, sample_data, sample_metadata)

    with pytest.raises(ValueError, match=f"Context '{context_name}' already exists"):
        context_manager.create_context(context_name, sample_data, sample_metadata)


def test_get_nonexistent_context(context_manager):
    """Test getting a context that doesn't exist."""
    with pytest.raises(KeyError, match="Context 'nonexistent' not found"):
        context_manager.get_context("nonexistent")


def test_update_context(context_manager, sample_data, sample_metadata):
    """Test updating an existing context."""
    context_name = "test_context"
    context_manager.create_context(context_name, sample_data, sample_metadata)

    # Create updated data and metadata
    updated_data = pd.DataFrame({"A": [4, 5, 6], "B": ["a", "b", "c"]})
    updated_metadata = {**sample_metadata, "description": "Updated test data"}

    # Update context
    context_manager.update_context(context_name, updated_data, updated_metadata)

    # Verify update
    context = context_manager.get_context(context_name)
    pd.testing.assert_frame_equal(context.data, updated_data)
    assert context.metadata == updated_metadata


def test_update_nonexistent_context(context_manager, sample_data, sample_metadata):
    """Test updating a context that doesn't exist."""
    with pytest.raises(KeyError, match="Context 'nonexistent' not found"):
        context_manager.update_context("nonexistent", sample_data, sample_metadata)


def test_delete_context(context_manager, sample_data, sample_metadata):
    """Test deleting a context."""
    context_name = "test_context"
    context_manager.create_context(context_name, sample_data, sample_metadata)

    # Add some notes
    context_manager.add_notes(context_name, "Test note")

    # Delete context
    context_manager.delete_context(context_name)

    # Verify context and notes are deleted
    with pytest.raises(KeyError):
        context_manager.get_context(context_name)
    with pytest.raises(KeyError):
        context_manager.get_notes(context_name)


def test_delete_nonexistent_context(context_manager):
    """Test deleting a context that doesn't exist."""
    with pytest.raises(KeyError, match="Context 'nonexistent' not found"):
        context_manager.delete_context("nonexistent")


def test_add_and_get_notes(context_manager, sample_data, sample_metadata):
    """Test adding and retrieving notes for a context."""
    context_name = "test_context"
    context_manager.create_context(context_name, sample_data, sample_metadata)

    # Add notes
    notes = ["First note", "Second note"]
    for note in notes:
        context_manager.add_notes(context_name, note)

    # Verify notes
    retrieved_notes = context_manager.get_notes(context_name)
    assert retrieved_notes == notes


def test_add_notes_to_nonexistent_context(context_manager):
    """Test adding notes to a context that doesn't exist."""
    with pytest.raises(KeyError, match="Context 'nonexistent' not found"):
        context_manager.add_notes("nonexistent", "Test note")


def test_get_notes_from_nonexistent_context(context_manager):
    """Test getting notes from a context that doesn't exist."""
    with pytest.raises(KeyError, match="Context 'nonexistent' not found"):
        context_manager.get_notes("nonexistent")


def test_list_contexts(context_manager, sample_data, sample_metadata):
    """Test listing all available contexts."""
    # Create multiple contexts
    contexts = ["context1", "context2", "context3"]
    for name in contexts:
        context_manager.create_context(name, sample_data, sample_metadata)

    # Verify list
    available_contexts = context_manager.list_contexts()
    assert set(available_contexts) == set(contexts)


def test_list_empty_contexts(context_manager):
    """Test listing contexts when none exist."""
    assert context_manager.list_contexts() == []

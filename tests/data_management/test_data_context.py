import pytest
import pandas as pd
from gauai.core.data_management.data_context import DataContextManager


def test_init():
    """Test DataContextManager initialization."""
    manager = DataContextManager()
    assert manager.contexts == {}
    assert manager.notes == {}


def test_create_context(sample_heart_data, heart_data_metadata):
    """Test creating a new data context for heart attack data."""
    manager = DataContextManager()
    context_name = "heart_analysis"

    manager.create_context(
        name=context_name, data=sample_heart_data, metadata=heart_data_metadata
    )

    assert context_name in manager.contexts
    context = manager.contexts[context_name]

    pd.testing.assert_frame_equal(context["data"], sample_heart_data)
    assert context["metadata"] == heart_data_metadata
    assert context["notes"] == []


def test_add_notes(sample_heart_data, heart_data_metadata, sample_heart_notes):
    """Test adding notes to heart attack data context."""
    manager = DataContextManager()
    context_name = "heart_analysis"

    # Create context first
    manager.create_context(
        name=context_name, data=sample_heart_data, metadata=heart_data_metadata
    )

    # Add notes
    manager.add_notes(context_name, sample_heart_notes)

    # Verify notes were added
    assert context_name in manager.notes
    assert manager.notes[context_name] == sample_heart_notes


def test_get_context(sample_heart_data, heart_data_metadata):
    """Test retrieving heart attack data context."""
    manager = DataContextManager()
    context_name = "heart_analysis"

    # Create and store context
    context = {"data": sample_heart_data, "metadata": heart_data_metadata, "notes": []}
    manager.contexts[context_name] = context

    # Retrieve and verify context
    retrieved_context = manager.get_context(context_name)
    assert retrieved_context == context


def test_get_nonexistent_context():
    """Test retrieving non-existent context."""
    manager = DataContextManager()
    assert manager.get_context("nonexistent") is None


def test_get_notes(sample_heart_notes):
    """Test retrieving notes for heart attack data context."""
    manager = DataContextManager()
    context_name = "heart_analysis"

    # Store notes
    manager.notes[context_name] = sample_heart_notes

    # Retrieve and verify notes
    retrieved_notes = manager.get_notes(context_name)
    assert retrieved_notes == sample_heart_notes


def test_list_contexts(sample_heart_data, heart_data_metadata):
    """Test listing available contexts."""
    manager = DataContextManager()
    context_names = ["heart_train", "heart_test", "heart_validation"]

    # Create multiple contexts
    for name in context_names:
        manager.create_context(
            name=name, data=sample_heart_data, metadata=heart_data_metadata
        )

    available_contexts = manager.list_contexts()
    assert set(available_contexts) == set(context_names)


def test_update_context(sample_heart_data, heart_data_metadata):
    """Test updating heart attack data context."""
    manager = DataContextManager()
    context_name = "heart_analysis"

    # Create initial context
    manager.create_context(
        name=context_name, data=sample_heart_data, metadata=heart_data_metadata
    )

    # Create updated data and metadata
    updated_data = sample_heart_data.copy()
    updated_data["NewColumn"] = 1
    updated_metadata = heart_data_metadata.copy()
    updated_metadata["columns"].append("NewColumn")

    # Update context
    manager.update_context(
        name=context_name, data=updated_data, metadata=updated_metadata
    )

    # Verify updates
    updated_context = manager.get_context(context_name)
    pd.testing.assert_frame_equal(updated_context["data"], updated_data)
    assert updated_context["metadata"] == updated_metadata


def test_delete_context():
    """Test deleting a context."""
    manager = DataContextManager()
    context_name = "heart_analysis"

    # Create context and notes
    manager.contexts[context_name] = {
        "data": pd.DataFrame(),
        "metadata": {},
        "notes": [],
    }
    manager.notes[context_name] = []

    # Delete context
    manager.delete_context(context_name)

    # Verify deletion
    assert context_name not in manager.contexts
    assert context_name not in manager.notes

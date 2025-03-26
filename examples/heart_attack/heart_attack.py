from gaubongai.data_management import DataContainer, DataProcessor
from gaubongai.data_management.loaders import LoaderManager, PandasCSVLoader

from gaubongai.data_management.transformers import (
    PandasDfTransformer,
    TransformerManager,
)
from pathlib import Path


def load_without_transformers():
    file_path = Path("examples/heart_attack/heart_attack_dataset.csv")
    loader = PandasCSVLoader()
    data_container = loader.load(file_path)

    print(data_container.metadata.get("dtypes").keys())
    print(data_container.data.shape)


def load_with_transformers():
    file_path = Path("examples/heart_attack/heart_attack_dataset.csv")

    loader = PandasCSVLoader()

    column_specs = {
        "Age": {"dtype": "int32", "rename": "age"},
        "BMI": {"dtype": "float64"},
        "Gender": {
            "rename": "gender",
            "dtype": "category",
            "na_values": ["unknown", "n/a"],
            "fillna": "other",
            "transform": str.lower,
        },
        "Cholesterol": {"remove": True},
    }

    transformer = PandasDfTransformer(column_specs=column_specs)

    data_processor = DataProcessor(loader=loader, transformers=[transformer])

    data_container = data_processor.process_file(file_path)

    new_metadata = data_container.metadata.copy()
    new_metadata["notes"] = new_metadata.get("notes", []) + [
        "removed Cholesterol column for testing"
    ]

    data_container = DataContainer(data=data_container.data, metadata=new_metadata)
    print(data_container.metadata.get("notes"))
    print(data_container.metadata)


def main():
    load_without_transformers()
    print("-" * 100)
    load_with_transformers()


if __name__ == "__main__":
    main()

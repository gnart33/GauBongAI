from gaubongai.data_management import DataProcessor
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


def load_with_transformers():
    file_path = Path("examples/heart_attack/heart_attack_dataset.csv")

    loader = PandasCSVLoader()

    column_specs = {
        "Age": {"dtype": "int32", "rename": "age"},
        "BMI": {"dtype": "float64"},
        "Gender": {
            "rename": "gender",
            "dtype": "category",
            "transform": str.lower,
            "na_values": ["unknown", "n/a"],
            "fillna": "other",
        },
    }

    transformer = PandasDfTransformer(column_specs=column_specs)

    data_processor = DataProcessor(loader=loader, transformers=[transformer])

    data_container = data_processor.process_file(file_path)

    print(data_container.metadata.get("transformation_history"))


def main():
    load_without_transformers()
    print("-" * 100)
    load_with_transformers()


if __name__ == "__main__":
    main()

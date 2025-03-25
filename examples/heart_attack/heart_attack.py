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

    print(data_container.data.head())


def load_with_transformers():
    file_path = Path("examples/heart_attack/heart_attack_dataset.csv")

    # loader_manager = LoaderManager()

    loader = PandasCSVLoader()

    column_specs = {
        # Simple conversion
        "Age": {"dtype": "float64"},
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

    # print(data_container.data.head())
    print(data_container.data.gender.dtype)
    print(data_container.data.BMI.dtype)


def check_loaders():
    # loader_manager = LoaderManager()
    plugin_manager = LoaderManager()
    print(plugin_manager._plugin_registry)


def main():
    # check_loaders()
    # load_without_transformers()
    load_with_transformers()


if __name__ == "__main__":
    main()

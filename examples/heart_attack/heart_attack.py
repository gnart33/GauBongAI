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


# def load_with_transformers():
#     file_path = Path("examples/heart_attack/heart_attack_dataset.csv")

#     loader_manager = LoaderManager()

#     loader = loader_manager.get_plugin("pandas_csv")

#     transformer = PandasDfTransformer(
#         rename_columns={
#             "Age": "age",
#         }
#     )

#     data_processor = DataProcessor(loader=loader, transformers=[transformer])

#     data_container = data_processor.process_file(file_path)

#     print(data_container.data.head())


def check_loaders():
    # loader_manager = LoaderManager()
    plugin_manager = LoaderManager()
    print(plugin_manager._plugin_registry)


def main():
    # check_loaders()
    load_without_transformers()


if __name__ == "__main__":
    main()

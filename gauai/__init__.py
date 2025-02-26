import polars as pl
from .dataframe import DataFrame


def read_csv(filepath: str) -> DataFrame:
    data = pl.read_csv(filepath)
    return DataFrame(data)


__all__ = [
    "DataFrame",
]

import polars as pl
from collections import ChainMap
from typing import Union, Dict, Any, List


def add_required_columns(df: pl.DataFrame, required_columns: Union[Dict[str, Any], List[Dict[str, Any]]]) -> pl.DataFrame:
    """Ensures the DataFrame contains all required columns.
    If a column is missing, it is added with a specified default value.

    Args:
        df (pl.DataFrame): The input Polars DataFrame.
        required_columns (Union[Dict[str, Any], List[Dict[str, Any]]]):
            - A dictionary of {column_name: default_value}.
            - A list of such dictionaries, which will be merged.

    Returns:
        pl.DataFrame: The DataFrame with all required columns added.
    """
    # Merge dictionaries if required_columns is a list
    if isinstance(required_columns, list):
        required_columns_all = dict(ChainMap(*required_columns)) if required_columns else {}
    else:
        required_columns_all = required_columns

    # Collect new columns that need to be added
    missing_cols = [
        pl.lit(default_value).alias(col)
        for col, default_value in required_columns_all.items() if col not in df.columns
    ]

    # Add only if there are missing columns
    return df.with_columns(missing_cols) if missing_cols else df


def handle_empty(df: pl.DataFrame, columns: list) -> pl.DataFrame:
    """_summary_

    Args:
        df (pl.DataFrame): _description_
        columns (list): _description_

    Returns:
        pl.DataFrame: _description_
    """
    return df.with_columns(
        [pl.col(c).fill_null("NON RENSEIGNE") for c in columns]
    ).with_columns(
        [
            pl.col(c).map_elements(
                lambda x: "NON RENSEIGNE" if x == "" else x, return_dtype=pl.String
            )
            for c in columns
        ]
    )

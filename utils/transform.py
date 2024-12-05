import polars as pl


def add_required_columns(df: pl.DataFrame, required_columns: dict) -> pl.DataFrame:
    """_summary_

    Args:
        df (pl.DataFrame): _description_
        required_columns (dict): _description_

    Returns:
        pl.DataFrame: _description_
    """
    for col, default_value in required_columns.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(default_value).alias(col))
    return df


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

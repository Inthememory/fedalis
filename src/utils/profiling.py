import polars as pl

from typing import List

def agg_kpis(dataset_lst:List[pl.DataFrame], cols:list)->pl.DataFrame:
    """_summary_

    Args:
        dataset_lst (List[pl.DataFrame]): _description_
        cols (list): _description_

    Returns:
        pl.DataFrame: _description_
    """
    return (
        pl.concat(dataset_lst, how="vertical")
        .with_columns(
            ms_turnover=pl.col("turnover")
            / pl.col("turnover").sum().over(cols)
        )
        .with_columns(
            ms_nb_product_ids=pl.col("nb_product_ids")
            / pl.col("nb_product_ids")
            .sum()
            .over(cols)
        )
        .sort(["retailer"] + cols)
    )

def check_unicity(
    df: pl.DataFrame, subset_col: list, cols_to_check: list
) -> tuple[list, pl.DataFrame]:
    """_summary_

    Args:
        df (pl.DataFrame): _description_
        subset_col (list): _description_
        cols_to_check (list): _description_

    Returns:
        tuple[list, pl.Dataframe]: _description_
    """
    if set(subset_col).issubset(set(df.columns)) and set(cols_to_check).issubset(
        set(df.columns)
    ):
        res = (
            df
            .select(cols_to_check + subset_col)
            .unique()
            .group_by(subset_col)
            .len()
            .filter(pl.col("len") > 1)
        )
        items_lst = res.select(subset_col).unique().to_series().to_list()
        items_details = (
            df.select(
                subset_col
                + [
                    c
                    for c in df.columns
                    if c.startswith("level") and c not in cols_to_check
                ]
                + cols_to_check
            )
            .unique()
            .filter(pl.col(subset_col).is_in(items_lst))
            .sort(subset_col)
        )

        return items_lst, items_details
    else:
        return [], None


def compute_kpis(df: pl.DataFrame, cols: list) -> pl.DataFrame:
    """_summary_

    Args:
        df (pl.DataFrame): _description_
        cols (list): _description_

    Returns:
        pl.DataFrame: _description_
    """
    if "product_id" not in df.columns:
        df = df.with_columns(pl.lit(None).alias("product_id"))
    else :
        pass
    return (
        df.group_by(cols)
        .agg(
            turnover=pl.col("turnover").sum(),
            nb_product_ids=pl.col("product_id").n_unique(),
            nb_product_codes=pl.col("product_code").n_unique(),
        )
        .with_columns(pl.col('year').cast(pl.String))
        .with_columns(pl.col('turnover').cast(pl.Float64))
        .with_columns(pl.col('nb_product_ids').cast(pl.Int32))
        .with_columns(pl.col('nb_product_codes').cast(pl.Int32))
        .sort(cols)
    )


def incremental_sublists(lst:list)->list:
    """_summary_

    Args:
        lst (list): _description_

    Returns:
        list: _description_
    """
    return [lst[:i] for i in range(1, len(lst) + 1)]
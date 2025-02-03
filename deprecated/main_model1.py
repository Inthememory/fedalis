import os
import polars as pl
import yaml
from loguru import logger
import xlsxwriter
from itertools import combinations
from functools import reduce

from utils.transform import add_required_columns, handle_empty
from utils.profiling import check_unicity, compute_kpis

from data import (
    coup_de_pates,
    ds_restauration,
    ducreux,
    even,
    metro,
    pomona,
    pro_a_pro,
    relais_dor,
    sysco,
)

pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_width_chars(0)

if __name__ == "__main__":

    # Load the configuration file
    with open("config.yml", "r", encoding="utf-8") as file:
        configuration = yaml.safe_load(file)

    logger.add("data/log/logs.txt", mode="w", level="INFO")

    # Set parameters
    retailers = [
        ("coup_de_pates", coup_de_pates),
        ("ds_restauration", ds_restauration),
        ("ducreux", ducreux),
        ("even", even),
        ("metro", metro),
        ("pomona", pomona),
        ("pro_a_pro", pro_a_pro),
        ("relais_dor", relais_dor),
        ("sysco", sysco),
    ]

    required_columns = configuration[
        "required_columns"
    ]  # Columns that must exist in the DataFrame

    selected_columns = (
        ["product_id", "product_code", "product_name", "brand_name"]
        + list(required_columns.keys())
        + ["level_1_standard", "level_2_standard"]
        + ["department", "year", "month", "turnover"]
    )

    col_strip = [
        "product_id",
        "product_code",
        "product_name",
        "brand_name",
        "year",
    ] + list(required_columns.keys())

    # --------
    # Retailer
    # --------

    l1_dataset_lst, l1Dep_dataset_lst, l2_dataset_lst, l2Dep_dataset_lst = (
        [],
        [],
        [],
        [],
    )

    for retailer_name, retailer_dataset in retailers:
        logger.info(
            f"------------- Datasets to be processed : {retailer_name.upper()} -------------"
        )

        # Configure file to write logs
        wb = xlsxwriter.Workbook(f"data/log/model_1/log_{retailer_name}.xlsx")

        ## Step 1: retailer_dataset

        # Rename columns
        retailer_dataset = retailer_dataset.rename(
            configuration[retailer_name]["rename"], strict=False
        )

        # Add required columns
        retailer_dataset = add_required_columns(retailer_dataset, required_columns)

        # Strip string columns
        retailer_dataset = retailer_dataset.with_columns(
            [
                pl.col(c).cast(pl.String).str.strip_chars()
                for c in col_strip
                if c in retailer_dataset.columns
            ]
        )

        # Fill null and ""
        retailer_dataset = handle_empty(retailer_dataset, list(required_columns.keys()))

        # Cast Turnover as Float
        if retailer_dataset.select("turnover").dtypes[0] == pl.String:
            retailer_dataset = retailer_dataset.with_columns(
                pl.col("turnover").str.replace(",", ".").cast(pl.Float64)
            )

        # Clean department column
        if "zip_code" in retailer_dataset.columns:
            retailer_dataset = retailer_dataset.with_columns(
                pl.col("zip_code").str.head(2).alias("department")
            )
        retailer_dataset = retailer_dataset.with_columns(
            pl.col("department").cast(pl.String).str.replace(r"^0|FR/", "")
        ).with_columns(pl.col("department").str.zfill(2))

        # Remove rows without product_id
        if "product_id" in retailer_dataset.columns:
            retailer_dataset = (
                retailer_dataset.filter(pl.col("product_id").is_not_null())
                .filter(pl.col("product_id") != "")
                .filter(pl.col("product_id") != "0")
                .filter(pl.col("product_id") != "#")
                .filter(pl.col("product_id") != "Non Defini")
            )

        ## Step 2 : retailer_map

        retailer_map = pl.read_excel(
            "data/mapping_model1.xlsx", sheet_name=retailer_name
        ).rename(configuration[retailer_name]["rename"], strict=False)

        # Add required columns
        retailer_map = add_required_columns(retailer_map, required_columns)

        # Fill null and ""
        retailer_map = handle_empty(retailer_map, list(required_columns.keys()))

        ## Step 3 : Mapping

        dataset = retailer_dataset.join(
            retailer_map, on=list(required_columns.keys()), how="left"
        ).select(
            [
                c
                for c in selected_columns
                if c in retailer_dataset.columns + retailer_map.columns
            ]
        )

        if "product_id" in retailer_dataset.columns:
            dataset_for_comparison = (
                dataset.with_columns(
                    pl.concat_str(list(required_columns.keys()), separator=">").alias(
                        "classification"
                    )
                )
                .select(
                    "product_id",
                    "product_name",
                    "classification",
                    "level_1_standard",
                    "level_2_standard",
                )
                .unique()
                .rename(lambda col: f"{col}_{retailer_name}")
                .rename({f"product_id_{retailer_name}": "product_id"})
            )
            dataset_for_comparison.write_parquet(f"data/tmp/{retailer_name}.parquet")

        logger.info(
            f"retailer_dataset shape/matched rows : {retailer_dataset.shape} / {dataset.filter(pl.col('level_1_standard').is_not_null()).shape})"
        )

        ## Step 4: Unmatched levels

        unmatched_levels = (
            dataset.filter(
                (pl.col("level_1_standard").is_null())
                | (pl.col("level_2_standard").is_null())
            )
            .select(
                list(required_columns.keys()) + ["level_1_standard", "level_2_standard"]
            )
            .unique()
            .sort(
                list(required_columns.keys()) + ["level_1_standard", "level_2_standard"]
            )
        )
        if unmatched_levels.is_empty():
            pass
        else:
            unmatched_levels.write_excel(workbook=wb, worksheet="unmatched_levels")

        ## Step 5: Unicity checks

        cols_to_check_lst = [["level_1_standard", "level_2_standard"], ["product_id"]]
        subset_cols_lst = [["product_id"], ["product_code"]]
        for i, cols_to_check in enumerate(cols_to_check_lst):
            for subset_cols in subset_cols_lst:
                if cols_to_check != subset_cols:
                    unicity_errors_lst, unicity_errors_details = check_unicity(
                        dataset, subset_cols, cols_to_check
                    )
                    if len(unicity_errors_lst) > 0:
                        unicity_errors_details.write_excel(
                            workbook=wb,
                            worksheet=f"unicity_errors_{', '.join(subset_cols)}_{i}",
                        )

        ## Step 6: KPIs computation
        logger.info(f"Turnover type : {configuration[retailer_name]['turnover_type']}")

        l1_kpis = compute_kpis(dataset, ["year", "level_1_standard"])
        l1_kpis = l1_kpis.with_columns(pl.lit(retailer_name).alias("retailer"))
        l1_dataset_lst.append(l1_kpis)
        l1_kpis.write_excel(workbook=wb, worksheet=f"kpis_l1")

        l1Dep_kpis = compute_kpis(dataset, ["year", "level_1_standard", "department"])
        l1Dep_kpis = l1Dep_kpis.with_columns(pl.lit(retailer_name).alias("retailer"))
        l1Dep_dataset_lst.append(l1Dep_kpis)
        l1Dep_kpis.write_excel(workbook=wb, worksheet=f"kpis_l1Dep")

        l2_kpis = compute_kpis(
            dataset, ["year", "level_1_standard", "level_2_standard"]
        )
        l2_kpis = l2_kpis.with_columns(pl.lit(retailer_name).alias("retailer"))
        l2_dataset_lst.append(l2_kpis)
        l2_kpis.write_excel(workbook=wb, worksheet=f"kpis_l2")

        l2Dep_kpis = compute_kpis(
            dataset, ["year", "level_1_standard", "level_2_standard", "department"]
        )
        l2Dep_kpis = l2Dep_kpis.with_columns(pl.lit(retailer_name).alias("retailer"))
        l2Dep_dataset_lst.append(l2Dep_kpis)
        l2Dep_kpis.write_excel(workbook=wb, worksheet=f"kpis_l2Dep")

        wb.close()

    # ------
    # Global
    # ------

    wb_global = xlsxwriter.Workbook(
        f"data/log/model_1/log_global.xlsx", options={"nan_inf_to_errors": True}
    )

    # Market Share
    logger.info(f"\n------------- Market Share -------------")
    ms_l1 = (
        pl.concat(l1_dataset_lst)
        .with_columns(
            ms_turnover=pl.col("turnover")
            / pl.col("turnover").sum().over("year", "level_1_standard")
        )
        .with_columns(
            ms_nb_product_ids=pl.col("nb_product_ids")
            / pl.col("nb_product_ids").sum().over("year", "level_1_standard")
        )
        .sort("retailer", "year", "level_1_standard")
    )
    if ms_l1.is_empty():
        pass
    else:
        ms_l1.write_excel(workbook=wb_global, worksheet="ms_l1")

    ms_l1Dep = (
        pl.concat(l1Dep_dataset_lst)
        .with_columns(
            ms_turnover=pl.col("turnover")
            / pl.col("turnover").sum().over("year", "level_1_standard", "department")
        )
        .with_columns(
            ms_nb_product_ids=pl.col("nb_product_ids")
            / pl.col("nb_product_ids")
            .sum()
            .over("year", "level_1_standard", "department")
        )
        .sort("retailer", "year", "level_1_standard", "department")
    )
    if ms_l1Dep.is_empty():
        pass
    else:
        ms_l1Dep.write_excel(workbook=wb_global, worksheet="ms_l1Dep")

    ms_l2 = (
        pl.concat(l2_dataset_lst)
        .with_columns(
            ms_turnover=pl.col("turnover")
            / pl.col("turnover")
            .sum()
            .over("year", "level_1_standard", "level_2_standard")
        )
        .with_columns(
            ms_nb_product_ids=pl.col("nb_product_ids")
            / pl.col("nb_product_ids")
            .sum()
            .over("year", "level_1_standard", "level_2_standard")
        )
        .sort("retailer", "year", "level_1_standard", "level_2_standard")
    )
    if ms_l2.is_empty():
        pass
    else:
        ms_l2.write_excel(workbook=wb_global, worksheet="ms_l2")

    ms_l2Dep = (
        pl.concat(l2Dep_dataset_lst)
        .with_columns(
            ms_turnover=pl.col("turnover")
            / pl.col("turnover")
            .sum()
            .over("year", "level_1_standard", "level_2_standard", "department")
        )
        .with_columns(
            ms_nb_product_ids=pl.col("nb_product_ids")
            / pl.col("nb_product_ids")
            .sum()
            .over("year", "level_1_standard", "level_2_standard", "department")
        )
        .sort("retailer", "year", "level_1_standard", "level_2_standard", "department")
    )
    if ms_l2Dep.is_empty():
        pass
    else:
        ms_l2Dep.write_excel(workbook=wb_global, worksheet="ms_l2Dep")

    ## Misclassification
    dataset_concat = pl.concat(
        [
            pl.read_parquet(f"data/tmp/{filename}")
            for filename in os.listdir("data/tmp")
        ],
        how="align",
    )

    columns_level1 = [
        c for c in dataset_concat.columns if c.startswith("level_1_standard_")
    ]
    columns_level2 = [
        c for c in dataset_concat.columns if c.startswith("level_2_standard_")
    ]

    # Filter rows with a unique retailer
    consider_empty_col_list = columns_level1

    filtered_dataset_concat = (
        dataset_concat.with_columns(
            [
                pl.col(c).is_null().cast(pl.Int8).alias(f"{c}_bool")
                for c in consider_empty_col_list
            ]
        )
        .with_columns(
            sum=pl.sum_horizontal([f"{c}_bool" for c in consider_empty_col_list])
        )
        .filter(pl.col("sum") < 6)
        .drop([f"{c}_bool" for c in consider_empty_col_list] + ["sum"])
    )

    # Filter rows with different classification between retailers
    filtered_dataset_concat = (
        filtered_dataset_concat.with_columns(
            pl.concat_list(columns_level1).alias("level1_lst")
        )
        .with_columns(pl.concat_list(columns_level2).alias("level2_lst"))
        .with_columns(
            pl.col("level1_lst").map_elements(
                lambda x: list(set([e for e in x if e is not None]))
            )
        )
        .with_columns(
            pl.col("level2_lst").map_elements(
                lambda x: list(set([e for e in x if e is not None]))
            )
        )
        .with_columns(pl.col("level1_lst").map_elements(lambda x: len(x)))
        .with_columns(pl.col("level2_lst").map_elements(lambda x: len(x)))
        .filter((pl.col("level1_lst") > 1) | (pl.col("level2_lst") > 1))
        .drop(["level1_lst", "level2_lst"])
        .unique()
    )

    if filtered_dataset_concat.is_empty():
        pass
    else:
        filtered_dataset_concat.write_excel(
            workbook=wb_global, worksheet="Misclassification"
        )

    wb_global.close()

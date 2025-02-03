import argparse
from loguru import logger
import os
import polars as pl
import xlsxwriter
import yaml

from data import BRONZE_PATH, SILVER_PATH, GOLD_PATH

from src.utils.transform import add_required_columns, handle_empty
from src.utils.profiling import (
    agg_kpis,
    check_unicity,
    compute_kpis,
    incremental_sublists,
)

from src.models.dataset import Dataset

if __name__ == "__main__":
    # Add argparse for the command line:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id", type=int, required=True, help="Id of the model to use."
    )
    args = parser.parse_args()

    # Load the configuration file
    with open("config.yml", "r", encoding="utf-8") as file:
        configuration = yaml.safe_load(file)

    logger.add(f"data/log/model_{args.model_id}/logs.txt", mode="w", level="INFO")

    # Set parameters
    retailers = [
        "coup_de_pates",
        "ds_restauration",
        "ducreux",
        "even",
        "metro",
        "pomona",
        "pro_a_pro",
        "relais_dor",
        "sysco",
        "transgourmet",
    ]

    # Columns that must exist in the DataFrame
    required_columns_pdt, required_columns_customer = (
        configuration["required_columns_pdt"],
        configuration["required_columns_customer"],
    )
    required_columns_pdt_std, required_columns_customer_std = (
        configuration["required_columns_pdt_standard"],
        configuration["required_columns_customer_standard"],
    )

    selected_columns = (
        ["product_id", "product_code", "product_name", "brand_name"]
        + list(required_columns_pdt.keys())
        + list(required_columns_customer.keys())
        + list(required_columns_pdt_std.keys())
        + list(required_columns_customer_std.keys())
        + ["department", "year", "month", "turnover", "volume", "volume_unit"]
    )

    # Columns to strip
    col_strip = (
        [
            "product_id",
            "product_code",
            "product_name",
            "brand_name",
            "year",
        ]
        + list(required_columns_pdt.keys())
        + list(required_columns_customer.keys())
    )

    # Create a dictionary with keys as level names and values as Dataset instances
    datasets = {level: Dataset() for level in list(required_columns_pdt_std.keys())}

    # --------
    # Retailer
    # --------

    for retailer_name in retailers:
        logger.info(
            f"------------- Datasets to be processed : {retailer_name.upper()} -------------"
        )

        # Configure file to write logs
        wb = xlsxwriter.Workbook(
            f"data/log/model_{args.model_id}/log_{retailer_name}.xlsx"
        )

        ## Step 1: Retailer Dataset

        # Rename columns
        retailer_dataset = pl.read_parquet(
            f"{BRONZE_PATH}{retailer_name}.parquet"
        ).rename(configuration[retailer_name]["rename"], strict=False)

        # Add required columns
        retailer_dataset = add_required_columns(
            retailer_dataset, [required_columns_pdt, required_columns_customer]
        )

        # Strip string columns
        retailer_dataset = retailer_dataset.with_columns(
            [
                pl.col(c).cast(pl.String).str.strip_chars()
                for c in col_strip
                if c in retailer_dataset.columns
            ]
        )

        # Fill null and ""
        retailer_dataset = handle_empty(
            retailer_dataset,
            list(required_columns_pdt.keys()) + list(required_columns_customer.keys()),
        )

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

        # add_columns
        if configuration.get(retailer_name, {}).get('add_columns', False):
            retailer_dataset = retailer_dataset.with_columns(
                [
                    pl.lit(default_value).alias(col)
                    for col, default_value in configuration[retailer_name][
                        "add_columns"
                    ].items()
                ]
            )

        ## Step 2 : Product Nomenclature

        if args.model_id == 1:
            retailer_map_pdt = pl.read_excel(
                f"{GOLD_PATH}mapping/mapping_model1.xlsx", sheet_name=retailer_name
            ).rename(configuration[retailer_name]["rename"], strict=False)
        elif args.model_id == 2 or args.model_id == 4:
            retailer_map_pdt = pl.read_csv(
                f"{GOLD_PATH}mapping/mapping_model{args.model_id}.csv",
                separator=";",
                encoding="cp1252",
            )
        else:
            retailer_map_pdt = pl.read_csv(
                f"{GOLD_PATH}mapping/mapping_model{args.model_id}.csv", separator=";"
            )

        retailer_map_pdt = retailer_map_pdt.filter(
            pl.col("ENTREPRISE") == retailer_name
        )

        # Add required columns
        retailer_map_pdt = add_required_columns(
            retailer_map_pdt, [required_columns_pdt, required_columns_pdt_std]
        )

        # Fill null and ""
        retailer_map_pdt = handle_empty(
            retailer_map_pdt, list(required_columns_pdt.keys())
        )

        # Strip
        retailer_map_pdt = retailer_map_pdt.with_columns(
            [
                pl.col(c).cast(pl.String).str.strip_chars()
                for c in retailer_map_pdt.columns
            ]
        )

        # Replace specific characters for metro
        if retailer_name == "metro":
            retailer_map_pdt = retailer_map_pdt.with_columns(
                [
                    pl.col(c).cast(pl.Utf8).str.replace("\?", "ї")
                    for c in [f"level_{i}" for i in range(1, 8)]
                ]
            )

        ## Step 2 bis : customer Nomenclature

        retailer_map_customer = pl.read_csv(
            f"{GOLD_PATH}mapping/mapping_customer.csv", separator=";", encoding="cp1252"
        ).filter(pl.col("ENSEIGNE") == retailer_name)

        # Add required columns
        retailer_map_customer = add_required_columns(
            retailer_map_customer,
            [required_columns_customer, required_columns_customer_std],
        )

        # Fill null and ""
        retailer_map_customer = handle_empty(
            retailer_map_customer, list(required_columns_customer.keys())
        )

        # Strip
        retailer_map_customer = retailer_map_customer.with_columns(
            [
                pl.col(c).cast(pl.String).str.strip_chars()
                for c in retailer_map_customer.columns
            ]
        )

        ## Step 3 : Mapping

        dataset = (
            retailer_dataset.join(
                retailer_map_pdt, on=list(required_columns_pdt.keys()), how="left"
            )
            .join(
                retailer_map_customer,
                on=list(required_columns_customer.keys()),
                how="left",
            )
            .select(
                [
                    c
                    for c in selected_columns
                    if c
                    in retailer_dataset.columns
                    + retailer_map_pdt.columns
                    + retailer_map_customer.columns
                ]
            )
        )

        # Upload enhanced dataset retailer
        if "product_id" in retailer_dataset.columns:
            dataset_for_comparison = (
                dataset.with_columns(
                    pl.concat_str(
                        list(required_columns_pdt.keys()), separator=">"
                    ).alias("classification")
                )
                .select(
                    ["product_id", "product_name", "classification", "volume_unit"]
                    + [c for c in dataset.columns if c.endswith("_standard")]
                )
                .unique()
                .rename(lambda col: f"{col}_{retailer_name}")
                .rename({f"product_id_{retailer_name}": "product_id"})
            )
            dataset_for_comparison.write_parquet(
                f"{SILVER_PATH}/with_id/{retailer_name}.parquet"
            )
        else:
            dataset_for_comparison = (
                dataset.with_columns(
                    pl.concat_str(
                        list(required_columns_pdt.keys()), separator=">"
                    ).alias("classification")
                )
                .select(
                    ["product_code", "product_name", "classification", "volume_unit"]
                    + [c for c in dataset.columns if c.endswith("_standard")]
                )
                .unique()
                .rename(lambda col: f"{col}_{retailer_name}")
            )
            dataset_for_comparison.write_parquet(
                f"{SILVER_PATH}/without_id/{retailer_name}.parquet"
            )

        logger.info(
            f"retailer_dataset shape/matched rows : {retailer_dataset.shape[0]} / {dataset.filter(pl.col('level_1_standard').is_not_null()).shape[0]})"
        )
        logger.info(
            f"retailer_dataset shape/matched rows customer : {retailer_dataset.shape[0]} / {dataset.filter(pl.col('segment_1_standard').is_not_null()).shape[0]})"
        )

        ## Step 4: Unmatched levels

        unmatched_levels = (
            dataset.filter(
                pl.any_horizontal(
                    pl.col(list(required_columns_pdt_std.keys())).is_null()
                )
            )
            .select(
                list(required_columns_pdt.keys())
                + list(required_columns_pdt_std.keys())
            )
            .unique()
            .sort(
                list(required_columns_pdt.keys())
                + list(required_columns_pdt_std.keys())
            )
        )
        if unmatched_levels.is_empty():
            pass
        else:
            unmatched_levels.write_excel(workbook=wb, worksheet="unmatched_levels_pdt")

        unmatched_levels_customer = (
            dataset.filter(
                pl.any_horizontal(
                    pl.col(list(required_columns_customer_std.keys())).is_null()
                )
            )
            .select(
                list(required_columns_customer.keys())
                + list(required_columns_customer_std.keys())
            )
            .unique()
            .sort(
                list(required_columns_customer.keys())
                + list(required_columns_customer_std.keys())
            )
        )
        if unmatched_levels_customer.is_empty():
            pass
        else:
            unmatched_levels_customer.write_excel(
                workbook=wb, worksheet="unmatched_levels_customer"
            )

        ## Step 5: Unicity checks

        cols_to_check_lst = [
            list(required_columns_pdt_std.keys()),
            ["product_id"],
        ]
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

        agg_cols_list = incremental_sublists(list(required_columns_pdt_std.keys()))
        for i, agg_cols in enumerate(agg_cols_list, start=1):
            # TOTAL
            kpis_total = compute_kpis(dataset, ["year"] + agg_cols)
            kpis_total = kpis_total.with_columns(
                pl.lit(retailer_name).alias("retailer")
            )

            datasets[f"level_{i}_standard"].main.append(kpis_total)
            kpis_total.write_excel(workbook=wb, worksheet=f"kpis_l{i}")

            # DEP
            Dep_kpis = compute_kpis(dataset, ["year", "department"] + agg_cols)
            Dep_kpis = Dep_kpis.with_columns(pl.lit(retailer_name).alias("retailer"))
            datasets[f"level_{i}_standard"].dep.append(Dep_kpis)
            Dep_kpis.write_excel(workbook=wb, worksheet=f"kpis_l{i}Dep")

            # C1 DEP
            c1Dep_kpis = compute_kpis(
                dataset, ["year", "department", "segment_1_standard"] + agg_cols
            )
            c1Dep_kpis = c1Dep_kpis.with_columns(
                pl.lit(retailer_name).alias("retailer")
            )
            datasets[f"level_{i}_standard"].c1_dep.append(c1Dep_kpis)
            c1Dep_kpis.write_excel(workbook=wb, worksheet=f"l{i}c1Dep_kpis")

            c2Dep_kpis = compute_kpis(
                dataset,
                [
                    "year",
                    "department",
                    "segment_1_standard",
                    "segment_2_standard",
                ]
                + agg_cols,
            )
            c2Dep_kpis = c2Dep_kpis.with_columns(
                pl.lit(retailer_name).alias("retailer")
            )
            datasets[f"level_{i}_standard"].c2_dep.append(c2Dep_kpis)
            c2Dep_kpis.write_excel(workbook=wb, worksheet=f"l{i}c2Dep_kpis")

            c3Dep_kpis = compute_kpis(
                dataset,
                [
                    "year",
                    "department",
                    "segment_1_standard",
                    "segment_2_standard",
                    "segment_3_standard",
                ]
                + agg_cols,
            )
            c3Dep_kpis = c3Dep_kpis.with_columns(
                pl.lit(retailer_name).alias("retailer")
            )
            datasets[f"level_{i}_standard"].c3_dep.append(c3Dep_kpis)
            c3Dep_kpis.write_excel(workbook=wb, worksheet=f"l{i}c3Dep_kpis")

        wb.close()

    # ------
    # Global
    # ------

    # Market Share
    logger.info(f"\n------------- Market Share -------------")

    wb_global = xlsxwriter.Workbook(
        f"data/log/model_{args.model_id}/log_global.xlsx",
        options={"nan_inf_to_errors": True},
    )

    for i, agg_cols in enumerate(agg_cols_list, start=1):
        logger.info(f"\n Level {i} Standard")
        ms_total = agg_kpis(
            dataset_lst=datasets[f"level_{i}_standard"].main, cols=["year"] + agg_cols
        )
        if ms_total.is_empty():
            pass
        else:
            ms_total.write_excel(workbook=wb_global, worksheet=f"ms_l{i}")

        ms_Dep = agg_kpis(
            dataset_lst=datasets[f"level_{i}_standard"].dep,
            cols=["year", "department"] + agg_cols,
        )
        if ms_Dep.is_empty():
            pass
        else:
            ms_Dep.write_excel(workbook=wb_global, worksheet=f"ms_l{i}Dep")

        ms_c1Dep = agg_kpis(
            dataset_lst=datasets[f"level_{i}_standard"].c1_dep,
            cols=["year", "segment_1_standard", "department"] + agg_cols,
        )
        if ms_c1Dep.is_empty():
            pass
        else:
            ms_c1Dep.write_parquet(
                f"data/log/model_{args.model_id}/ms_l{i}c1Dep.parquet"
            )
            ms_c1Dep.write_excel(workbook=wb_global, worksheet=f"ms_l{i}c1Dep")

        ms_c2Dep = agg_kpis(
            dataset_lst=datasets[f"level_{i}_standard"].c2_dep,
            cols=[
                "year",
                "segment_1_standard",
                "segment_2_standard",
                "department",
            ]
            + agg_cols,
        )
        if ms_c2Dep.is_empty():
            pass
        else:
            ms_c2Dep.write_parquet(
                f"data/log/model_{args.model_id}/ms_l{i}c2Dep.parquet"
            )
            ms_c2Dep.write_excel(workbook=wb_global, worksheet=f"ms_l{i}c2Dep")

        ms_c3Dep = agg_kpis(
            dataset_lst=datasets[f"level_{i}_standard"].c3_dep,
            cols=[
                "year",
                "segment_1_standard",
                "segment_2_standard",
                "segment_3_standard",
                "department",
            ]
            + agg_cols,
        )
        if ms_c3Dep.is_empty():
            pass
        else:
            ms_c3Dep.write_parquet(
                f"data/log/model_{args.model_id}/ms_l{i}c3Dep.parquet"
            )
            ms_c3Dep.write_excel(workbook=wb_global, worksheet=f"ms_l{i}c3Dep")

    wb_global.close()

    ## Misclassification
    logger.info(f"\n------------- Misclassification -------------")

    dataset_concat = pl.concat(
        [
            pl.read_parquet(f"{SILVER_PATH}with_id/{filename}").filter(
                ~pl.col(f'level_4_standard_{filename.split(".")[0]}').is_in(
                    ["A L'EAN", "A l'EAN"]
                )
            )
            for filename in os.listdir(f"{SILVER_PATH}with_id")
            if filename.endswith("parquet")
        ],
        how="align",
    )

    columns_level1 = [
        c for c in dataset_concat.columns if c.startswith("level_1_standard_")
    ]
    columns_level2 = [
        c for c in dataset_concat.columns if c.startswith("level_2_standard_")
    ]
    columns_level3 = [
        c for c in dataset_concat.columns if c.startswith("level_3_standard_")
    ]
    columns_level4 = [
        c for c in dataset_concat.columns if c.startswith("level_4_standard_")
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
        .with_columns(pl.concat_list(columns_level3).alias("level3_lst"))
        .with_columns(pl.concat_list(columns_level3).alias("level4_lst"))
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
        .with_columns(
            pl.col("level3_lst").map_elements(
                lambda x: list(set([e for e in x if e is not None]))
            )
        )
        .with_columns(
            pl.col("level4_lst").map_elements(
                lambda x: list(set([e for e in x if e is not None]))
            )
        )
        .with_columns(pl.col("level1_lst").map_elements(lambda x: len(x)))
        .with_columns(pl.col("level2_lst").map_elements(lambda x: len(x)))
        .with_columns(pl.col("level3_lst").map_elements(lambda x: len(x)))
        .with_columns(pl.col("level4_lst").map_elements(lambda x: len(x)))
        .filter(
            (pl.col("level1_lst") > 1)
            | (pl.col("level2_lst") > 1)
            | (pl.col("level3_lst") > 1)
            | (pl.col("level4_lst") > 1)
        )
        .drop(["level1_lst", "level2_lst", "level3_lst", "level4_lst"])
        .unique()
    )
    filtered_dataset_concat.write_parquet(
        f"data/log/model_{args.model_id}/misclassification.parquet"
    )

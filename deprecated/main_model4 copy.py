import os
import polars as pl
import yaml
import argparse
from loguru import logger
import xlsxwriter
from itertools import combinations
from functools import reduce

from src.utils.transform import add_required_columns, handle_empty
from src.utils.profiling import check_unicity, compute_kpis, agg_kpis

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
    transgourmet,
)

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
        ("coup_de_pates", coup_de_pates),
        ("ds_restauration", ds_restauration),
        ("ducreux", ducreux),
        ("even", even),
        ("metro", metro),
        ("pomona", pomona),
        ("pro_a_pro", pro_a_pro),
        ("relais_dor", relais_dor),
        ("sysco", sysco),
        ("transgourmet", transgourmet),
    ]

    # Columns that must exist in the DataFrame
    required_columns_pdt, required_columns_client = configuration["required_columns_pdt"], configuration["required_columns_client"]

    selected_columns = (
        ["product_id", "product_code", "product_name", "brand_name"]
        + list(required_columns_pdt.keys())
        + list(required_columns_client.keys())
        + [
            "level_1_standard",
            "level_2_standard",
            "level_3_standard",
            "level_4_standard",
            "segment_1_standard",
            "segment_2_standard",
            "segment_3_standard",
        ]
        + ["department", "year", "month", "turnover"]
    )

    col_strip = (
        [
            "product_id",
            "product_code",
            "product_name",
            "brand_name",
            "year",
        ]
        + list(required_columns_pdt.keys())
        + list(required_columns_client.keys())
    )

    # --------
    # Retailer
    # --------

    (
        l1_dataset_lst,
        l1Dep_dataset_lst,
        l1c1Dep_dataset_lst,
        l1c2Dep_dataset_lst,
        l1c3Dep_dataset_lst,
    ) = ([], [], [], [], [])
    (
        l2_dataset_lst,
        l2Dep_dataset_lst,
        l2c1Dep_dataset_lst,
        l2c2Dep_dataset_lst,
        l2c3Dep_dataset_lst,
    ) = ([], [], [], [], [])
    (
        l3_dataset_lst,
        l3Dep_dataset_lst,
        l3c1Dep_dataset_lst,
        l3c2Dep_dataset_lst,
        l3c3Dep_dataset_lst,
    ) = ([], [], [], [], [])
    (
        l4_dataset_lst,
        l4Dep_dataset_lst,
        l4c1Dep_dataset_lst,
        l4c2Dep_dataset_lst,
        l4c3Dep_dataset_lst,
    ) = ([], [], [], [], [])

    for retailer_name, retailer_dataset in retailers:
        logger.info(
            f"------------- Datasets to be processed : {retailer_name.upper()} -------------"
        )

        # Configure file to write logs
        wb = xlsxwriter.Workbook(
            f"data/log/model_{args.model_id}/log_{retailer_name}.xlsx"
        )

        ## Step 1: retailer_dataset

        # Rename columns
        retailer_dataset = retailer_dataset.rename(
            configuration[retailer_name]["rename"], strict=False
        )

        # Add required columns
        retailer_dataset = add_required_columns(retailer_dataset, required_columns_pdt)
        retailer_dataset = add_required_columns(
            retailer_dataset, required_columns_client
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
            list(required_columns_pdt.keys()) + list(required_columns_client.keys()),
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

        ## Step 2 : retailer_map
        if args.model_id == "1":
            retailer_map = pl.read_excel(
                "data/mapping_model1.xlsx", sheet_name=retailer_name
            ).rename(configuration[retailer_name]["rename"], strict=False)
        elif args.model_id == "2" or args.model_id == "4":
            retailer_map = pl.read_csv(
                f"data/mapping_model{args.model_id}.csv", separator=";", encoding="cp1252"
            )
        else:
            retailer_map = pl.read_csv(
                f"data/mapping_model{args.model_id}.csv", separator=";"
            )

        retailer_map = retailer_map.filter(pl.col("ENTREPRISE") == retailer_name)

        # Add required columns
        retailer_map = add_required_columns(retailer_map, required_columns_pdt)

        # Fill null and ""
        retailer_map = handle_empty(retailer_map, list(required_columns_pdt.keys()))

        # Strip
        retailer_map = retailer_map.with_columns(
            [pl.col(c).cast(pl.String).str.strip_chars() for c in retailer_map.columns]
        )
        if retailer_name == "metro":
            retailer_map = retailer_map.with_columns(
                [
                    pl.col(c).cast(pl.Utf8).str.replace("\?", "ї")
                    for c in [
                        "level_1",
                        "level_2",
                        "level_3",
                        "level_4",
                        "level_5",
                        "level_6",
                        "level_7",
                    ]
                ]
            )

        ## Step 2 bis : retailer_map_client
        retailer_map_client = pl.read_csv(
            f"data/mapping_client.csv", separator=";", encoding="cp1252"
        ).filter(pl.col("ENSEIGNE") == retailer_name)

        # Add required columns
        retailer_map_client = add_required_columns(
            retailer_map_client, required_columns_client
        )

        # Fill null and ""
        retailer_map_client = handle_empty(
            retailer_map_client, list(required_columns_client.keys())
        )

        # Strip
        retailer_map_client = retailer_map_client.with_columns(
            [
                pl.col(c).cast(pl.String).str.strip_chars()
                for c in retailer_map_client.columns
            ]
        )

        ## Step 3 : Mapping

        dataset = (
            retailer_dataset.join(
                retailer_map, on=list(required_columns_pdt.keys()), how="left"
            )
            .join(
                retailer_map_client, on=list(required_columns_client.keys()), how="left"
            )
            .select(
                [
                    c
                    for c in selected_columns
                    if c
                    in retailer_dataset.columns
                    + retailer_map.columns
                    + retailer_map_client.columns
                ]
            )
        )

        if "product_id" in retailer_dataset.columns:
            dataset_for_comparison = (
                dataset.with_columns(
                    pl.concat_str(
                        list(required_columns_pdt.keys()), separator=">"
                    ).alias("classification")
                )
                .select(
                    ["product_id", "product_name", "classification"]
                    + [c for c in dataset.columns if c.endswith("_standard")]
                )
                .unique()
                .rename(lambda col: f"{col}_{retailer_name}")
                .rename({f"product_id_{retailer_name}": "product_id"})
            )
            dataset_for_comparison.write_parquet(f"data/tmp/{retailer_name}.parquet")

        logger.info(
            f"retailer_dataset shape/matched rows : {retailer_dataset.shape} / {dataset.filter(pl.col('level_1_standard').is_not_null()).shape})"
        )
        logger.info(
            f"retailer_dataset shape/matched rows client : {retailer_dataset.shape} / {dataset.filter(pl.col('segment_1_standard').is_not_null()).shape})"
        )

        ## Step 4: Unmatched levels

        unmatched_levels = (
            dataset.filter(
                (pl.col("level_1_standard").is_null())
                | (pl.col("level_2_standard").is_null())
                | (pl.col("level_3_standard").is_null())
                | (pl.col("level_4_standard").is_null())
            )
            .select(
                list(required_columns_pdt.keys())
                + [
                    "level_1_standard",
                    "level_2_standard",
                    "level_3_standard",
                    "level_4_standard",
                ]
            )
            .unique()
            .sort(
                list(required_columns_pdt.keys())
                + [
                    "level_1_standard",
                    "level_2_standard",
                    "level_3_standard",
                    "level_4_standard",
                ]
            )
        )
        if unmatched_levels.is_empty():
            pass
        else:
            unmatched_levels.write_excel(workbook=wb, worksheet="unmatched_levels")

        unmatched_levels_client = (
            dataset.filter(
                (pl.col("segment_1_standard").is_null())
                | (pl.col("segment_2_standard").is_null())
                | (pl.col("segment_3_standard").is_null())
            )
            .select(
                list(required_columns_client.keys())
                + [
                    "segment_1_standard",
                    "segment_2_standard",
                    "segment_3_standard",
                ]
            )
            .unique()
            .sort(
                list(required_columns_client.keys())
                + [
                    "segment_1_standard",
                    "segment_2_standard",
                    "segment_3_standard",
                ]
            )
        )
        if unmatched_levels_client.is_empty():
            pass
        else:
            unmatched_levels_client.write_excel(
                workbook=wb, worksheet="unmatched_levels_client"
            )

        ## Step 5: Unicity checks

        cols_to_check_lst = [
            [
                "level_1_standard",
                "level_2_standard",
                "level_3_standard",
                "level_4_standard",
            ],
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

        # TOTAL
        l1_kpis = compute_kpis(dataset, ["year", "level_1_standard"])
        l1_kpis = l1_kpis.with_columns(pl.lit(retailer_name).alias("retailer"))
        l1_dataset_lst.append(l1_kpis)
        l1_kpis.write_excel(workbook=wb, worksheet=f"kpis_l1")

        # DEP
        l1Dep_kpis = compute_kpis(dataset, ["year", "level_1_standard", "department"])
        l1Dep_kpis = l1Dep_kpis.with_columns(pl.lit(retailer_name).alias("retailer"))
        l1Dep_dataset_lst.append(l1Dep_kpis)
        l1Dep_kpis.write_excel(workbook=wb, worksheet=f"kpis_l1Dep")

        # CLI
        l1c1Dep_kpis = compute_kpis(
            dataset, ["year", "level_1_standard", "department", "segment_1_standard"]
        )
        l1c1Dep_kpis = l1c1Dep_kpis.with_columns(
            pl.lit(retailer_name).alias("retailer")
        )
        l1c1Dep_dataset_lst.append(l1c1Dep_kpis)
        l1c1Dep_kpis.write_excel(workbook=wb, worksheet=f"l1c1Dep_kpis")

        l1c2Dep_kpis = compute_kpis(
            dataset,
            [
                "year",
                "level_1_standard",
                "department",
                "segment_1_standard",
                "segment_2_standard",
            ],
        )
        l1c2Dep_kpis = l1c2Dep_kpis.with_columns(
            pl.lit(retailer_name).alias("retailer")
        )
        l1c2Dep_dataset_lst.append(l1c2Dep_kpis)
        l1c2Dep_kpis.write_excel(workbook=wb, worksheet=f"l1c2Dep_kpis")

        l1c3Dep_kpis = compute_kpis(
            dataset,
            [
                "year",
                "level_1_standard",
                "department",
                "segment_1_standard",
                "segment_2_standard",
                "segment_3_standard",
            ],
        )
        l1c3Dep_kpis = l1c3Dep_kpis.with_columns(
            pl.lit(retailer_name).alias("retailer")
        )
        l1c3Dep_dataset_lst.append(l1c3Dep_kpis)
        l1c3Dep_kpis.write_excel(workbook=wb, worksheet=f"l1c3Dep_kpis")

        # TOTAL
        l2_kpis = compute_kpis(
            dataset, ["year", "level_1_standard", "level_2_standard"]
        )
        l2_kpis = l2_kpis.with_columns(pl.lit(retailer_name).alias("retailer"))
        l2_dataset_lst.append(l2_kpis)
        l2_kpis.write_excel(workbook=wb, worksheet=f"kpis_l2")

        # DEP
        l2Dep_kpis = compute_kpis(
            dataset, ["year", "level_1_standard", "level_2_standard", "department"]
        )
        l2Dep_kpis = l2Dep_kpis.with_columns(pl.lit(retailer_name).alias("retailer"))
        l2Dep_dataset_lst.append(l2Dep_kpis)
        l2Dep_kpis.write_excel(workbook=wb, worksheet=f"kpis_l2Dep")

        # CLI
        l2c1Dep_kpis = compute_kpis(
            dataset,
            [
                "year",
                "level_1_standard",
                "level_2_standard",
                "department",
                "segment_1_standard",
            ],
        )
        l2c1Dep_kpis = l2c1Dep_kpis.with_columns(
            pl.lit(retailer_name).alias("retailer")
        )
        l2c1Dep_dataset_lst.append(l2c1Dep_kpis)
        l2c1Dep_kpis.write_excel(workbook=wb, worksheet=f"l2c1Dep_kpis")

        l2c2Dep_kpis = compute_kpis(
            dataset,
            [
                "year",
                "level_1_standard",
                "level_2_standard",
                "department",
                "segment_1_standard",
                "segment_2_standard",
            ],
        )
        l2c2Dep_kpis = l2c2Dep_kpis.with_columns(
            pl.lit(retailer_name).alias("retailer")
        )
        l2c2Dep_dataset_lst.append(l2c2Dep_kpis)
        l2c2Dep_kpis.write_excel(workbook=wb, worksheet=f"l2c2Dep_kpis")

        l2c3Dep_kpis = compute_kpis(
            dataset,
            [
                "year",
                "level_1_standard",
                "level_2_standard",
                "department",
                "segment_1_standard",
                "segment_2_standard",
                "segment_3_standard",
            ],
        )
        l2c3Dep_kpis = l2c3Dep_kpis.with_columns(
            pl.lit(retailer_name).alias("retailer")
        )
        l2c3Dep_dataset_lst.append(l2c3Dep_kpis)
        l2c3Dep_kpis.write_excel(workbook=wb, worksheet=f"l2c3Dep_kpis")

        # TOTAL
        l3_kpis = compute_kpis(
            dataset,
            ["year", "level_1_standard", "level_2_standard", "level_3_standard"],
        )
        l3_kpis = l3_kpis.with_columns(pl.lit(retailer_name).alias("retailer"))
        l3_dataset_lst.append(l3_kpis)
        l3_kpis.write_excel(workbook=wb, worksheet=f"kpis_l3")

        # DEP
        l3Dep_kpis = compute_kpis(
            dataset,
            [
                "year",
                "level_1_standard",
                "level_2_standard",
                "level_3_standard",
                "department",
            ],
        )
        l3Dep_kpis = l3Dep_kpis.with_columns(pl.lit(retailer_name).alias("retailer"))
        l3Dep_dataset_lst.append(l3Dep_kpis)
        l3Dep_kpis.write_excel(workbook=wb, worksheet=f"kpis_l3Dep")

        # CLI
        l3c1Dep_kpis = compute_kpis(
            dataset,
            [
                "year",
                "level_1_standard",
                "level_2_standard",
                "level_3_standard",
                "department",
                "segment_1_standard",
            ],
        )
        l3c1Dep_kpis = l3c1Dep_kpis.with_columns(
            pl.lit(retailer_name).alias("retailer")
        )
        l3c1Dep_dataset_lst.append(l3c1Dep_kpis)
        l3c1Dep_kpis.write_excel(workbook=wb, worksheet=f"l3c1Dep_kpis")

        l3c2Dep_kpis = compute_kpis(
            dataset,
            [
                "year",
                "level_1_standard",
                "level_2_standard",
                "level_3_standard",
                "department",
                "segment_1_standard",
                "segment_2_standard",
            ],
        )
        l3c2Dep_kpis = l3c2Dep_kpis.with_columns(
            pl.lit(retailer_name).alias("retailer")
        )
        l3c2Dep_dataset_lst.append(l3c2Dep_kpis)
        l3c2Dep_kpis.write_excel(workbook=wb, worksheet=f"l3c2Dep_kpis")

        l3c3Dep_kpis = compute_kpis(
            dataset,
            [
                "year",
                "level_1_standard",
                "level_2_standard",
                "level_3_standard",
                "department",
                "segment_1_standard",
                "segment_2_standard",
                "segment_3_standard",
            ],
        )
        l3c3Dep_kpis = l3c3Dep_kpis.with_columns(
            pl.lit(retailer_name).alias("retailer")
        )
        l3c3Dep_dataset_lst.append(l3c3Dep_kpis)
        l3c3Dep_kpis.write_excel(workbook=wb, worksheet=f"l3c3Dep_kpis")

        # TOTAL
        l4_kpis = compute_kpis(
            dataset,
            [
                "year",
                "level_1_standard",
                "level_2_standard",
                "level_3_standard",
                "level_4_standard",
            ],
        )
        l4_kpis = l4_kpis.with_columns(pl.lit(retailer_name).alias("retailer"))
        l4_dataset_lst.append(l4_kpis)
        l4_kpis.write_excel(workbook=wb, worksheet=f"kpis_l4")

        # DEP
        l4Dep_kpis = compute_kpis(
            dataset,
            [
                "year",
                "level_1_standard",
                "level_2_standard",
                "level_3_standard",
                "level_4_standard",
                "department",
            ],
        )
        l4Dep_kpis = l4Dep_kpis.with_columns(pl.lit(retailer_name).alias("retailer"))
        l4Dep_dataset_lst.append(l4Dep_kpis)
        l4Dep_kpis.write_excel(workbook=wb, worksheet=f"kpis_l4Dep")

        # CLI
        l4c1Dep_kpis = compute_kpis(
            dataset,
            [
                "year",
                "level_1_standard",
                "level_2_standard",
                "level_3_standard",
                "level_4_standard",
                "department",
                "segment_1_standard",
            ],
        )
        l4c1Dep_kpis = l4c1Dep_kpis.with_columns(
            pl.lit(retailer_name).alias("retailer")
        )
        l4c1Dep_dataset_lst.append(l4c1Dep_kpis)
        l4c1Dep_kpis.write_excel(workbook=wb, worksheet=f"l4c1Dep_kpis")

        l4c2Dep_kpis = compute_kpis(
            dataset,
            [
                "year",
                "level_1_standard",
                "level_2_standard",
                "level_3_standard",
                "level_4_standard",
                "department",
                "segment_1_standard",
                "segment_2_standard",
            ],
        )
        l4c2Dep_kpis = l4c2Dep_kpis.with_columns(
            pl.lit(retailer_name).alias("retailer")
        )
        l4c2Dep_dataset_lst.append(l4c2Dep_kpis)
        l4c2Dep_kpis.write_excel(workbook=wb, worksheet=f"l4c2Dep_kpis")

        l4c3Dep_kpis = compute_kpis(
            dataset,
            [
                "year",
                "level_1_standard",
                "level_2_standard",
                "level_3_standard",
                "level_4_standard",
                "department",
                "segment_1_standard",
                "segment_2_standard",
                "segment_3_standard",
            ],
        )
        l4c3Dep_kpis = l4c3Dep_kpis.with_columns(
            pl.lit(retailer_name).alias("retailer")
        )
        l4c3Dep_dataset_lst.append(l4c3Dep_kpis)
        l4c3Dep_kpis.write_excel(workbook=wb, worksheet=f"l4c3Dep_kpis")

        wb.close()

    # ------
    # Global
    # ------

    wb_global = xlsxwriter.Workbook(
        f"data/log/model_{args.model_id}/log_global.xlsx",
        options={"nan_inf_to_errors": True},
    )

    # Market Share
    # logger.info(f"\n------------- Market Share -------------")
    ms_l1 = agg_kpis(dataset_lst=l1_dataset_lst, cols=["year", "level_1_standard"])
    if ms_l1.is_empty():
        pass
    else:
        ms_l1.write_excel(workbook=wb_global, worksheet="ms_l1")

    ms_l1Dep = agg_kpis(
        dataset_lst=l1Dep_dataset_lst, cols=["year", "level_1_standard", "department"]
    )
    if ms_l1Dep.is_empty():
        pass
    else:
        ms_l1Dep.write_excel(workbook=wb_global, worksheet="ms_l1Dep")
    ms_l1c1Dep = agg_kpis(
        dataset_lst=l1c1Dep_dataset_lst,
        cols=["year", "level_1_standard", "segment_1_standard", "department"],
    )

    if ms_l1c1Dep.is_empty():
        pass
    else:
        ms_l1c1Dep.write_excel(workbook=wb_global, worksheet="ms_l1c1Dep")

    ms_l1c2Dep = agg_kpis(
        dataset_lst=l1c2Dep_dataset_lst,
        cols=[
            "year",
            "level_1_standard",
            "segment_1_standard",
            "segment_2_standard",
            "department",
        ],
    )
    if ms_l1c2Dep.is_empty():
        pass
    else:
        ms_l1c2Dep.write_excel(workbook=wb_global, worksheet="ms_l1c2Dep")

    ms_l1c3Dep = agg_kpis(
        dataset_lst=l1c3Dep_dataset_lst,
        cols=[
            "year",
            "level_1_standard",
            "segment_1_standard",
            "segment_2_standard",
            "segment_3_standard",
            "department",
        ],
    )
    if ms_l1c3Dep.is_empty():
        pass
    else:
        ms_l1c3Dep.write_excel(workbook=wb_global, worksheet="ms_l1c3Dep")

    ms_l2 = agg_kpis(
        dataset_lst=l2_dataset_lst,
        cols=["year", "level_1_standard", "level_2_standard"],
    )
    if ms_l2.is_empty():
        pass
    else:
        ms_l2.write_excel(workbook=wb_global, worksheet="ms_l2")

    ms_l2Dep = agg_kpis(
        dataset_lst=l2Dep_dataset_lst,
        cols=["year", "level_1_standard", "level_2_standard", "department"],
    )
    if ms_l2Dep.is_empty():
        pass
    else:
        ms_l2Dep.write_excel(workbook=wb_global, worksheet="ms_l2Dep")

    ms_l2c1Dep = agg_kpis(
        dataset_lst=l2c1Dep_dataset_lst,
        cols=[
            "year",
            "level_1_standard",
            "level_2_standard",
            "segment_1_standard",
            "department",
        ],
    )
    if ms_l2c1Dep.is_empty():
        pass
    else:
        ms_l2c1Dep.write_excel(workbook=wb_global, worksheet="ms_l2c1Dep")

    ms_l2c2Dep = agg_kpis(
        dataset_lst=l2c2Dep_dataset_lst,
        cols=[
            "year",
            "level_1_standard",
            "level_2_standard",
            "segment_1_standard",
            "segment_2_standard",
            "department",
        ],
    )
    if ms_l1c2Dep.is_empty():
        pass
    else:
        ms_l2c2Dep.write_excel(workbook=wb_global, worksheet="ms_l2c2Dep")

    ms_l2c3Dep = agg_kpis(
        dataset_lst=l2c3Dep_dataset_lst,
        cols=[
            "year",
            "level_1_standard",
            "level_2_standard",
            "segment_1_standard",
            "segment_2_standard",
            "segment_3_standard",
            "department",
        ],
    )
    if ms_l2c3Dep.is_empty():
        pass
    else:
        ms_l2c3Dep.write_excel(workbook=wb_global, worksheet="ms_l2c3Dep")

    ms_l3 = agg_kpis(
        dataset_lst=l3_dataset_lst,
        cols=["year", "level_1_standard", "level_2_standard", "level_3_standard"],
    )
    if ms_l3.is_empty():
        pass
    else:
        ms_l3.write_excel(workbook=wb_global, worksheet="ms_l3")

    ms_l3Dep = agg_kpis(
        dataset_lst=l3Dep_dataset_lst,
        cols=[
            "year",
            "level_1_standard",
            "level_2_standard",
            "level_3_standard",
            "department",
        ],
    )
    if ms_l3Dep.is_empty():
        pass
    else:
        ms_l3Dep.write_excel(workbook=wb_global, worksheet="ms_l3Dep")

    ms_l3c1Dep = agg_kpis(
        dataset_lst=l3c1Dep_dataset_lst,
        cols=[
            "year",
            "level_1_standard",
            "level_2_standard",
            "level_3_standard",
            "segment_1_standard",
            "department",
        ],
    )
    if ms_l3c1Dep.is_empty():
        pass
    else:
        ms_l3c1Dep.write_excel(workbook=wb_global, worksheet="ms_l3c1Dep")

    ms_l3c2Dep = agg_kpis(
        dataset_lst=l3c2Dep_dataset_lst,
        cols=[
            "year",
            "level_1_standard",
            "level_2_standard",
            "level_3_standard",
            "segment_1_standard",
            "segment_2_standard",
            "department",
        ],
    )
    if ms_l3c2Dep.is_empty():
        pass
    else:
        ms_l3c2Dep.write_excel(workbook=wb_global, worksheet="ms_l3c2Dep")

    ms_l3c3Dep = agg_kpis(
        dataset_lst=l3c3Dep_dataset_lst,
        cols=[
            "year",
            "level_1_standard",
            "level_2_standard",
            "level_3_standard",
            "segment_1_standard",
            "segment_2_standard",
            "segment_3_standard",
            "department",
        ],
    )
    if ms_l3c3Dep.is_empty():
        pass
    else:
        ms_l3c3Dep.write_excel(workbook=wb_global, worksheet="ms_l3c3Dep")

    ms_l4 = agg_kpis(
        dataset_lst=l4_dataset_lst,
        cols=[
            "year",
            "level_1_standard",
            "level_2_standard",
            "level_3_standard",
            "level_4_standard",
        ],
    )
    if ms_l4.is_empty():
        pass
    else:
        ms_l4.write_excel(workbook=wb_global, worksheet="ms_l4")

    ms_l4Dep = agg_kpis(
        dataset_lst=l4Dep_dataset_lst,
        cols=[
            "year",
            "level_1_standard",
            "level_2_standard",
            "level_3_standard",
            "level_4_standard",
            "department",
        ],
    )
    if ms_l4Dep.is_empty():
        pass
    else:
        ms_l4Dep.write_excel(workbook=wb_global, worksheet="ms_l4Dep")

    ms_l4c1Dep = agg_kpis(
        dataset_lst=l4c1Dep_dataset_lst,
        cols=[
            "year",
            "level_1_standard",
            "level_2_standard",
            "level_3_standard",
            "level_4_standard",
            "segment_1_standard",
            "department",
        ],
    )
    if ms_l4c1Dep.is_empty():
        pass
    else:
        ms_l4c1Dep.write_excel(workbook=wb_global, worksheet="ms_l4c1Dep")
    ms_l4c1Dep.write_parquet(f"data/log/model_{args.model_id}/ms_l4c1Dep.parquet")

    ms_l4c2Dep = agg_kpis(
        dataset_lst=l4c2Dep_dataset_lst,
        cols=[
            "year",
            "level_1_standard",
            "level_2_standard",
            "level_3_standard",
            "level_4_standard",
            "segment_1_standard",
            "segment_2_standard",
            "department",
        ],
    )
    if ms_l4c2Dep.is_empty():
        pass
    else:
        ms_l4c2Dep.write_excel(workbook=wb_global, worksheet="ms_l4c2Dep")
    ms_l4c2Dep.write_parquet(f"data/log/model_{args.model_id}/ms_l4c2Dep.parquet")

    ms_l4c3Dep = agg_kpis(
        dataset_lst=l4c3Dep_dataset_lst,
        cols=[
            "year",
            "level_1_standard",
            "level_2_standard",
            "level_3_standard",
            "level_4_standard",
            "segment_1_standard",
            "segment_2_standard",
            "segment_3_standard",
            "department",
        ],
    )
    if ms_l4c3Dep.is_empty():
        pass
    else:
        ms_l4c3Dep.write_excel(workbook=wb_global, worksheet="ms_l4c3Dep")
    ms_l4c3Dep.write_parquet(f"data/log/model_{args.model_id}/ms_l4c3Dep.parquet")

    wb_global.close()

    ## Misclassification
    wb_misclassification = xlsxwriter.Workbook(
        f"data/log/model_{args.model_id}/log_misclassification.xlsx",
        options={"nan_inf_to_errors": True},
    )

    dataset_concat = pl.concat(
        [
            pl.read_parquet(f"data/tmp/{filename}").filter(
                ~pl.col(f'level_4_standard_{filename.split(".")[0]}').is_in(
                    ["A L'EAN", "A l'EAN"]
                )
            )
            for filename in os.listdir("data/tmp")
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
    wb_misclassification.write_parquet(
        f"data/log/model_{args.model_id}/misclassification.parquet"
    )

    if filtered_dataset_concat.is_empty():
        pass
    else:
        filtered_dataset_concat.write_excel(
            workbook=wb_misclassification, worksheet="Misclassification"
        )

    wb_misclassification.close()

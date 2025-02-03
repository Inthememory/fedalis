import polars as pl

from data import RAW_PATH, BRONZE_PATH

# COUP_DE_PATES
coup_de_pates = pl.read_excel(f"{RAW_PATH}COUP_DE_PATES.xlsx")
coup_de_pates.write_parquet(f"{BRONZE_PATH}coup_de_pates.parquet")

# DS_RESTAURATION
ds_restauration = pl.concat(
    [
        pl.read_excel(f"{RAW_PATH}DS_RESTAURATION_2023.xlsx"),
        pl.read_excel(f"{RAW_PATH}DS_RESTAURATION_2024.xlsx"),
    ]
)
ds_restauration.write_parquet(f"{BRONZE_PATH}ds_restauration.parquet")

# DUCREUX
ducreux = pl.concat(
    [
        pl.read_excel(f"{RAW_PATH}DUCREUX_2023.xlsx"),
        pl.read_excel(f"{RAW_PATH}DUCREUX_2024.xlsx"),
    ]
)
ducreux.write_parquet(f"{BRONZE_PATH}ducreux.parquet")

# EVEN
even = pl.read_excel(f"{RAW_PATH}EVEN.xlsx")
even.write_parquet(f"{BRONZE_PATH}even.parquet")

# METRO
metro = (
    pl.read_csv(f"{RAW_PATH}METRO_pdt.csv", separator=";")
    .with_columns(pl.col("concatenated_gtin").str.split("|").alias("gtin"))
    .explode("gtin")
)
metro.write_parquet(f"{BRONZE_PATH}metro.parquet")

# POMONA
pomona = pl.concat(
    [
        pl.read_excel(f"{RAW_PATH}POMONA_ES_2023.xlsx", sheet_name="Extract"),
        pl.read_excel(f"{RAW_PATH}POMONA_ES_2024.xlsx", sheet_name="Extract"),
        pl.read_excel(f"{RAW_PATH}POMONA_PF_2023.xlsx", sheet_name="Extract"),
        pl.read_excel(f"{RAW_PATH}POMONA_PF_2024.xlsx", sheet_name="Extract"),
    ]
)
pomona.write_parquet(f"{BRONZE_PATH}pomona.parquet")

# PRO_A_PRO
pro_a_pro = pl.concat(
    [
        pl.read_excel(f"{RAW_PATH}PRO_A_PRO_2023.xlsx"),
        pl.read_excel(f"{RAW_PATH}PRO_A_PRO_2024.xlsx"),
    ]
).with_columns(
    pl.when(pl.col("Sous Sous Famille") == "0")
    .then(None)
    .otherwise(pl.col("Sous Sous Famille"))
    .alias("Sous Sous Famille")
)
pro_a_pro.write_parquet(f"{BRONZE_PATH}pro_a_pro.parquet")

# RELAIS_D_OR
relais_dor = pl.concat(
    [
        pl.read_excel(f"{RAW_PATH}RELAIS_D_OR_2023.xlsx"),
        pl.read_excel(f"{RAW_PATH}RELAIS_D_OR_2024.xlsx"),
    ]
).with_columns(
    [
        pl.when(pl.col(c) == "-").then(None).otherwise(pl.col(c)).alias(c)
        for c in [
            "Hiérarchie Tout Produit niv 2",
            "Hiérarchie Tout Produit niv 3",
            "Hiérarchie Tout Produit niv 4",
        ]
    ]
)
relais_dor.write_parquet(f"{BRONZE_PATH}relais_dor.parquet")

# SYSCO
sysco = (
    pl.read_csv(f"{RAW_PATH}SYSCO_pdt.txt", separator="\t")
    .with_columns(
        pl.col("Mois civil YY-MM")
        .str.split_exact("-", 1)
        .struct.rename_fields(["year", "month"])
        .alias("fields")
    )
    .unnest("fields")
    .with_columns([pl.col("year").cast(pl.String), pl.col("month").cast(pl.String)])
    .with_columns(
        pl.when(pl.col("year") == "23")
        .then(pl.lit("2023"))
        .when(pl.col("year") == "24")
        .then(pl.lit("2024"))
        .otherwise(pl.col("year"))
        .alias("year")
    )
    .with_columns(pl.col("CA Brut").cast(pl.Utf8).str.replace(" ", ""))
)
sysco.write_parquet(f"{BRONZE_PATH}sysco.parquet")

# TRANSGOURMET
transgourmet = (
    pl.read_excel(f"{RAW_PATH}TRANSGOURMET.xlsx")
    .with_columns(pl.col("Annee Mois").str.slice(-2).alias("Mois"))
    .with_columns(pl.col("Annee Mois").str.slice(0, length=4).alias("Annee"))
)
transgourmet.write_parquet(f"{BRONZE_PATH}transgourmet.parquet")

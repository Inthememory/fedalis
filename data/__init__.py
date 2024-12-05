import polars as pl

coup_de_pates = pl.read_excel("data/raw/COUP_DE_PATES.xlsx")

ds_restauration = pl.concat(
    [
        pl.read_excel("data/raw/DS_RESTAURATION_2023.xlsx"),
        pl.read_excel("data/raw/DS_RESTAURATION_2024.xlsx"),
    ]
)

ducreux = pl.concat(
    [
        pl.read_excel("data/raw/DUCREUX_2023.xlsx"),
        pl.read_excel("data/raw/DUCREUX_2024.xlsx"),
    ]
)

even = pl.read_excel("data/raw/EVEN.xlsx")

metro = (
    pl.read_csv("data/raw/METRO_pdt.csv", separator=";")
    .with_columns(pl.col("concatenated_gtin").str.split("|").alias("gtin"))
    .explode("gtin")
)

pomona = pl.concat(
    [
        pl.read_excel("data/raw/POMONA_ES_2023.xlsx", sheet_name="Extract"),
        pl.read_excel("data/raw/POMONA_ES_2024.xlsx", sheet_name="Extract"),
        pl.read_excel("data/raw/POMONA_PF_2023.xlsx", sheet_name="Extract"),
        pl.read_excel("data/raw/POMONA_PF_2024.xlsx", sheet_name="Extract"),
    ]
)

pro_a_pro = pl.concat(
    [
        pl.read_excel("data/raw/PRO_A_PRO_2023.xlsx"),
        pl.read_excel("data/raw/PRO_A_PRO_2024.xlsx"),
    ]
).with_columns(
    pl.when(pl.col("Sous Sous Famille") == "0")
    .then(None)
    .otherwise(pl.col("Sous Sous Famille"))
    .alias("Sous Sous Famille")
)

relais_dor = pl.concat(
    [
        pl.read_excel("data/raw/RELAIS_D_OR_2023.xlsx"),
        pl.read_excel("data/raw/RELAIS_D_OR_2024.xlsx"),
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

sysco = (
    pl.read_csv("data/raw/SYSCO_pdt.txt", separator="\t")
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

transgourmet = (
    pl.concat(
        [
            pl.read_excel("data/raw/transgourmet_2023.xlsx"),
            pl.read_excel("data/raw/transgourmet_2024.xlsx"),
        ]
    )
    .with_columns(pl.col("Annee Mois").str.slice(-2).alias("Mois"))
    .with_columns(pl.col("Annee Mois").str.slice(0, length=4).alias("Annee"))
)

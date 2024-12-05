# fedalis

## Presentation
This project enables the classification of products from different companies into a common nomenclature based on a correspondence table at the category and subcategory levels.

## How to run ?

```
python -m venv.venv
source/bin/activate
pip install -r requirements.txt

# Copy raw data
Copy raw data from Sharepoint [51. Fedalis > 2. POC Nomenclature > 02. DATA > RAW] (https://storeelectronicsystems.sharepoint.com/sites/teamemory/Shared%20Documents/Forms/AllItems.aspx?ga=1&isAscending=true&id=%2Fsites%2Fteamemory%2FShared%20Documents%2FOneDrive%20%2D%20teamemory%2F1%2E%20Clients%2F01%2E%20Retailers%2F51%2E%20Fedalis%2F2%2E%20POC%20Nomenclature%2F02%2E%20DATA%2FRAW&sortField=LinkFilename&viewid=41f51fdf%2D13de%2D429d%2Db472%2D09a5e1b1fca6) to local folder `data/raw`

# Copy correspondence tables
Copy and rename correspondence tables from Sharepoint [51. Fedalis > 2. POC Nomenclature > 02. DATA > 04. CONSTRUCTION NOMENCLATURE > 02. PRODUIT > AUDIT DATA MODELE X] (https://storeelectronicsystems.sharepoint.com/sites/teamemory/Shared%20Documents/Forms/AllItems.aspx?ga=1&isAscending=true&id=%2Fsites%2Fteamemory%2FShared%20Documents%2FOneDrive%20%2D%20teamemory%2F1%2E%20Clients%2F01%2E%20Retailers%2F51%2E%20Fedalis%2F2%2E%20POC%20Nomenclature%2F04%2E%20CONSTRUCTION%20NOMENCLATURE%2F02%2E%20PRODUIT&sortField=LinkFilename&viewid=41f51fdf%2D13de%2D429d%2Db472%2D09a5e1b1fca6) to local folder `data/`
- `mapping_model1.xlsx`
- `mapping_model2.csv`


# group 
python main_model1.py
python main_model2.py 
```

import sqlite3


def percentage(a, b):
    return a / b * 100


conn = sqlite3.connect("original_quarter.db")

cursor = conn.execute(
    "SELECT C.ID, COUNT(*) AS Total FROM Company AS C JOIN AssetsCategories AS AC ON AC.CompanyID = C.ID JOIN EquityLiabilitiesCategories AS ELC ON ELC.CompanyID = C.ID AND ELC.Date = AC.Date JOIN MarketValues MV ON MV.CompanyID = C.ID AND MV.[Period end] = ELC.Date GROUP BY C.ID HAVING Total > 1"
)

companies_with_needed_data = [row[0] for row in cursor]
# print(companies_with_needed_data)
# print(len(companies_with_needed_data))

cursor = conn.execute(
    "SELECT C.ID, MV.[Period end], MV.[Market value], AC.[Non-current assets], AC.[Current assets], AC.[Assets held for sale and discontinuing operations], AC.[Called up capital], AC.[Own shares], ELC.[Equity shareholders of the parent], ELC.[Non-controlling interests], ELC.[Non-current liabilities], ELC.[Current liabilities], ELC.[Liabilities related to assets held for sale and discontinued operations] FROM Company AS C JOIN AssetsCategories AS AC ON AC.CompanyID = C.ID JOIN EquityLiabilitiesCategories AS ELC ON ELC.CompanyID = C.ID AND ELC.Date = AC.Date JOIN MarketValues MV ON MV.CompanyID = C.ID AND MV.[Period end] = ELC.Date WHERE C.ID IN ({seq}) ORDER BY C.ID, MV.[Period end]".format(
        seq=','.join(['?'] * len(companies_with_needed_data))),
    companies_with_needed_data
)

data = [row for row in cursor]
# print(data)
# print(len(data))

final_data = []
for i in range(len(data)):
    company_id = data[i][0]
    period = data[i][1]
    market_value = data[i][2]
    assets = data[i][3:8]
    liabilities = data[i][8:13]
    total_assets = sum(assets)
    total_liabilities = sum(liabilities)

    row = [company_id, period, market_value]
    for asset in assets:
        row.append(percentage(asset, total_assets))
    for liability in liabilities:
        row.append(percentage(liability, total_liabilities))
    final_data.append(row)

# company_id, period, market_value, 5 x assets, 5 x liabilities
print(final_data)
print(len(final_data))

conn.close()

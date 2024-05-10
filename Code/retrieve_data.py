import sqlite3
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from tensorflow import keras
from sklearn.model_selection import train_test_split
from scikeras.wrappers import KerasRegressor

from Company import Company


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

conn.close()

temporary_data = []
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
    temporary_data.append(row)

# company_id, period, market_value, 5 x assets, 5 x liabilities
print(temporary_data)
print(len(temporary_data))

final_data = []

for i in range(1, len(temporary_data)):
    if temporary_data[i][0] != temporary_data[i - 1][0]:
        continue
    x = [temporary_data[i][0], temporary_data[i][1],
         (temporary_data[i][2] - temporary_data[i - 1][2]) / temporary_data[i - 1][2]]
    for j in range(3, len(temporary_data[i])):
        x.append(temporary_data[i][j] - temporary_data[i - 1][j])
    final_data.append(x)

print(final_data)
print(len(final_data))

df = pd.DataFrame(final_data, columns=["CompanyID", "Period", "MarketValue", "NonCurrentAssets", "CurrentAssets",
                                       "AssetsHeldForSaleAndDiscountinuingOperations", "CalledUpCapital", "OwnShares",
                                       "EquityShareholdersOfTheParent", "NonControllingInterests",
                                       "NonCurrentLiabilities",
                                       "CurrentLiabilities",
                                       "LiabilitiesRelatedToAssetsHeldForSaleAndDiscontinuedOperations"])

print(df)

df = df.drop(["CompanyID", "Period"], axis=1)
Q1 = df.quantile(0.1)
Q3 = df.quantile(0.9)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[~((df < lower_bound) | (df > upper_bound)).any(axis=1)]
X = df.drop(["MarketValue"], axis=1)
y = df["MarketValue"]

print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.07)

n_nodes = [16, 8, 4, 2]
model = keras.models.Sequential()
model.add(keras.layers.Dense(n_nodes[0], input_dim=X.shape[1], activation="relu"))
for i in range(1, len(n_nodes)):
    model.add(keras.layers.Dense(n_nodes[i], input_dim=n_nodes[i - 1], activation="relu"))
    model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(1))
model.compile(
    optimizer="adam",
    loss="mean_squared_error",
    metrics=["mean_absolute_error"],
)

inputs = keras.Input(shape=(X.shape[1],))

x = keras.layers.Dense(n_nodes[0], activation="relu")(inputs)

for i in range(1, len(n_nodes)):
    x = keras.layers.Dense(n_nodes[i], activation="relu")(x)
    x = keras.layers.Dropout(0.1)(x)

outputs = keras.layers.Dense(1)(x)

model = keras.Model(inputs=inputs, outputs=outputs)

learning_rate = 0.00001  # You can adjust this value as needed

optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

model.compile(
    optimizer=optimizer,
    loss="mean_squared_error",
    metrics=["mean_absolute_error"],
)
#
model.fit(
    X_train,
    y_train,
    epochs=100,
    validation_data=(X_val, y_val),
    validation_freq=10,
    # callbacks=[
    #     keras.callbacks.EarlyStopping(monitor="mean_absolute_error", patience=10),
    # ],
    verbose=True
)

predictions = model.predict(X_test)
for i in range(len(predictions)):
    print(predictions[i], y_test.iloc[i])
#
# mape_loss = keras.metrics.mean_absolute_percentage_error(y_test, predictions)
# mse_loss = keras.metrics.mean_squared_error(y_test, predictions)
#
# print(mape_loss, mse_loss)

# dbscan = DBSCAN(eps=5, min_samples=2)
# dbscan.fit(X_train)
# test_labels = dbscan.fit_predict(X_test)
#
# test_outliers = X_test[test_labels == -1]
#
# print("Number of outliers in test data:", len(test_outliers))

isolation_forest = IsolationForest()
isolation_forest.fit(X_train)
test_outliers = isolation_forest.predict(X_test)

test_outliers = X_test[test_outliers == -1]

print("Number of outliers in test data:", len(test_outliers), "out of:", len(X_test))


# generated_companies = []
# for i in range(100):
#     generated_companies.append(Company([for i in range()]))
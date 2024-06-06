import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split


class StructureChange:
    def __init__(self, final_data):
        df = pd.DataFrame(final_data,
                          columns=["CompanyID", "Period", "MarketValue", "NonCurrentAssets", "CurrentAssets",
                                   "AssetsHeldForSaleAndDiscountinuingOperations", "CalledUpCapital", "OwnShares",
                                   "EquityShareholdersOfTheParent", "NonControllingInterests",
                                   "NonCurrentLiabilities",
                                   "CurrentLiabilities",
                                   "LiabilitiesRelatedToAssetsHeldForSaleAndDiscontinuedOperations"])

        df = df.drop(["CompanyID", "Period"], axis=1)
        Q1 = df.quantile(0.1)
        Q3 = df.quantile(0.9)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[~((df < lower_bound) | (df > upper_bound)).any(axis=1)]
        self.X = df.drop(["MarketValue"], axis=1)
        self.y = df["MarketValue"]

    def train_model(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.15)

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.07)

        n_nodes = [128, 64, 32, 16]

        inputs = keras.Input(shape=(self.X.shape[1],))

        x = keras.layers.Dense(n_nodes[0], activation="relu")(inputs)

        for i in range(1, len(n_nodes)):
            x = keras.layers.Dense(n_nodes[i], activation="relu")(x)
            x = keras.layers.Dropout(0.1)(x)

        outputs = keras.layers.Dense(1)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        learning_rate = 0.000001

        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        model.compile(
            optimizer=optimizer,
            loss="mean_squared_error",
            metrics=["mean_absolute_error"],
        )

        model.fit(
            X_train,
            y_train,
            epochs=100,
            validation_data=(X_val, y_val),
            validation_freq=10,
            validation_split=0.2,
            # callbacks=[
            #     keras.callbacks.EarlyStopping(monitor="mean_absolute_error", patience=10),
            # ],
            verbose=True
        )

        predictions = model.predict(X_test)
        for i in range(len(predictions)):
            print(predictions[i], y_test.iloc[i])

        loss = model.evaluate(X_test, y_test)
        print("Test Loss:", loss)

        return model

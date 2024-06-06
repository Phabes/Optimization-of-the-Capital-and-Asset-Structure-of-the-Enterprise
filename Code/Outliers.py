import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split


class Outliers:
    def __init__(self, percentage_structure_data):
        self.df_outliers = pd.DataFrame(percentage_structure_data,
                                        columns=["CompanyID", "Period", "MarketValue", "NonCurrentAssets",
                                                 "CurrentAssets",
                                                 "AssetsHeldForSaleAndDiscountinuingOperations", "CalledUpCapital",
                                                 "OwnShares",
                                                 "EquityShareholdersOfTheParent", "NonControllingInterests",
                                                 "NonCurrentLiabilities",
                                                 "CurrentLiabilities",
                                                 "LiabilitiesRelatedToAssetsHeldForSaleAndDiscontinuedOperations"])

        self.df_outliers = self.df_outliers.drop(["CompanyID", "Period", "MarketValue"], axis=1)

    def train_model(self, test_size=0.1, contamination=0.01):
        X_outliers_train, X_outliers_test = train_test_split(self.df_outliers, test_size=test_size)

        isolation_forest = IsolationForest(contamination=contamination)
        isolation_forest.fit(X_outliers_train)
        test_outliers = isolation_forest.predict(X_outliers_test)

        test_outliers = X_outliers_test[test_outliers == -1]

        print("Number of outliers in test data:", len(test_outliers), "out of:", len(X_outliers_test))

        # dbscan = DBSCAN(eps=10, min_samples=2)
        # dbscan.fit(X_train)
        # test_labels = dbscan.fit_predict(X_test)
        #
        # test_outliers = X_test[test_labels == -1]
        #
        # print("Number of outliers in test data:", len(test_outliers))

        return isolation_forest

    def get_mean_values(self):
        return self.df_outliers.mean()

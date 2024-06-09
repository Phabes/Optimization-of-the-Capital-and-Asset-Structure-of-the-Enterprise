import random

import pandas as pd

from Company import Company
from generate_company_structure import generate_structure_mean


class EvolutionaryAlgorithm:
    def __init__(self, number_of_companies, means, outliers_model, structure_change_model):
        self.outliers_model = outliers_model
        self.structure_change_model = structure_change_model
        self.generated_companies = []

        while len(self.generated_companies) != number_of_companies:
            assets = generate_structure_mean(means[:5], 5)
            liabilities = generate_structure_mean(means[5:], 5)
            company = Company(*assets, *liabilities)
            if outliers_model.predict(company.to_dataframe())[0] == -1:
                # if dbscan.fit_predict(company.to_dataframe())[0] == -1:
                continue
            self.generated_companies.append(company)
            print("Company structure generated. Total structures:", len(self.generated_companies))

    def check_generated_structures(self):
        for company in self.generated_companies:
            values = company.to_dataframe().values[0]
            print(["{:.2f}".format(num) for num in values])

    def generate_gradient(self):
        values = []
        for _ in range(4):
            values.append(random.uniform(-1, 1))
        fifth_value = -sum(values)
        if -1 <= fifth_value <= 1:
            values.append(fifth_value)
        else:
            return self.generate_gradient()
        return values

    def generate_offspring(self):
        # for each company generate offspring (by modifying parent) and check if offspring is not outlier (generate until not outlier)
        new_companies = []
        for company in self.generated_companies:
            gradient_assets = self.generate_gradient()
            gradient_liabilities = self.generate_gradient()
            df = pd.DataFrame([[*gradient_assets, *gradient_liabilities]],
                              columns=["NonCurrentAssets", "CurrentAssets",
                                       "AssetsHeldForSaleAndDiscountinuingOperations", "CalledUpCapital", "OwnShares",
                                       "EquityShareholdersOfTheParent", "NonControllingInterests",
                                       "NonCurrentLiabilities",
                                       "CurrentLiabilities",
                                       "LiabilitiesRelatedToAssetsHeldForSaleAndDiscontinuedOperations"])
            predictions = self.structure_change_model.predict(df)
            prediction = predictions[0][0]
            if prediction > 0:
                new_company_values = [x + y for x, y in
                                      zip(company.to_array(), [*gradient_assets, *gradient_liabilities])]
                child_company = Company(*new_company_values)
                if self.outliers_model.predict(child_company.to_dataframe())[0] != -1:
                    new_companies.append(child_company)
                    continue
            new_companies.append(company)

        self.generated_companies = new_companies

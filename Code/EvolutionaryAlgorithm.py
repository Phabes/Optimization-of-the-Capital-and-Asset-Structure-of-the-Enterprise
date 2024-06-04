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
            print(values, sum(values[:5]), sum(values[5:]))

    def generate_offspring(self):
        # for each company generate offspring (by modifying parent) and check if offspring is not outlier (generate until not outlier)
        pass
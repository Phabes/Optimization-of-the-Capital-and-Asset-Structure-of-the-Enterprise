from generate_random_structure import generate_numbers_summing_to_100
from Company import Company
import pandas as pd

n = 100
generated_companies = []

for i in range(n):
    assets = generate_numbers_summing_to_100(5)
    liabilities = generate_numbers_summing_to_100(5)
    company = Company(*assets, *liabilities)
    generated_companies.append(company)

epochs = 100

data = pd.DataFrame()

for company in generated_companies:
    data = pd.concat([data, company.to_fraction_dataframe()], axis=0)

print(data)

for company in generated_companies:
    x = company.generate_random_modification()
    print(x)
from Outliers import Outliers
from StructureChange import StructureChange
from EvolutionaryAlgorithm import EvolutionaryAlgorithm
from retrieve_data import get_raw_sql_data
from data_modification import get_percentage_structure, get_structure_changes

data = get_raw_sql_data()

# company_id, period, market_value, 5 x assets, 5 x liabilities
percentage_structure_data = get_percentage_structure(data)

outliers = Outliers(percentage_structure_data)
isolation_forest = outliers.train_model()
means = outliers.get_mean_values()

final_data = get_structure_changes(percentage_structure_data)

structure_change = StructureChange(final_data)
structure_change_model = structure_change.train_model()

number_of_companies = 10
evolutionary_algorithm = EvolutionaryAlgorithm(number_of_companies, means, isolation_forest, structure_change_model)

evolutionary_algorithm.check_generated_structures()

epochs = 100
for i in range(epochs):
    evolutionary_algorithm.generate_offspring()

evolutionary_algorithm.check_generated_structures()


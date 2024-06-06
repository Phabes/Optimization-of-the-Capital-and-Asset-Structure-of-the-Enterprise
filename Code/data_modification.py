from calculations import percentage, only_positive_values


def get_percentage_structure(data):
    percentage_structure_data = []

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

        # if only_positive_values(assets) and only_positive_values(liabilities): // if we accept only positive values
        percentage_structure_data.append(row)

    return percentage_structure_data


def get_structure_changes(data):
    final_data = []

    for i in range(1, len(data)):
        if data[i][0] != data[i - 1][0]:
            continue
        x = [data[i][0], data[i][1],
             (data[i][2] - data[i - 1][2]) / data[i - 1][2]]
        for j in range(3, len(data[i])):
            x.append(data[i][j] - data[i - 1][j])
        final_data.append(x)

    return final_data

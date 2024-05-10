import pandas as pd


class Company:
    def __init__(self, non_current_assets, current_assets, assets_held_for_sale_and_discontinued_operations,
                 called_up_capital, own_shares, equity_shareholders_of_the_parent, non_controlling_interests,
                 non_current_liabilities, current_liabilities, liabilities_related_to_assets_held_for_sale_and_discontinued_operations):
        self.non_current_assets = non_current_assets
        self.current_assets = current_assets
        self.assets_held_for_sale_and_discontinued_operations = assets_held_for_sale_and_discontinued_operations
        self.called_up_capital = called_up_capital
        self.own_shares = own_shares
        self.equity_shareholders_of_the_parent = equity_shareholders_of_the_parent
        self.non_controlling_interests = non_controlling_interests
        self.non_current_liabilities = non_current_liabilities
        self.current_liabilities = current_liabilities
        self.liabilities_related_to_assets_held_for_sale_and_discontinued_operations = liabilities_related_to_assets_held_for_sale_and_discontinued_operations

    def to_dataframe(self):
        data = {
            "NonCurrentAssets": [self.non_current_assets],
            "CurrentAssets": [self.current_assets],
            "AssetsHeldForSaleAndDiscountinuingOperations": [self.assets_held_for_sale_and_discontinued_operations],
            "CalledUpCapital": [self.called_up_capital],
            "OwnShares": [self.own_shares],
            "EquityShareholdersOfTheParent": [self.equity_shareholders_of_the_parent],
            "NonControllingInterests": [self.non_controlling_interests],
            "NonCurrentLiabilities": [self.non_current_liabilities],
            "CurrentLiabilities": [self.current_liabilities],
            "LiabilitiesRelatedToAssetsHeldForSaleAndDiscontinuedOperations": [
                self.liabilities_related_to_assets_held_for_sale_and_discontinued_operations]
        }
        return pd.DataFrame(data)

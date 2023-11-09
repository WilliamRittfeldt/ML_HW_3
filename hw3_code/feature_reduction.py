import pandas as pd
import statsmodels.api as sm
from typing import List


class FeatureReduction(object):
    def __init__(self):
        pass

    @staticmethod
    def forward_selection(
        data: pd.DataFrame, target: pd.Series, significance_level: float = 0.05
    ) -> List[str]:  # 9 pts
        """
        Args:
            data: (pandas data frame) contains the feature matrix
            target: (pandas series) represents target feature to search to generate significant features
            significance_level: (float) threshold to reject the null hypothesis
        Return:
            forward_list: (python list) contains significant features. Each feature
            name is a string
        """
        
        forward_list = []
        remaining_features = data.columns.tolist()
        while len(remaining_features) > 0:
            new_p_value = pd.Series(index=remaining_features, dtype=float)
            for new_column in remaining_features:
                model = sm.OLS(target, sm.add_constant(data[forward_list + [new_column]])).fit()
                new_p_value[new_column] = model.pvalues[new_column]
            min_p_value = new_p_value.min()
            if min_p_value < significance_level:
                forward_list.append(new_p_value.idxmin())
                remaining_features.remove(new_p_value.idxmin())
            else:
                break
            
        return forward_list
        
        # initial_features = data.columns.tolist()
        # forward_list = []
        # while len(initial_features) > 0:
        #     remaining_features = list(set(initial_features) - set(forward_list))
        #     new_pval = pd.Series(index=remaining_features, dtype=float)
        #     for new_column in remaining_features:
        #         model = sm.OLS(target, sm.add_constant(data[forward_list + [new_column]])).fit()
        #         new_pval[new_column] = model.pvalues[new_column]
        #     min_p_value = new_pval.min()
        #     if min_p_value < significance_level:
        #         forward_list.append(new_pval.idxmin())
        #     else:
        #         break
        # return forward_list

    @staticmethod
    def backward_elimination(
        data: pd.DataFrame, target: pd.Series, significance_level: float = 0.05
    ) -> List[str]:  # 9 pts
        """
        Args:
            data: (pandas data frame) contains the feature matrix
            target: (pandas series) represents target feature to search to generate significant features
            significance_level: (float) threshold to reject the null hypothesis
        Return:
            backward_list: (python list) contains significant features. Each feature
            name is a string
        """
        backward_list = data.columns.tolist()
        while len(backward_list) > 0:
            features_with_constant = sm.add_constant(data[backward_list])
            model = sm.OLS(target, features_with_constant).fit()
            p_values = model.pvalues.iloc[1:]  # Exclude the intercept
            max_p_value = p_values.max()
            if max_p_value > significance_level:
                excluded_feature = p_values.idxmax()
                backward_list.remove(excluded_feature)
            else:
                break
        return backward_list
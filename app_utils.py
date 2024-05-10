
import pandas as pd

def featurize_formula(formula):
    """
    Parameters
    ----------
    formula : str
        Chemical formula of the material.
    Returns
    -------
    np.array
        2d array of Features of the material.
    """
    from matminer.featurizers import composition as cf
    from matminer.featurizers.base import MultipleFeaturizer
    from matminer.featurizers.conversions import StrToComposition
    try:
        df = pd.DataFrame({"formula": [formula]})
        df = StrToComposition().featurize_dataframe(df, "formula")
        feature_calculators = MultipleFeaturizer([cf.Stoichiometry(),
                                                  cf.ElementProperty.from_preset("magpie"),
                                                  cf.ValenceOrbital(props=['avg']),
                                                  cf.IonProperty(fast=True)])
        data = feature_calculators.featurize_dataframe(df, col_id=['composition'])
        feature_labels = feature_calculators.feature_labels()
        return data[feature_labels]
    except:
        "Formula cannot be featurized. Please input a valid formula."

def load_matbench_data(dataset):
    """
    Parameters
    ----------
    dataset : str
        Name of the dataset in matminer.datasets
    Returns
    ----------
    pd.DataFrame
        DataFrame of the dataset.
    """
    from matminer.datasets import get_all_dataset_info, load_dataset
    # print(get_all_dataset_info(dataset))
    df = load_dataset(dataset) # composition/regression
    return df

def featurize_df_composition(df, y_drop="gap expt"):
    """
    Parameters
    ----------

    """
    from matminer.featurizers.conversions import StrToComposition
    from matminer.featurizers import composition as cf
    from matminer.featurizers.base import MultipleFeaturizer
    # have to rename the composition column to formula
    df.rename(columns={"composition": "formula"}, inplace=True)
    df = StrToComposition().featurize_dataframe(df, "formula")
    # composition col is the input taken by matminer Featurizer classes
    feature_calculators = MultipleFeaturizer([cf.Stoichiometry(),
                                              cf.ElementProperty.from_preset("magpie"),
                                              cf.ValenceOrbital(props=['avg']),
                                              cf.IonProperty(fast=True)])
    # feature names
    feature_labels = feature_calculators.feature_labels()
    # drop expt_gap column
    data = feature_calculators.featurize_dataframe(df.drop(columns=[y_drop]), col_id=['composition'])
    # drop formula and composition columns
    feature_data = data[feature_labels]
    print(f'Generated {len(feature_labels)} features')
    print(feature_data.shape)
    return feature_data
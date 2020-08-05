import pandas as pd
import seaborn as sns

# --------------
# Misc Functions
# --------------


def flatten_list_of_lists(list_of_lists: list, remove_non_lists: bool = True) -> list:
    """
    Flatten a list of lists
    
    Arguments
    ----------
    remove_non_lists - bool
        Option to remove non list elements
    
    Returns
    ---------
    Flattened list
    """

    if remove_non_lists:
        list_of_lists = [lst for lst in list_of_lists if isinstance(lst, list)]

    return [val for lst in list_of_lists for val in lst]


def dict_from_list(lst: list, value: any) -> dict:
    """
    Create a dictionary with keys from a `lst` and all values equal to `value`
    """

    return {key: value for key in lst}


# ------------------
# Cleaning Functions
# ------------------


def convert_dtypes(
    df: pd.DataFrame,
    str_cols: list = None,
    int_cols: list = None,
    float_cols: list = None,
    cat_cols: list = None,
    dt_cols: dict = None,
) -> pd.DataFrame:
    """
    Convert pandas DataFrame columns to specific dtypes
    
    Convert column dtypes given lists/dicts of columns
    
    Arguments
    ----------
    str_cols : list
        column names to convert to str
    int_cols : list
        column names to convert to int
    float_cols : list
        column names to convert to float
    cat_cols : list
        column names to convert to pd.category
    dt_cols : dict
        datetime columns and format for datetime parsing
    
    Returns
    ---------
    df : pd.DataFrame
        DataFrame with columns converted
    """

    df_typed = df.copy()

    str_dict, int_dict, float_dict, cat_dict = {}, {}, {}, {}

    if str_cols:
        str_dict = dict_from_list(str_cols, str)
    if int_cols:
        int_dict = dict_from_list(int_cols, int)
    if float_cols:
        float_dict = dict_from_list(float_cols, float)
    if cat_cols:
        cat_dict = dict_from_list(cat_cols, "category")

    df_typed = df_typed.astype({**str_dict, **int_dict, **float_dict, **cat_dict})

    if dt_cols:
        for col, form in dt_cols.items():
            df_typed[col] = pd.to_datetime(df_typed[col], format=form)

    return df_typed


def snake_case_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Snake case DataFrame columns

    i.e. [Start Date, Final Value] -> [start_date, final_vale]
    """

    df_out = df.copy()
    df_out.columns = [col.lower().replace(" ", "_") for col in df_out.columns]

    return df_out


# -------------
# EDA Functions
# -------------


def perc_estimator(x, df):
    """
    Estimator used to create normed count plot with sns.barplot()

    Example:
        ax = sns.barplot(data=df, x='col_name', y='col_name', estimator=lambda x: perc_estimator(x, df))
        ax.set_ylabel('percent')
    
    See: https://github.com/mwaskom/seaborn/issues/1027
    """

    return len(x) / len(df) * 100


if __name__ == "__main__":
    pass

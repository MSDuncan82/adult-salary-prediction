import pandas as pd
from src.helper_funcs import convert_dtypes


class Preprocessor:
    columns = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education_num",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital_gain",
        "capital_loss",
        "hours_per_week",
        "native_country",
        "salary_class",
    ]
    int_cols = [
        "salary_class",
        "age",
        "fnlwgt",
        "education_num",
        "is_cap",
        "hours_per_week",
    ]
    cat_cols = [
        "workclass",
        "marital_status",
        "sex",
        "occupation",
        "relationship",
    ]
    drop_cols = ["capital_gain", "capital_loss", "native_country", "education", "race"]

    dtypes_dict = {"int_cols": int_cols, "cat_cols": cat_cols}

    def clean(self, df):

        self.df_raw = df

        df = self._target_to_int(df)
        df = self._add_is_cap(df)
        df = df.drop(self.drop_cols, axis=1)
        df = convert_dtypes(df, **self.dtypes_dict)

        self.df = df
        self.y_ser = df.salary_class
        self.X_df = df.drop('salary_class', axis=1)

        return df

    def _add_is_cap(self, df):
        """
        Add `is_cap` column which represents if row has capital_gain 
        or capital loss other than 0
        """

        df_out = df.copy()
        is_cap = (df[["capital_gain", "capital_loss"]] != 0).any(axis=1)

        df_out["is_cap"] = is_cap

        return df_out

    def _target_to_int(self, df):
        """
        Map salary_class column from " <=50K" to 0 and " >50K" to 1 
        """
        df_out = df.copy()

        df_out["salary_class"] = df.salary_class.str.strip(' .').replace("<=50K", 0).replace(
            ">50K", 1
        )

        return df_out


if __name__ == "__main__":
    preproc = Preprocessor()
    df = pd.read_csv("data/raw/adult.data", header=None, names=preproc.columns)

    preproc.clean(df)
    import ipdb; ipdb.set_trace()
    

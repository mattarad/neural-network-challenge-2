import pandas as pd

from typing import Union
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def label_encode_columns(columns, df) -> Union[pd.DataFrame, dict[str, LabelEncoder]]:
    """ takes in a list of columns and the data frame to encode, returns the encoded dataframe and a dictionary of the LabelEncoders used
        params:
            columns: a list of columns to encode using a LabelEncoder
            df: the data frame you want to encode on
    """
    label_encoders = {}
    for col in columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    return (df, label_encoders)

def label_encode_test_df(label_encoders_dict, df) -> pd.DataFrame:
    """ takes in a dictionary of label_encoders, applies the label encoder on each column and returns the new dataframe
        params:
            label_encoder_dict: this is dictionary of LabelEncoders where the key is the column to encode, and the
            value is the encoder that was used on the train data set.
        df: the dataframe you are encoding.

        returns an encoded dataframe
    """
    # print(label_encoders_dict.items())
    for col, encoder in label_encoders_dict.items():
        # df[col] = df[col].apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)
        df[col] = encoder.transform(df[col])
    return df


def ohe_columns(columns, df) -> Union[pd.DataFrame, dict[str, OneHotEncoder]]:
    """ takes in a list of columns and the data frame to one hot encode, returns the encoded dataframe
        and a dictionary of the ohe_encoders
        params:
            columns: a list of columns to encode using a one hot encoder
            df: the data frame you want to encode on
    """

    ohe_encoders = {}
    for col in columns:
        ohe = OneHotEncoder(sparse_output=False)
        ohe.fit(df[[col]])
        encoded = ohe.transform(df[[col]])
        encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out([col]))
        df = pd.concat([df.reset_index(drop=True), encoded_df], axis=1).drop(columns=[col])
        ohe_encoders[col] = ohe
    return (df, ohe_encoders)

def ohe_columns_test(ohe_encoders, df) -> pd.DataFrame:
    """ takes in a dictionary of one hot encoders, applies the one hot encoder on each column and returns the new dataframe
        params:
            ohe_encoders: this is dictionary of OneHotEncoders where the key is the column to encode, and the
            value is the encoder that was used on the train data set.
        df: the dataframe you are encoding.

        returns an encoded dataframe
    """
    for col, encoder in ohe_encoders.items():
        encoded = encoder.transform(df[[col]])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([col]))
        df = pd.concat([df.reset_index(drop=True), encoded_df], axis=1).drop(columns=[col])
    return df
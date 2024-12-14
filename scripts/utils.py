from typing import List
from pandas.core.frame import DataFrame


def split_df(df: DataFrame, split_size=5000) -> List[DataFrame]:
    df_res = []
    one_index = df.shape[0] // split_size
    if one_index * split_size < df.shape[0]:
        one_index += 1
    for i in range(one_index):
        if i != one_index - 1:
            df_res.append(df.iloc[i * split_size : (i + 1) * split_size])
        else:
            df_res.append(df.iloc[i * split_size :])

    return df_res

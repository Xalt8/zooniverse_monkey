import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from functions import get_image_df, train_test_split_df


total_df = get_image_df()
total_df = train_test_split_df(total_df=total_df)


def test_df_index() -> None:
    """ Test to ensure that the index values are continuous from 0 to len(total_df)"""
    global total_df
    for i,j in zip(range(len(total_df)), total_df.index.values):
        assert i == j


def test_train_test_split() -> None:
    """ Test to make sure that each image file instance has been 
        categorized only into Train, Test, Val """
    global total_df
    check = []
    for file_name in total_df['Image File'].unique():
        image_file_df = total_df[total_df['Image File'] == file_name]
        if len(image_file_df['train_test_val'].unique()) == 1:
            continue
        else:
            check.append(file_name)
    assert len(check) == 0


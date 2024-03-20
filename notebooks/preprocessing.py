import argparse
import os
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelBinarizer, KBinsDiscretizer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.compose import ColumnTransformer

from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action="ignore", category=DataConversionWarning)

def print_shape(df):
    negative_examples, positive_examples = np.bincount(df["target"])
    print(
        "Data shape: {}, {} positive examples, {} negative examples".format(
            df.shape, positive_examples, negative_examples
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-test-split-ratio", type=float, default=0.3)
    args, _ = parser.parse_known_args()

    print("Received arguments {}".format(args))

    input_data_path = os.path.join("/opt/ml/processing/input", "df_train.csv")

    print("Reading input data from {}".format(input_data_path))
    df_train = pd.read_csv(input_data_path)
    cat_cols = list(df_train.select_dtypes(include='object').columns)
    num_cols = list(df_train.select_dtypes(include='float64').columns)
    int_cols = list(df_train.select_dtypes(include='int64').columns)
    df_train.drop_duplicates(inplace=True)
    
    # handle missing data
    for cat in cat_cols: 
        df_train[cat] = df_train[cat].fillna('sin dato')
    
    for num in num_cols: 
        df_train[num] = df_train[num].fillna(0.0)
    
    for i in int_cols: 
        df_train[i] = df_train[i].fillna(0)
    
    negative_examples, positive_examples = np.bincount(df_train["target"])
    print_shape(df_train)
    
    # Create transformers for categorical and numerical features
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse=False)
    numerical_transformer = StandardScaler()

    # Create the column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, cat_cols),
            ('num', numerical_transformer, num_cols)
        ]
    )
    
    
    
    print("Running preprocessing and feature engineering transformations")
    df_train_ = preprocessor.fit_transform(df_train.drop("target", axis=1))
    print("Train data shape after preprocessing: {}".format(df_train_.shape))
       
    split_ratio = args.train_test_split_ratio
    print("Splitting data into train and test sets with ratio {}".format(split_ratio))
    X_train, X_test, y_train, y_test = train_test_split(df_train_, 
                                                        df_train["target"], 
                                                        test_size=split_ratio, 
                                                        random_state=2023)
    
    print("Splitting data into train and validation sets with ratio 0.15")
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, 
                                                        y_train, 
                                                        test_size=0.15, 
                                                        random_state=2023)
    
    train = pd.concat([pd.Series(y_train, index=y_train.index,
                             name='target', dtype=int), pd.DataFrame(X_train, index=y_train.index)], axis=1)
    validation = pd.concat([pd.Series(y_valid, index=y_valid.index,
                            name='target', dtype=int), pd.DataFrame(X_valid, index=y_valid.index)], axis=1)
    test = pd.concat([pd.Series(y_test, index=y_test.index,
                            name='target', dtype=int), pd.DataFrame(X_test, index=y_test.index)], axis=1)
    
    negative_examples, positive_examples = np.bincount(train["target"])
    print(
        "Train data after spliting: {}, {} positive examples, {} negative examples, {} churn rate".format(
            train.shape, positive_examples, negative_examples, round(100*positive_examples/(positive_examples+negative_examples),2) 
        )
    )

    negative_examples, positive_examples = np.bincount(validation["target"])
    print(
        "Validation data after spliting: {}, {} positive examples, {} negative examples, {} churn rate".format(
            validation.shape, positive_examples, negative_examples, round(100*positive_examples/(positive_examples+negative_examples),2) 
        )
    )
    
    negative_examples, positive_examples = np.bincount(test["target"])
    print(
        "Test data after spliting: {}, {} positive examples, {} negative examples, {} churn rate".format(
            test.shape, positive_examples, negative_examples, round(100*positive_examples/(positive_examples+negative_examples),2) 
        )
    )

    train_output_path = os.path.join("/opt/ml/processing/train", "train.csv")
    validation_output_path = os.path.join("/opt/ml/processing/validation", "validation.csv")
    test_output_path = os.path.join("/opt/ml/processing/test", "test.csv")
    
    print("Saving train set to {}".format(train_output_path))
    train.to_csv(train_output_path, index=False)
    
    print("Saving validation set to {}".format(validation_output_path))
    validation.to_csv(validation_output_path, index=False)

    print("Saving test set to {}".format(test_output_path))
    test.to_csv(test_output_path, index=False)
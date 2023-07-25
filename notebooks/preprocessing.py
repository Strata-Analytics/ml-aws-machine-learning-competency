import argparse
import os
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelBinarizer, KBinsDiscretizer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.compose import make_column_transformer

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

    input_data_path = os.path.join("s3://aws-machine-learning-competency/sagemaker/sample/processed_data/", "df_train.csv")

    print("Reading input data from {}".format(input_data_path))
    df_train = pd.read_csv(input_data_path)
    cat_cols = list(df_train.select_dtypes(include='object').columns)
    num_cols = list(df_train.select_dtypes(include='float64').columns)
    int_cols = list(df_train.select_dtypes(include='int64').columns)
    df_train.drop_duplicates(inplace=True)
    
    negative_examples, positive_examples = np.bincount(df_train["target"])
    print_shape(df_train)
    
    split_ratio = args.train_test_split_ratio
    print("Splitting data into train and test sets with ratio {}".format(split_ratio))
    X_train, X_test, y_train, y_test = train_test_split(df_train.drop("target", axis=1), 
                                                        df_train["target"], 
                                                        test_size=split_ratio, 
                                                        random_state=2023)

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
    train_features = preprocessor.fit_transform(X_train)
    test_features = preprocessor.transform(X_test)
    print("Train data shape after preprocessing: {}".format(train_features.shape))
    print("Test data shape after preprocessing: {}".format(test_features.shape))

    train_features_output_path = os.path.join("/opt/ml/processing/train", "train_features.csv")
    train_labels_output_path = os.path.join("/opt/ml/processing/train", "train_labels.csv")

    test_features_output_path = os.path.join("/opt/ml/processing/test", "test_features.csv")
    test_labels_output_path = os.path.join("/opt/ml/processing/test", "test_labels.csv")

    print("Saving training features to {}".format(train_features_output_path))
    pd.DataFrame(train_features).to_csv(train_features_output_path, header=False, index=False)

    print("Saving test features to {}".format(test_features_output_path))
    pd.DataFrame(test_features).to_csv(test_features_output_path, header=False, index=False)

    print("Saving training labels to {}".format(train_labels_output_path))
    y_train.to_csv(train_labels_output_path, header=False, index=False)

    print("Saving test labels to {}".format(test_labels_output_path))
    y_test.to_csv(test_labels_output_path, header=False, index=False)
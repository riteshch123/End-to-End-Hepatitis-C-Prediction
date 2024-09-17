import pandas as pd
from data_transformation import create_pipeline
from model_pipeline import model_pipeline
import joblib


def main():
    # Load the dataset
    # Adjust path as necessary
    dataset_path = "https://raw.githubusercontent.com/Riteshch-123/Data-Science-engg/main/HepatitisCdata.csv"
    df = pd.read_csv(dataset_path)

    # Define the target column if it's not part of the transformations
    target_column = 'Category'  # Adjust as necessary
    X = df.drop('Category', axis=1)
    y = df['Category']

    # Encode the target variable to binary
    y = y.map({
        '0=Blood Donor': 0,
        '0s=suspect Blood Donor': 0,
        '1=Hepatitis': 1,
        '2=Fibrosis': 1,
        '3=Cirrhosis': 1
    })

    # Create the pipeline
    pipeline = create_pipeline()
    # Apply the pipeline to features only
    X_processed = pipeline.fit_transform(X)
    joblib.dump(pipeline, 'pipeline.pkl')

    # Ensure that 'y' contains only binary values
    print("Unique values in target variable after encoding:", y.unique())
    # Continue with further processing or model training
    model_pipeline(X_processed, y)


if __name__ == '__main__':
    main()

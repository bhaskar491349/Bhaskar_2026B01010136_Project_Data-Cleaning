import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates and irrelevant columns, handle missing values, and add derived features."""
    df = df.drop_duplicates().copy()

    
    df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

    
    df['Age'] = df['Age'].fillna(
        df.groupby(['Pclass', 'Sex'])['Age'].transform('median')
    )

    
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype('int8')

    return df


def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numerical types and convert repeated text columns to category."""
    int8_cols = ['Survived', 'Pclass', 'SibSp', 'Parch', 'FamilySize', 'IsAlone']
    float32_cols = ['Age', 'Fare']
    category_cols = ['Sex', 'Embarked']

    df[int8_cols] = df[int8_cols].astype('int8')
    df[float32_cols] = df[float32_cols].astype('float32')
    df[category_cols] = df[category_cols].astype('category')

    return df


def main():
    input_file = 'Titanic-Dataset.csv'
    output_file = 'Titanic-Cleaned.csv'

    
    raw_df = pd.read_csv(input_file)
    raw_memory = raw_df.memory_usage(deep=True).sum() / 1024

    
    df = clean_data(raw_df)
    df = optimize_memory(df)

    
    clean_memory = df.memory_usage(deep=True).sum() / 1024
    memory_reduction = ((raw_memory - clean_memory) / raw_memory) * 100

    print(f"Original Shape: {raw_df.shape}")
    print(f"Cleaned Shape:  {df.shape}")
    print(f"Missing Values: {df.isnull().sum().sum()}")
    print(f"Memory Before:  {raw_memory:.2f} KB")
    print(f"Memory After:   {clean_memory:.2f} KB ({memory_reduction:.2f}% reduction)")

    
    df.to_csv(output_file, index=False)
    print(f"\nCleaned dataset saved successfully to '{output_file}'.")


if __name__ == '__main__':
    main()

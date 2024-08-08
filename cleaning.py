import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from config import get_config

THRESHOLD = 30

def remove_and_cast(data):
    new_df = data.drop(['manufacturer','model','version','fuel_date','fuel_type'],axis=1)
    if 'fuel_note' in new_df.columns:
        new_df.drop('fuel_note',axis=1, inplace=True)

    new_df['city'] = new_df['city'].astype('object')
    new_df['motor_way'] = new_df['motor_way'].astype('object')
    new_df['country_roads'] = new_df['country_roads'].astype('object')
    new_df['A/C'] = new_df['A/C'].astype('object')
    new_df['park_heating'] = new_df['park_heating']
    new_df['trip_distance(km)']= new_df['trip_distance(km)'].astype('float')

    if new_df['trip_distance(km)'].dtypes == 'object':
        new_df['trip_distance(km)']= new_df['trip_distance(km)'].str.split(",").str[0]

    return new_df

def remove_null(data: pd.DataFrame):
    data = data[data['trip_distance(km)'].isnull() == False]

    null_columns = data.columns[data.isnull().any()]
    null_percentages = {}

    for column in null_columns:
        null_percentage = (data[column].isnull().sum() / len(data)) * 100
        null_percentages[column] = null_percentage

    null_df = pd.DataFrame.from_dict(null_percentages, orient='index', columns=['Null Percentage'])

    columns_to_drop = null_df[null_df['Null Percentage'] > THRESHOLD].index

    df_cleaned = data.drop(columns=columns_to_drop)

    return df_cleaned

def count_outliers(data, column):
    count=0
    q1=data[column].describe()[4]
    q3=data[column].describe()[6]
    iqr=q3-q1
    for i in data[column]:
        if (i<q1-(1.5*iqr)) or (i>q3+(1.5*iqr)):
            count+=1
    return count

def remove_outliers(data, column_list):
    for column in column_list:
        q1 = data[column].describe()[4]
        q3 = data[column].describe()[6]
        iqr = q3-q1
        for i in data[column]:
            if (i<q1-(1.5*iqr)) or (i>q3+(1.5*iqr)):
                data = data.loc[data[column] != i]
    
    return data

def fill_data(data: pd.DataFrame):
    categorical_columns = [col for col in data.columns if pd.api.types.is_categorical_dtype(data[col]) or data[col].dtype == 'object']

    for column in categorical_columns:
        mode_value = data[column].mode()[0]
        data[column].fillna(mode_value, inplace=True)

    numerical_columns = data.select_dtypes(include=['int64','float64']).columns
    imputer = SimpleImputer(strategy='mean')

    data[numerical_columns] = imputer.fit_transform(data[numerical_columns])

    outlier_columns = data.select_dtypes(exclude='object')

    data = remove_outliers(data, outlier_columns)

    ordinalEncoder = OrdinalEncoder(categories=[['Normal', 'Moderate', 'Fast']])
    data['encoded_driving_style'] = ordinalEncoder.fit_transform(data.driving_style.values.reshape(-1,1))
    data.drop("driving_style", axis=1, inplace=True)

    labelEncoder = LabelEncoder()
    data['encoded_tire_type'] = labelEncoder.fit_transform(data.tire_type)
    data.drop("tire_type", axis=1, inplace=True)

    data[['city','motor_way','country_roads','A/C']] = pd.get_dummies(data[['city','motor_way','country_roads','A/C']], drop_first=True)

    return data

def process():
    config = get_config()
    
    path = config.dataset
    df = pd.read_csv(path,encoding="ISO-8859-1")
    
    return fill_data(remove_null(remove_and_cast(df)))

def main():
    final_data = process()

    final_data.to_csv("final_data.csv",index=False)


if __name__ == '__main__':
    main()
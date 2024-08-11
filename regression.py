from config import get_config

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle

config = get_config()

FILE_PATH = config.dataset
OUT_PATH = './model_regression_v1/{name}'.format(name=config.car_name)
SEED = config.seed
NUM_FOLDS = config.num_folds
MODEL_LIST = [
    'Linear Regression',
    'Elastic Net',
    'Random Forest R',
    'AdaBoost R'
]

LN_PARAMS = {
    'penalty': ['l1', 'l2', 'elasticnet', None],
    'multi_class': ['ovr', 'multinomial'],
    'C': np.linspace(0.001, 5, 2),
    'l1_ratio': np.linspace(0, 1, 2)
}

ELASTIC_PARAMS = {
    'alpha': np.linspace(0.001, 15, 20),
    'l1_ratio': np.linspace(0.001, 0.999, 25),
    'random_state': [SEED]
}

RF_PARAMS = {
    'max_features':[2,3]
}

AB_PARAMS = {
    'learning_rate': np.linspace(0.001, 0.8, 2),
    'n_estimators': [50]
}

def process_data():
    df = pd.read_csv(FILE_PATH)
    col = ['consumption(kWh/100km)','manufacturer','model','version','fuel_date','fuel_type','power(kW)']
    for i in col:
        if i in df.columns:
            df = df.drop(i,axis=1)
    df['encoded_driving_style']  = df['encoded_driving_style'].astype('int')
    df['park_heating']  = df['park_heating'].astype('int')

    df['ecr_dev_type'] = df['ecr_deviation'].apply(lambda x: 1 if x >= 0 else 0 )
    df.drop('ecr_deviation',axis=1,inplace=True)

    return df

def split_data(df: pd.DataFrame):
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    train, test  = train_test_split(df, test_size = 0.2, random_state = SEED)

    y_train = train['trip_distance(km)']
    X_train = train.drop('trip_distance(km)', axis=1)

    y_test = test['trip_distance(km)']
    X_test = test.drop('trip_distance(km)', axis=1)

    y_train = y_train.values
    y_test = y_test.values

    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled = scaler_x.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))

    return X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, scaler_x, scaler_y

def basic_train(models, X_train, y_train, is_plot: bool):
    def plot():
        train_score = pd.DataFrame(data = training_score, columns = ['MAE'])
        train_score.index = ['LN',  'ELAS', 'RF', 'AB']
        train_score = train_score.sort_values(by = 'MAE')

        plt.figure(figsize=(15, 5))
        sns.barplot(x=train_score.index, y='MAE', data=train_score,palette="rocket")
        plt.xlabel('Models')
        plt.ylabel('MAE')
        plt.title('MAE vs Models')
    
    training_score = []
    for model in models:
        model.fit(X_train,y_train)
        y_pred_train = model.predict(X_train)
        train_mse = mean_absolute_error(y_train, y_pred_train)
        training_score.append(train_mse)
    
    if is_plot:
        plot()

    return models

def cv_train(models, X_train, y_train, is_plot: bool):
    def cross_validation(model, X_train, y_train):
        kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
        
        neg_x_val_score = cross_val_score(model, X_train, y_train, cv = kf, n_jobs = -1, scoring = 'neg_mean_absolute_error')
        x_val_score = np.round((-1*neg_x_val_score), 5)

        return x_val_score.mean()
    
    def plot():
        x_val_score = pd.DataFrame(data = cv_score, columns = ['Cross Validation Scores (MAE)'])
        x_val_score.index = ['Linear Reg',  'Elastic',  'Random Forest', 'Ada Boost']
        x_val_score = x_val_score.round(5)
        x_val_score  = x_val_score.sort_values(by = 'Cross Validation Scores (MAE)')

        plt.figure(figsize=(15, 5))
        sns.barplot(x=x_val_score.index, y='Cross Validation Scores (MAE)', data=x_val_score,palette="rocket")
        plt.xlabel('Models')
        plt.ylabel('MAE')
        plt.title('Cross Validation Scores (MAE) vs models')
        plt.show()

    cv_score = []
    for model in models:
        cv_score.append(cross_validation(model, X_train, y_train))

    if is_plot:
        plot()

    return models

def gs_train(models, X_train, y_train, is_plot: bool):
    def grid_search_cv(model, params):
        kfold = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
        global best_params, best_score

        grid_search = RandomizedSearchCV(
            estimator = model, 
            param_distributions=params,
            verbose=1,
            n_iter=5, 
            random_state=SEED,
            cv=kfold, 
            scoring='neg_mean_absolute_error',
            n_jobs = -1
        )

        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        best_score = (np.round(grid_search.best_score_, 5))
        
        return best_params, best_score

    def repr(name, value_1, value_2):
        return '{} best params: {} and best score: {:0.5f}'.format(name, value_1, value_2)

    def plot():
        optimized_scores = pd.DataFrame({'Optimized Scores':[elas_best_score, rf_best_score,ab_best_score] })
        optimized_scores.index = ['ELAS', 'Random Forest', 'Ada Boost']
        optimized_scores = optimized_scores.sort_values(by = 'Optimized Scores')
        optimized_scores
        plt.figure(figsize=(15, 5))
        sns.barplot(x=optimized_scores.index, y='Optimized Scores', data=optimized_scores,palette="rocket")
        plt.xlabel('Models')
        plt.ylabel('MAE')
        plt.title('Optimized_MAE vs models')
        plt.show()

    elas_best_params, elas_best_score = grid_search_cv(models[1], ELASTIC_PARAMS)
    rf_best_params, rf_best_score = grid_search_cv(models[2], RF_PARAMS)
    ab_best_params, ab_best_score = grid_search_cv(models[3], AB_PARAMS)

    if is_plot:
        plot()

    opt_model = [
        ElasticNet(**elas_best_params),
        RandomForestRegressor(**rf_best_params),
        AdaBoostRegressor(**ab_best_params)
    ]

    return opt_model

def test(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    
    y_pred_test = model.predict(X_test)

    return mean_absolute_error(y_pred_test, y_test)

def save_model(model_name, model):
    with open(OUT_PATH + '/{}.pkl'.format(model_name), 'wb') as f:
        pickle.dump(model, f)

def save_scaler(scaler_name, scaler):
    with open(OUT_PATH + '/{}.pkl'.format(scaler_name), 'wb') as f:
        pickle.dump(scaler, f)

def final_plot(mse_values):
    plt.figure(figsize=(10, 6))
    plt.bar(MODEL_LIST, mse_values, color='skyblue')
    plt.xlabel('Models')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.title('Mean Absolute Error Comparison of Models')
    plt.xticks(rotation=45)  
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def main():
    data = process_data()
    X_train, y_train, X_test, y_test, scaler_x, scaler_y = split_data(data)

    ln = LinearRegression()
    elas = ElasticNet(random_state = SEED)
    rf =  RandomForestRegressor(random_state = SEED)
    ab = AdaBoostRegressor(random_state = SEED)

    models = [ln, elas, rf, ab]

    init_models = basic_train(models, X_train, y_train, is_plot=False)
    ft_models = [init_models[0]] + gs_train(init_models, X_train, y_train, is_plot=False)    

    # 0 is Linear Regression
    # 1 is Elastic Net
    # 2 is Random Forest Reg
    # 3 is AdaBoost Reg
    final_result = []
    for model in ft_models:
        final_result.append(test(model, X_train, y_train, X_test, y_test))

    final_plot(final_result)

    save_model('ln', ft_models[0])
    save_model('elas', ft_models[1])
    save_model('rf', ft_models[2])
    save_model('ab', ft_models[3])
    save_scaler('scaler_x', scaler_x)
    save_scaler('scaler_y', scaler_y)


if __name__ == '__main__':
    main()
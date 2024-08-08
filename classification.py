from config import get_config

from sklearn.base import accuracy_score
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import SVC

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

config = get_config()

FILE_PATH = './dataset/{name}.csv'.format(name=config.car_name)
OUT_PATH = './model/{name}'.format(name=config.car_name)
SEED = config.seed
NUM_FOLDS = config.num_folds
MODEL_LIST = [
    'Logistic Regression',
    'SVC',
    'Random Forest C',
    'AdaBoost C'
]

LR_PARAMS = {
    'penalty': ['l1', 'l2', 'elasticnet', None],
    'multi_class': ['ovr', 'multinomial'],
    'C': np.linspace(0.001, 5, 2),
    'l1_ratio': np.linspace(0, 1, 2)
}

SVM_PARAMS = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'C': np.linspace(0.001, 1, 2),
    'gamma': np.linspace(0.001, 1, 2)
}

RF_PARAMS = {
    'max_features':[2,3],
    'min_samples_leaf':[2]
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
    def scaler_x(data):
        scaler = MinMaxScaler()
        return scaler.fit_transform(data)
    
    def scaler_y(data):
        scaler = MinMaxScaler()
        return scaler.fit_transform(data.reshape(-1, 1))
    
    train, test  = train_test_split(df, test_size = 0.2, random_state = SEED)

    y_train = train['ecr_dev_type']
    X_train = train.drop('ecr_dev_type',axis=1)

    y_test = test['ecr_dev_type']
    X_test = test.drop('ecr_dev_type',axis=1)

    y_train = y_train.values
    y_test = y_test.values

    return scaler_x(X_train), scaler_y(y_train), scaler_x(X_test), scaler_y(y_test)

def basic_train(models, X_train, y_train, is_plot: bool):
    def plot():
        train_score = pd.DataFrame(data = training_score, columns = ['Training_Accuracy'])
        train_score.index = ['LR',  'SVM',  'RF', 'AB']
        train_score = train_score.sort_values(by = 'Training_Accuracy')

        plt.figure(figsize=(15, 5))
        sns.barplot(x=train_score.index, y='Training_Accuracy', data=train_score,palette="rocket")
        plt.xlabel('Models')
        plt.ylabel('Training_Accuracy')
        plt.title('Training_Accuracy vs Models')
    
    training_score = []
    for model in models:
        model.fit(X_train,y_train)
        y_pred_train = model.predict(X_train)
        train_mse = accuracy_score(y_train, y_pred_train)
        training_score.append(train_mse)
    
    if is_plot:
        plot()

    return models

def cv_train(models, X_train, y_train, is_plot: bool):
    def cross_validation(model, X_train, y_train):
        kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
        
        neg_x_val_score = cross_val_score(model, X_train, y_train, cv = kf, n_jobs = -1, scoring = 'accuracy')
        x_val_score = np.round(neg_x_val_score, 5)

        return x_val_score.mean()
    
    def plot():
        x_val_score = pd.DataFrame(data = cv_score, columns = ['Cross Validation Scores (Accuracy)'])
        x_val_score.index = ['Logistic Reg',  'SVM',  'Random Forest', 'Ada Boost']
        x_val_score = x_val_score.round(5)
        x_val_score  = x_val_score.sort_values(by = 'Cross Validation Scores (Accuracy)')

        plt.figure(figsize=(15, 5))
        sns.barplot(x=x_val_score.index, y='Cross Validation Scores (Accuracy)', data=x_val_score,palette="rocket")
        plt.xlabel('Models')
        plt.ylabel('Accuracy')
        plt.title('Cross Validation Scores (Accuracy) vs models')
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
            scoring='accuracy',
            n_jobs = -1
        )

        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        best_score = (np.round(grid_search.best_score_, 5))
        
        return best_params, best_score

    def repr(name, value_1, value_2):
        return '{} best params: {} and best score: {:0.5f}'.format(name, value_1, value_2)

    def plot():
        optimized_scores = pd.DataFrame({'Optimized Scores':[lr_best_score, svm_best_score, rf_best_score,ab_best_score] })
        optimized_scores.index = ['Logistic Reg',  'SVM',  'Random Forest', 'Ada Boost']
        optimized_scores = optimized_scores.sort_values(by = 'Optimized Scores')
        optimized_scores
        plt.figure(figsize=(15, 5))
        sns.barplot(x=optimized_scores.index, y='Optimized Scores', data=optimized_scores,palette="rocket")
        plt.xlabel('Models')
        plt.ylabel('Accuracy')
        plt.title('Optimized_Accuracy vs models')
        plt.show()

    lr_best_params, lr_best_score = grid_search_cv(models[0], LR_PARAMS)
    svm_best_params, svm_best_score = grid_search_cv(models[1], SVM_PARAMS)
    rf_best_params, rf_best_score = grid_search_cv(models[2], RF_PARAMS)
    ab_best_params, ab_best_score = grid_search_cv(models[3], AB_PARAMS)

    if is_plot:
        plot()

    opt_model = [
        LogisticRegression(**lr_best_params),
        SVC(**svm_best_params),
        RandomForestClassifier(**rf_best_params),
        AdaBoostClassifier(**ab_best_params)
    ]

    return opt_model

def test(model, X_test, y_test):
    y_pred_test = model.predict(X_test)

    return accuracy_score(y_pred_test, y_test)

def final_plot(mse_values):
    plt.figure(figsize=(10, 6))
    plt.bar(MODEL_LIST, mse_values, color='skyblue')
    plt.xlabel('Models')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Mean Squared Error Comparison of Models')
    plt.xticks(rotation=45)  
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def main():
    data = process_data()
    X_train, y_train, X_test, y_test = split_data(data)

    lr = LogisticRegression(n_jobs = -1)
    svm = SVC(random_state = SEED)
    rf =  RandomForestClassifier(n_jobs = -1, random_state = SEED)
    ab = AdaBoostClassifier(random_state = SEED)

    models = [lr, svm, rf, ab]

    init_models = basic_train(models, X_train, y_train, is_plot=False)
    ft_models = gs_train(init_models, X_train, y_train, is_plot=False)      

    # 0 is Linear Regression
    # 1 is Support Vector Classifier
    # 2 is Random Forest Classifier
    # 3 is AdaBoost Classifier
    final_result = []
    for model in ft_models:
        final_result.append(test(model, X_test, y_test))


if __name__ == '__main__':
    main()
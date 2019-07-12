from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor


def tune_parameters(model_name, X_train, y_train, X_test, y_test):
    '''
    The function do a Grid Search to find the best parameters to fit the data.

    Parameters:
        model_name: The name of a regression model.
        X_train: Features training set.
        y_train: Target training set.
        X_test: Features testing set.
        y_test: Target testing set.

    Returns:
        grid_search: The fitted model with the best parameters
    '''

    if (model_name == 'LinearRegression'):
        model = LinearRegression()
        parameters = {'fit_intercept': [True,False],
                'normalize': [True,False],
             }

    elif (model_name == 'DecisionTreeRegressor'):
        model = DecisionTreeRegressor(random_state = 50)
        parameters = { 'max_depth':[None, 2, 5, 10, 20, 50, 100],
                        'min_samples_split' :[None, 2, 5, 10, 20, 50, 100],
                        'min_samples_leaf' :[None, 2, 5, 10, 20, 50, 100],
                        'max_leaf_nodes' :[None, 2, 5, 10, 20, 50, 100],
         }

    elif (model_name == 'RandomForestRegressor'):
        model = RandomForestRegressor(random_state = 50)
        parameters = { 'n_estimators':[None, 5, 10, 50, 100,150,200],
                        'max_depth':[None, 2, 5, 10, 20, 50, 100],
                        'min_samples_split' :[None, 2, 5, 10, 20, 50, 100],
                        'min_samples_leaf' :[None, 2, 5, 10, 20, 50, 100],
                        'max_leaf_nodes' :[None, 2, 5, 10, 20, 50, 100],
         }

    elif (model_name == 'SVR'):
        model = SVR(random_state = 50)
        parameters = { 'kernel':['linear', 'poly', 'rbf', 'sigmoid', 'precomputed' ],
                        'degree':[1, 2, 3, 4, 5],
                        'C' :[1.0, 1.2, 2.0, 2.5, 4.0],
                        'shrinking' :[True,False]
         }

    elif (model_name == 'Lasso'):
        model = Lasso(random_state = 50)
        parameters = { 'alpha':[0.0001, 0.001, 0.02, 0.025, 0.03, 0.1, 0.7],
                'fit_intercept': [True,False],
                'normalize':[True,False],
                'max_iter' : [1,20,100,200],
                'tol': [1,20,100,200],
                'warm_start': [True,False],
                'positive' :[True,False]
        }

    elif (model_name == 'ElasticNet'):
        model = ElasticNet(random_state = 50)
        parameters = { 'alpha':[0.0001, 0.001, 0.02, 0.03, 0.1, 0.9],
                'l1_ratio' : [0.0001, 0.001, 0.02, 0.03, 0.1,0.9],
                'fit_intercept': [True,False],
                'normalize':[True,False],
                'max_iter' : [1,20,100,200],
                'tol': [1,20,100,200],
                'warm_start': [True,False],
                'positive' :[True,False]
        }

    elif (model_name == 'GradientBoostingRegressor'):
        model = GradientBoostingRegressor(random_state = 50)
        parameters = { 'n_estimators':[None, 5, 10, 50, 100,150,200],
                'learning_rate' : [0.0001, 0.001, 0.02, 0.025, 0.03, 0.1,0.9],
                'max_depth':[None, 2, 5, 10, 20, 50, 100],
                'min_samples_split' :[None, 2, 5, 10, 20, 50, 100],
                'min_samples_leaf' :[None, 2, 5, 10, 20, 50, 100],
                'max_leaf_nodes' :[None, 2, 5, 10, 20, 50, 100],
        }

    elif (model_name == 'AdaBoostRegressor'):
        model = AdaBoostRegressor(random_state = 50)
        parameters = { 'n_estimators':[None, 5, 10, 50, 100,150,200],
                'learning_rate' : [0.0001, 0.001, 0.02, 0.025, 0.03, 0.1,0.9],
        }

    elif (model_name == 'KNeighborsRegressor'):
        model = KNeighborsRegressor(random_state = 50)
        parameters = { 'n_neighbors':[2, 5, 10, 50, 100,150,200],
                'weights' : ['uniform' , 'distance'],
                'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'p': [1, 2, 3, 4, 5],
        }

    grid_search = GridSearchCV(model, parameters, scoring = 'r2', return_train_score=True, verbose=2)
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)

    train_predict = grid_search.predict(X_train)
    test_predict = grid_search.predict(X_test)

    print("r2 score for train: ",r2_score(y_train,train_predict))
    print("r2 score for test: ",r2_score(y_test,test_predict))

    return grid_search

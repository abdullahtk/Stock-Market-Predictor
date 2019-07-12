
from source import ModelsParametersTunning as mpt
import sys
import numpy as np
import pandas as pd
from yahoofinancials import YahooFinancials
from ta import *
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from pandas.tseries.offsets import BDay



def install_and_import(package):
    '''
    The function to download, install and import python packages.

    Parameters:
        package: Name of a package.

    Returns:
        None
    '''
    import importlib
    try:
        importlib.import_module(package)
    except ImportError:
        import pip
        pip.main(['install', package])
    finally:
        globals()[package] = importlib.import_module(package)

install_and_import('yahoofinancials')
install_and_import('ta')
install_and_import('sklearn')


def getData(tickers,start_date,end_date):
    '''
    The function to download data from Yahoo Finance API. It takes a list of tickers, start date
    and end date, which is the last trading date. The return is a dataframe containing the
    tickers' adjusted close price and volume.

    Parameters:
        tickers: A list of tickers.
        start_date: The start date for fetching the data.
        end_date: The last date to fetch the data.
    Returns:
        df: Dataframe that contains the tickers' adjusted close price and volume.
    '''

    df = pd.DataFrame()
    print(tickers)
    if(len(tickers) > 0):
        for ticker in tickers:
            yahoo_financials = YahooFinancials(str(ticker))
            daily = yahoo_financials.get_historical_price_data(start_date, end_date, 'daily')

            daily = daily.get(ticker).get('prices')
            df_temp = pd.DataFrame.from_records(daily)
            df_temp = df_temp.drop(['close','date','high','low','open'],axis=1)
            df_temp = df_temp.rename(columns={'formatted_date':'date','adjclose':ticker, 'volume':ticker +'_volume' })
            df_temp['date'] = pd.to_datetime(df_temp['date'])
            df_temp.set_index('date', inplace=True)
            df = pd.concat([df,df_temp],axis=1)

    return df

def introduce_features(df, ticker_of_interest, tickers, number_of_days_to_predict):
    '''
    The function to calculate different indecators and add it to the original
    dataframe. The indicatores are:
        *

    Parameters:
        df: Dataframe the contains the daily close price.
        ticker_of_interest: The ticker that will be predicted.
        tickers: A list of tickers.
        number_of_days_to_predict: number of days that wanted to be predicted
    Returns:
        df: Dataframe that contains the tickers' adjusted close price along
        with the calculated indecators and temporary future prices
    '''

    for ticker in tickers:
        print(ticker)
        close_price = df[ticker]
        df[ticker] = close_price
        daily_pct = close_price.pct_change(1)
        df[ticker + '_pct'] = daily_pct

        # Calculate the 5 days moving averages of the closing prices
        short_rolling = close_price.rolling(window=5).mean()
        df[ticker + '_short_rolling'] = short_rolling
        short_rolling_std = short_rolling.std()
        df[ticker + '_short_upper_band'] = df[ticker + '_short_rolling'] + (2 * short_rolling_std)
        df[ticker + '_short_lower_band'] = df[ticker + '_short_rolling'] - (2 * short_rolling_std)

        # Calculate the 10 and 100 days moving averages of the closing prices
        short_rolling = close_price.rolling(window=10).mean()
        df[ticker + '_short_rolling'] = short_rolling
        short_rolling_std = short_rolling.std()
        df[ticker + '_short_upper_band'] = df[ticker + '_short_rolling'] + (2 * short_rolling_std)
        df[ticker + '_short_lower_band'] = df[ticker + '_short_rolling'] - (2 * short_rolling_std)

        # 64 for three months working days. Or 100 as used by others
        long_rolling = close_price.rolling(window=100).mean()
        df[ticker + '_long_rolling'] = long_rolling
        long_rolling_std = long_rolling.std()
        df[ticker + '_long_upper_band'] = df[ticker + '_long_rolling'] + (2 * long_rolling_std)
        df[ticker + '_long_lower_band'] = df[ticker + '_long_rolling'] - (2 * long_rolling_std)

        df[ticker + '_rsi_5'] = momentum.rsi(df[ticker], n=5, fillna=False)
        df[ticker + '_rsi_10'] = momentum.rsi(df[ticker], n=10, fillna=False)
        df[ticker + '_rsi_64'] = momentum.rsi(df[ticker], n=100, fillna=False)

        df[ticker + '_force_index_5'] = volume.force_index(df[ticker], df[ticker + '_volume'], n=5, fillna=False)
        df[ticker + '_force_index_10'] = volume.force_index(df[ticker], df[ticker + '_volume'], n=10, fillna=False)
        df[ticker + '_force_index_64'] = volume.force_index(df[ticker], df[ticker + '_volume'], n=64, fillna=False)

        df[ticker + '_obv'] = volume.on_balance_volume(df[ticker], df[ticker + '_volume'], fillna=False)

        df[ticker + '_ema_5'] = trend.ema_indicator(df[ticker], n=5, fillna=False)
        df[ticker + '_ema_10'] = trend.ema_indicator(df[ticker], n=10, fillna=False)
        df[ticker + '_ema_64'] = trend.ema_indicator(df[ticker], n=100, fillna=False)

        if(ticker != ticker_of_interest):
            pass

        else:
            # add the price and the future price of the symbol to the output data frame ans_df
            df[ticker + '_future_price'] = np.concatenate((close_price[int(number_of_days_to_predict):].values, [np.nan]*int(number_of_days_to_predict)))

    df = df.fillna(method='backfill')
    df = df.fillna(method='pad')

    return df

def split_data(more_features, ticker_of_interest, end_date):
    '''
    The function normalizes and splits the data into different training,testing and validation
    sets.

    Parameters:
        more_features: Dataframe that contains all the data with the intorduced
        featuers.
        ticker_of_interest: The ticker that will be predicted.
        end_date: The last date to fetch the data.

    Returns:
        data_dict: dictioanry that contains all the training,testing and validation
        sets.
    '''
    all_features = more_features.drop([(ticker_of_interest + '_future_price')], axis=1)
    all_mean = all_features.mean(axis = 0)
    all_std = all_features.std(axis = 0)
    all_features_normed = (all_features - all_mean) / all_std
    all_target = more_features[ticker_of_interest + '_future_price']

    training_features = all_features[:end_date]
    training_mean = training_features.mean(axis = 0)
    training_std = training_features.std(axis = 0)
    training_features_normed = (training_features - training_mean) / training_std
    training_target = all_target[:end_date]

    n = all_features.shape[0]
    small_features_size = int(n * 0.75)

    small_features = all_features[:small_features_size]
    small_mean = small_features.mean(axis = 0)
    small_std = small_features.std(axis = 0)
    small_features_normed = (small_features - small_mean) / small_std
    #self.small_trainX is a ndarray

    small_target = all_target[:small_features_size]
    #self.small_trainY is a Series

    features_validation = all_features[small_features_size:]
    features_validation_normed = (features_validation - small_mean) / small_std
    #self.validationX is a ndarray

    price_validation = more_features[ticker_of_interest][small_features_size:]
    #self.validationPrices is a Series

    future_price_validation = more_features[ticker_of_interest + '_future_price'][small_features_size:]
    #self.validationFuturePrices is a Series

    data_dict = {
        'all_features_normed' : all_features_normed,
        'all_target' : all_target,
        'training_features_normed' : training_features_normed,
        'training_target' : training_target,
        'small_features_normed' : small_features_normed,
        'small_target' : small_target,
        'features_validation_normed' : features_validation_normed,
        'price_validation' : price_validation,
        'future_price_validation' : future_price_validation
    }

    return data_dict

def train_predict(models, X_train, y_train, X_test, y_test):
    """
    The function loop through list of models and fit the training set
    then it prints the r2 score.

    Parameters:
        models (list): The list of models.
        X_train: Features training set.
        y_train: Target training set.
        X_test: Features testing set.
        y_test: Target testing set.

    Returns:
        models_dict: dictioanry contains each model with the r2 score
    """
    models_dict = {}
    for model in models:
        print("Train using: " , model.__class__.__name__)
        model.fit(X_train,y_train)
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)
        print("r2 score for train: ",r2_score(y_train,train_predict))

        print("r2 score for test: ",r2_score(y_test,test_predict))
        print(model)
        print()
        model_dict = {
            'model' : model,
            'test_score' : r2_score(y_test,test_predict)
        }

        models_dict[model.__class__.__name__] = model_dict

    return models_dict

def pick_best_regressor(training_features_normed, training_target, features_validation_normed, future_price_validation):
    """
    The function initiates a set of regression models and then call
    train_predict to fit and print the r2 score. Then, it returns the model with
    the highest r2 score.

    Parameters:
        training_features_normed: Normalized features training set.
        training_target: Target training set.
        features_validation_normed: Normalized features testing set.
        future_price_validation: Target testing set.

    Returns:
        highest_model: Model with the highest r2 score
        highest_score: r2 score of the highest model
    """

    models = []
    models.append(LinearRegression())
    models.append(DecisionTreeRegressor())
    models.append(RandomForestRegressor())
    models.append(SVR())
    models.append(Lasso())
    models.append(ElasticNet())
    models.append(GradientBoostingRegressor())
    models.append(AdaBoostRegressor())
    models.append(KNeighborsRegressor())

    models_dict = train_predict(models, training_features_normed, training_target, features_validation_normed, future_price_validation)

    is_init_high_score_set = False
    highest_score = None
    highest_model = None
    for model in models_dict:
        if (is_init_high_score_set == False):
            highest_score = models_dict[model]['test_score']
            highest_model = models_dict[model]['model']
            is_init_high_score_set = True

        for items in models_dict[model]:
            test_score = models_dict[model]['test_score']
            if(float(test_score) > float(highest_score)):
                print('Change highest_score')
                highest_score = test_score
                highest_model = models_dict[model]['model']

    print(highest_model, highest_score)
    return highest_model, highest_score

def check_trading_day(df_indices, date):
    """
    The function return the next trading day.

    Parameters:
        df_indices: List of trading days in the whole dataset
        date: The prediction day

    Returns:
        date: The next trading day

    """
    while (True):
        if (date in df_indices):
            break
        else:
            date = date+BDay()
            date = date.date()

    return date

def predict_n_days(model, all_features_normed, prediction_day, number_of_days_to_predict):
    """
    The function return the predictions for n days.

    Parameters:
        model: The regression model
        all_features_normed: The normalized main dataset
        prediction_day: The prediction day
        number_of_days_to_predict: The number of days to pridect after the
        prediction day


    Returns:
        predictions: List of predictions with the date

    """
    prediction_day = check_trading_day(all_features_normed.index, prediction_day)
    predictions = pd.DataFrame(columns=['Date', 'Predicted Price'])
    next_day = prediction_day
    for i in range(int(number_of_days_to_predict)):
        try:
            value_to_predict = all_features_normed.loc[next_day].values
            value_to_predict = value_to_predict.reshape(1, -1)
            predict_current = model.predict(value_to_predict)
            future_date = next_day + BDay(int(number_of_days_to_predict))
            predictions.loc[i] = [future_date] + [predict_current[0]]
            next_day = next_day + BDay()
        except:
            print (sys.exc_info()[0])

    return predictions

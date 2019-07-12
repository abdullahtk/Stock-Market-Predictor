from flask import Flask
from flask import render_template, request, jsonify
from source import StockPredictor as sp
from source import ModelsParametersTunning as mpt
from datetime import datetime
import json
from plotly.graph_objs import Scatter
from pandas.tseries.offsets import BDay


app = Flask(__name__)

def install_and_import(package):
    import importlib
    try:
        importlib.import_module(package)
    except ImportError:
        import pip
        pip.main(['install', package])
    finally:
        globals()[package] = importlib.import_module(package)

install_and_import('plotly')

@app.route('/')
def index():
    print('index')
    return render_template('master.html')

@app.route('/go', methods=['GET', 'POST'])
def go():
    # save user input in query
    query = request.values
    print('go')

    tickers = []
    ticker_of_interest = request.values.get('ticker')
    tickers.append(ticker_of_interest)
    tickers.append('SPY')
    start_date_str = request.values.get('start_date')
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
    end_date_str = request.values.get('end_date')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
    prediction_date_str = request.values.get('prediction_date')
    prediction_date = datetime.strptime(prediction_date_str, '%Y-%m-%d').date()
    number_of_days = request.values.get('number_of_days')

    if (number_of_days == ""):
        number_of_days = 5

    df = sp.getData(tickers , start_date.strftime("%Y-%m-%d"), '2019-07-08')

    more_features = sp.introduce_features(df, ticker_of_interest,tickers,number_of_days)
    data_dict = sp.split_data(more_features, ticker_of_interest, end_date)
    all_features_normed = data_dict['all_features_normed']
    all_target = data_dict['all_target']
    training_features_normed = data_dict["training_features_normed"]
    training_target = data_dict["training_target"]
    small_features_normed = data_dict["small_features_normed"]
    small_target = data_dict["small_target"]
    features_validation_normed = data_dict["features_validation_normed"]
    future_price_validation = data_dict["future_price_validation"]
    price_validation = data_dict["price_validation"]
    highest_model, highest_score = sp.pick_best_regressor(small_features_normed, small_target, features_validation_normed, future_price_validation)
    tunned_model = mpt.tune_parameters(highest_model.__class__.__name__, small_features_normed, small_target, features_validation_normed, future_price_validation)

    model = tunned_model.fit(all_features_normed,all_target)
    predictions = sp.predict_n_days(model, all_features_normed, prediction_date, number_of_days)
    real_data = df[predictions['Date'][0]:predictions['Date'][0]+BDay(int(number_of_days)-1)][ticker_of_interest]

    pct = [abs(float(r)-float(p))/float(r)*100 for r,p in zip(real_data,predictions['Predicted Price'])]


    # Plot closing prices
    graphs = [
        {
            'data': [
                Scatter(
                    x=df[ticker_of_interest].index,
                    y=df[ticker_of_interest],
                )
            ],

            'layout': {
                'title': 'Adjusted Close Price' ,
                'yaxis': {
                    'title': "Price"
                },
                'xaxis': {
                    'title': "Date"
                }
            }
    },
    {
        'data': [
            Scatter(
                x=predictions['Date'],
                y=predictions['Predicted Price'],
                name= 'Predicted Price',
            ),
            Scatter(
                x=predictions['Date'],
                y=real_data,
                name= 'Actual Price',
            ),
            Scatter(
                x=predictions['Date'],
                y=pct,
                name= 'PCT',
                yaxis= 'y2',
                line = dict(
                    width = 1,
                    dash = 'dash')
            )
        ],

        'layout': {
            'title': 'Predicted Adjusted Close Price' ,
            'xaxis': {
                'title': "Date"
            },
            'yaxis': {
                'title': "Price"
            },
            'yaxis2': {
                'title': 'Actual vs. Predicted',
                'overlaying': 'y',
                'side': 'right'
            }
        }
    }]

    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('go.html', query=query , df=data_dict, ids=ids, graphJSON=graphJSON)

if __name__ == '__main__':
    app.run(debug=True)

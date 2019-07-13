# Stock Price Predictor (Udacity Capestone Project)

### Project Description

The purpose of the project is to collect historical data of a stock and then analyze and predict the close price for a range of days.


### Analysis


### Conclusion


## Libraries
The following Libraries were used in this project:
* numpy
* pandas
* matplotlib
* sklearn
* flask
* plotly
* yahoofinancials
* ta

## Folders and Files
- Investment and Trading.ipynb: Jupyter notebook contains all the code from collecting the data until visualizing the predicted close price.
- Investment and Trading.html: Similar to the Jupyter notebook but in HTML format.
- requirements.txt: Contains all the libraries used to run the project.
- stock_price_predictor (folder): contains one file and two folders:
    * app.py: Flask web application
    * templates (folder): contains two HTML files for the web application
    * source (folder): contains two python files
		- StockPredictor.py: Contains the code to get the data from Yahoo Financial, clean the data, features engineering, model generation and testing, and prediction.
		- ModelsParametersTunning.py: Contains the code to tune the regression models parameters.


## Instructions:

1. Run the following command in the app's directory to run your web app.
    `python app.py`

3. Go to http://0.0.0.0:5000/


## License
The datasets used in this analysis were acquired from Yahoo Financial. The template code were given by Udacity.

# Stock Price Predictor (Udacity Capestone Project)

### Project Description

The purpose of the project is to predict the adjusted close price for a range of days for a particular stock. To do so, first, historical daily data is collected from [Yahoo Finance](https://finance.yahoo.com/ "Yahoo Finance"). Next, several technical indicators will be calculated and added to the data set as featuers. After that, the data will be fit on different regression models to find the best one that score the highest R2 score. That model will then go under Grid search to find the best parameters to fit the data. Finally, the optimized model will be used to predict the adjusted close price for future dates.


### Analysis
The first thing to do is to fetch historical daily stock prices from of interest from  [Yahoo Finance](https://finance.yahoo.com/ "Yahoo Finance"). The returned data contains (close, date, high, low, open, adjusted close, volume). As I'm only interested in adjusted close price, all the columns dropped except for the date, adjusted close, and volume. Next, the historical daily prices for the S&P index (SPY) were fetched too and added to the dataset.

One benifite from using  [Yahoo Finance](https://finance.yahoo.com/ "Yahoo Finance") is that the data is clean from missing values so no need to treat missing values.

The daily close price for the ticker of interest, which is Google in my case, and the S&P are shown below.

# To Do (Add image)

Next, using the adjusted close price and volume for each ticker, several technical indicators were calculated and added to the dataset. The technical indecatores are as follows:
- Daily percentage of change
- Moving averages of different window sizes (5,10,100)
- Bollinger Bands for each moving average
- RSI
- force_index
- obv
- ema
# To Do (Describe each indicator)

A new column for the target is added which constructed by shifting the daily close price of the ticker of interest by the number of days to predict and then add null values to the end of the column to match the shape of the dataset.

Finally, the missing values intorduced by the moving averages were filled with using backfill. The other missing values filled with padding.

Now the data is ready for training and testing splitting.
# To Do (Cont.)


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

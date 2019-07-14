# Stock Price Predictor (Udacity Capestone Project)

### Project Description

The purpose of the project is to predict the adjusted close price for a range of days for a particular stock. To do so, first, historical daily data is collected from [Yahoo Finance](https://finance.yahoo.com/ "Yahoo Finance"). Next, several technical indicators will be calculated and added to the data set as featuers. After that, the data will be fit on different regression models to find the best one that score the highest R2 score. That model will then go under Grid search to find the best parameters to fit the data. Finally, the optimized model will be used to predict the adjusted close price for future dates. I built a web application where the user can select a ticker, prediction date and number of days to predict.


### Analysis
The first thing to do is to fetch historical daily stock prices from of interest from  [Yahoo Finance](https://finance.yahoo.com/ "Yahoo Finance"). The returned data contains (close, date, high, low, open, adjusted close, volume). As I'm only interested in adjusted close price, all the columns dropped except for the date, adjusted close, and volume. Next, the historical daily prices for the S&P index (SPY) were fetched too and added to the dataset.

One benifite from using  [Yahoo Finance](https://finance.yahoo.com/ "Yahoo Finance") is that the data is clean from missing values so no need to treat missing values.

The daily close price for the ticker of interest, which is Google in my case, and the S&P are shown below.
![](https://i.ibb.co/R7mmym6/GOOG-and-SPY.png)

Next, using the adjusted close price and volume for each ticker, several technical indicators were calculated and added to the dataset. The technical indecatores are as follows:
- Daily percentage of change
	Percentage change between the current and a prior element. This indecate if the stock is increasing or decreasing. The formula to calculate it is:
(new value - old value)/ old value * 100

- Moving averages of different window sizes (5,10,100)
It is the average calculation for a certin number of values. It provides smoother line than the daily values.  The formula to calculate it is:
(P1+P2+...Pn)/n
where:
 - Pn is the price at the nth day
 - n is the window size or the number of days to calculate the average

- EMA
It is the average calculation for a certin number of values. places a greater weight and significance on the most recent data points. The following is the formula to calculate EMA
![](https://hedgetrade.com/wp-content/uploads/2019/04/ema-formula-1024x261.png)

- Bollinger Bands for each moving average
They are an upper and lower bands along with the daily movements of the stock's price to help on deciding when to buy or sell the stock. It can be calculated using the following formula:
upper band = moving_avg + (2 * moving_avg_std)
lower band = moving_avg - (2 * moving_avg_std)
where:
 - moving_avg is the moving average calculated earlier
 - moving_avg_std is the standered deviation for the moving average

- RSI
The relative strength index (RSI) is a momentum indicator that measures the magnitude of recent price changes to evaluate overbought or oversold conditions in the price of a stock. The formula to calculate it is:
![](http://hsp2gp1.etnet.com.hk/images/eng/chart_RSI_formula.gif)

- Force Index
The Force Index is an indicator that uses price and volume to assess the power behind a move or identify possible turning points. It can be calculated using the following formula:
[Close (current period) – Close (prior period)] x Volume

- On Balance Volume
It combines price and volume in an attempt to determine whether price movements are strong or are weak. The formula to calculate the OBV is:

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/fbbbd91a366a4a7177715be92442a468cadb8b78)


A new column for the target is added which constructed by shifting the daily close price of the ticker of interest by the number of days to predict and then add null values to the end of the column to match the shape of the dataset.

Finally, the missing values intorduced by the moving averages were filled with using backfill. The other missing values filled with padding.

Now the data is ready for training and testing splitting. First, the data has been splitted into featuers and target sets. The featuers was normalized before splitting ti further into training and testing sets. The training set consist of 75% of the data.

Next, I trained different regression model to check which one perfromes better. The models are LinearRegression, DecisionTreeRegressor, RandomForestRegressor, SVR, Lasso, ElasticNet, GradientBoostingRegressor, AdaBoostRegressor, KNeighborsRegressor. The models were evaluated using R2 score, where higher values are better than lower ones. The best model is then used in Grid search to find the best paramaters to fit the training data. From observation, LinearRegression, Lasso, and ElasticNet are the best three models.

After tuning the model, it is then used to predict the futuer close prices for N days specified by the user. The predicted prices are graphed with the real close prices along with the percentage change between the real and predicted price.

### Result
I run the model on three stocks, GOOG, AAPL, and MSFT. The following are the results from the web application for each stock.
##### GOOG
![](https://i.ibb.co/BcbFqNN/GOOG-pred.png)

The prediction date is set to 02/01/2019 and the number of days to predict is set to 5 days. The best performing model is Lasso with R2 score of 0.835 after tuning the parameters. The maximum diference between the real price and predicted price is 4.9663% while the lowest is 0.0017%

##### AAPL
![](https://i.ibb.co/B6fQGzN/Screen-Shot-2019-07-14-at-11-34-30-AM.png)

The prediction date is set to 04/02/2019 and the number of days to predict is set to 14 days. The best performing model is Lasso with R2 score of 0.727 after tuning the parameters. The maximum diference between the real price and predicted price is 2.369% while the lowest is 0.339%

##### MSFT
![](https://i.ibb.co/FhVGWg8/Screen-Shot-2019-07-14-at-11-41-08-AM.png)

The prediction date is set to 18/03/2019 and the number of days to predict is set to 5 days. The best performing model is LinearRegression with R2 score of 0.9544 after tuning the parameters. The maximum diference between the real price and predicted price is 1.935% while the lowest is 0.181%


### Conclusion
In this project, historical data for stock prices are collected from [Yahoo Finance](https://finance.yahoo.com/ "Yahoo Finance") to predict future close price. To do so, adjusted close price and volume are used to get new features, technical indecators, to help in the prediction. Several regrisson models were evaluated on R2 score and the one with the highest score picked for parameters tuning using Grid search. The final tunned model is then used for prediction. After conducting three case studies, the model were able to predict future prices with accuracy between ~5% and ~0.001%

The prediction accurecy can be further improved by incorporating more technical indecators that are really relevent to the prediction process. Additionally, sentiment analysis of corporates news could be added to further support the price movement and help in making the prediction more accurate.

Several obsticales were faced during the development of this project. First, technical analysis is out of my expertise and I spent some time just to understand the different technical indecators and how can they assisst in predicting the price movement. Second, deciding which indecators are relevent to the prediction was a time consuming task, but it helped alot in making the prediction more accurate. Lastly, picking the right regression algorithm was hard as most of the algorithms that I studied wasn't applied for time-series predictions.



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

# Machine Learning Engineer Nanodegree
## Capstone Proposal
Sandeep Paulraj 
December 25th, 2017

## Proposal

Proposal for stock price estimator.
This proposal also has an associated ipython notebook stock_price_estimator_proposal.ipynb that has the initial data exploration and setup.


### Domain Background

Fundamentally, owning stocks is a way to grow wealth. Usually investors buy stocks after some due dilligence and research. If the stock price goes up, the investor has the option of selling the stocks and making a profit. At the same time, several companies shell out dividends to shareholders if they own stock. This too is a way to accumulate wealth. However, traditional buy and hold techniques are no longer in vogue these days. Retail investors find it very tough to enter a particular stock since stock prices do vary and large institutional investors hold sway. High Frequency trading results in significant stock price variation within a trading day. Also fundamental research though still very important is kind of taking a back seat to algorithm based trading.

Machine learning technques are being used to make investment decisions. Machine learning techniques have carved out a niche for themselves in various domains and especially thsoe domains where there is a plethora of data. Stock Prices are a very appealing domain since they provide several stocks and large amounts of data. Over the last few years we have also started hearing more about several hedge funds and investment banks beginning to use these techniques. I intend to investigate some of these techniques to predict stock prices.

To be precise, i would like to be able to predict Broadcom(AVGO) stock price. This company has been in the nwes lately for its takeover attempt of Qualcomm. Broadcom is also a very large Apple Supplier. Also owning Broadcom stock myself, i want to gauge if i can come up with a model to better gauge future price movement based on available data.

### Problem Statement

Stock Prices fluctuate from day to day and to be precise, fluctuate by the second. Using publicly available data stock price data, i will attempt to predict the adjusted close price of the stock for the next seven trading days. If we are able to gauge the closing stock price, we might be able to make smart trading decisions based on the predictions. By giving a start date and a finite set of following trading days, it should be able to predict the following 7 days adjusted closing stock price.

### Datasets and Inputs

It is possible to obtain stock price financial data from variosu sources. It is also possible to use python api and the yahoo finance library to obtain this data. For some reason, i am having trouble installing the yahoo finance library in python 3.6. So i have decided, to obtain the csv data from the yahoo finance website and read in the dataframe pandas. For the initial exploratory analysis, i read in the data and realize that the dates increase, i.e in the csv file and data frame  February 1 will come before February 2. The first thing that needs to be done is to reverse this order.

Next, i remove the firdt 7 rows. The reasn i do this is as follows. We have to predict the ensuing 7 trading days stock price. Hence the first 7 rows which happen to be the last 7 trading daya will definitely have atleast 1 piece of data that we will not be aware of. To be precise let us take an example. As we are setting up and enhancing data frame, yesterday's data cane be appended with today's closing stock price; however we don't know the next 6 trading days stock price. This is something which will be good to predict.

After saving of the first seven rows, i append the remaining data frame with seven columns that have the seven following trading days adjusted closing stock price. These seven columns will become the "prediction" columns. I also have another column  where i store the difference between the highest and lowest daily stock price. 

It is important to use standard pandas routines to set up the dataframe. This essentially will result in a more elegant and cleaner final solution. Please take a look at the accompanying notebook to look at all the exploratory analysis


### Solution Statement

We are dealing with time series data. Also we fundamentally have a regression problem. This is not a classification problem. We have to predict an actual adjusted stock price; not whether the stock goes up or down. We have to predict seven outputs instead of one that i have been accustommed to do. So let us take an example. Say we need to predict 7 trading days closing stock price. We will have variosu inpust that are available to use such as trading volume, opening price, high price, low price. With this we can use regression techniques to predict the following seven days closing stock price. Now, we have already setup our data to know the following 7 days closing stock price. Thus we will have both actual closing stock price and predicted stock price based on our model. With this we can gauge how well our model is behaving. It is intended that the model will predict stock price withing a +- 5% range. Though this initial analysis tried to predict 7 trading days worth of stock price, it is possible to predict more stock prices. For the sake of this project, i will be predicting the next 7 days stock price only.


### Benchmark Model

As mentioned to some extent above, we will have actual adjusted clsoing stock prices. It is these same stock prices that we will be predicting. So we have both actual and predicted prices.This is essentially our benchmark. We can use this information to gauge how good our model is. In a way, we can consider actual adjusted closing stock prices as our benchmark.


### Evaluation Metrics

I will be leveraging sklearn in this project. From sklearn metrics we will have access to an array of metrics from our model.
The main evaluation metric i will be using is the root mean squared error.
The root mean squared is simple to calculate and can be calulated as show below. Based on previous projects and experience, i don't think a metric exists to calculate this for us. This has to be derived and is simple to derive as can be seen below.

```sh
from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(y_actual, y_predicted))
```

Now the difference between actual and prediction prices can be positive or negative. Hence it is important to take the square of the difference. We then have to take the mean of these squared values and finally take the square root.

### Project Design




-----------

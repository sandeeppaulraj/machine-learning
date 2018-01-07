# Machine Learning Engineer Nanodegree
## Stock Price Predictor
Sandeep Paulraj  
January 7th, 2018

## I. Definition
_(approx. 1-2 pages)_

### Project Overview

Investing in stocks has been a way to grow wealth for a very long time. This has been done for well over a century and as more and more countries/societies are developing, each country has its own stocks markets where securities are listed and investors can invest there money. The fundamental reason why people invest in stocks is to participate in the company's growth. If the company grows and makes money, the stock price will rise. The investor might then get regular dividends and can also sell his/her shares and make a profit if the selling price is greater than the investment. Afcourse, it is also possible to lose money.

We live in an interconencted world with an over abundance of data and we can see a large numbers of fields and industries where machine learning is being used to make informed decisions. The stock market is also one such field/entity where there is a plethora of data and this data can be used to make buy/sell decisions. Machine learning algorithms are already being used by investment banks and hedge funds and these are getting more and more complex.

In this project i will leverage publicly available historical stock price data from yahoo finance in my models.

Some papers which discuss machine learning techniques are sited below.

[Predicting Long term price Movement](https://arxiv.org/ftp/arxiv/papers/1603/1603.00751.pdf)

[Machine Learning for predicting Bond Prices](https://arxiv.org/pdf/1705.01142.pdf)


### Problem Statement

In this project, i will attempt to predict the next trading day's adjusted closing stock price for Broadcom. Broadcom is semiconductor company that is a major Apple supplier and has been very aggresively buying other companies to grow its portfolio. It is in the process of orchestarting a takeover of Qualcomm as well. I will attempt to use Broadcom stock data from yahoo finance in my model.
I will also attempt to gauge if Apple stock has an effect on the Broadcom stock price. Also, i will try to guage if the VanEck Vectors Semiconductor ETF has an effect on Broadcom.

Initially while going through the capstone review process, i was attempting to predict the following seven trading day's Adjusted Closing stock price. However, review comments suggested that i should only try to predict the following trading day's Adjusted closing stock price. On thinking about this a little more, we have to be cognizant of the fact that there are several geopolitical uncertainties that exist in the market. Let us take an example: Let us say the US amrkets end a day strong; Japan and Asia start of strong but towards the end of the Asian trading day, there is some bad news on economic indictors or say there was a data breach. Asia may end up week and this will in all likelihood have an impact on European markets which then digest this information and actually have a bad day. Europe has a bad trading daya and markets end down. This sequence of events, results in the US markets starting of week. Thus, it is difficult to make informed judgments of stock prices seven trading days ahead. To think of it, if we are on a Friday evening trying to gauge the following seven trading days' closing sdjusted stock price; that is trying to go as far ahead as the next to next Tuesday, we have two weekends in between where a lot of events might happen. Based on this and my first capstone review, it was decided to predict only the next trading day's adjsuetd closing stock price.

Why do i want to use Apple and SMH?

- I decided to use Apple Data since Broadcom is a major Apple supplier
- I decided to use SMH since it is a semiconductor ETF and broadcom is a semiconductor company.

I will use various regression techniques in my various models and settle for the model that performs the best. Now is a good time to note that we are dealing with Time Series of data and will need special consideration. We cannot shuffle the data in any of our analysis. This actually is a good deviation from what i ahve learned in the Nanodegree since we didn't exactly learn how to deal with Time Series data. Different types of regressors will be required. In the end, it is anticipated that i will end up learning new models and techniques.

The strategy I will employ will be to setup a dataframe of Broadom stock price. Initially atleast, i will use all the data provided as my features. My prediction data will actually be the next trading days' adjusted clsoing stock price. This will need some amount of data manipulation since. I explain this in the section below. I will run various regression models to predict my "next trading days' adjusted clsoing stock price". My benchmark model will be a simple linear regression model and the final model should be better then this simple linear regression. 

In this section, you will want to clearly define the problem that you are trying to solve, including the strategy (outline of tasks) you will use to achieve the desired solution. You should also thoroughly discuss what the intended solution will be for this problem. Questions to ask yourself when writing this section:


### Metrics

I will be leveraging sklearn in this project. From sklearn metrics we will have access to an array of metrics from our model.
The main evaluation metric I will be using is the root mean squared error.
The root mean squared is simple to calculate and can be calulated as shown below. Based on previous projects and experience, i don't think a metric exists to calculate this for us. This has to be derived and is simple to derive as can be seen below.

```sh
from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(y_actual, y_predicted))
```

## II. Analysis
_(approx. 2-4 pages)_

### Data Exploration
In this section, you will be expected to analyze the data you are using for the problem. This data can either be in the form of a dataset (or datasets), input data (or input files), or even an environment. The type of data should be thoroughly described and, if possible, have basic statistics and information presented (such as discussion of input features or defining characteristics about the input or environment). Any abnormalities or interesting qualities about the data that may need to be addressed have been identified (such as features that need to be transformed or the possibility of outliers). Questions to ask yourself when writing this section:
- _If a dataset is present for this problem, have you thoroughly discussed certain features about the dataset? Has a data sample been provided to the reader?_
- _If a dataset is present for this problem, are statistics about the dataset calculated and reported? Have any relevant results from this calculation been discussed?_
- _If a dataset is **not** present for this problem, has discussion been made about the input space or input data for your problem?_
- _Are there any abnormalities or characteristics about the input space or dataset that need to be addressed? (categorical variables, missing values, outliers, etc.)_

### Exploratory Visualization
In this section, you will need to provide some form of visualization that summarizes or extracts a relevant characteristic or feature about the data. The visualization should adequately support the data being used. Discuss why this visualization was chosen and how it is relevant. Questions to ask yourself when writing this section:
- _Have you visualized a relevant characteristic or feature about the dataset or input data?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Algorithms and Techniques
In this section, you will need to discuss the algorithms and techniques you intend to use for solving the problem. You should justify the use of each one based on the characteristics of the problem and the problem domain. Questions to ask yourself when writing this section:
- _Are the algorithms you will use, including any default variables/parameters in the project clearly defined?_
- _Are the techniques to be used thoroughly discussed and justified?_
- _Is it made clear how the input data or datasets will be handled by the algorithms and techniques chosen?_

### Benchmark

As suggested in the capstone proposal review, I will train and test an out-of-the-box linear regression model on the project data. My final solution should outperform this linear regression model. By using a linear regression model as the benchmark, and training/testing it on exactly the same data as my final solution I will be able to make a clear, objective comparison between the two.


In this section, you will need to provide a clearly defined benchmark result or threshold for comparing across performances obtained by your solution. The reasoning behind the benchmark (in the case where it is not an established result) should be discussed. Questions to ask yourself when writing this section:
- _Has some result or value been provided that acts as a benchmark for measuring performance?_
- _Is it clear how this result or value was obtained (whether by data or by hypothesis)?_


## III. Methodology
_(approx. 3-5 pages)_

### Data Preprocessing
In this section, all of your preprocessing steps will need to be clearly documented, if any were necessary. From the previous section, any of the abnormalities or characteristics that you identified about the dataset will be addressed and corrected here. Questions to ask yourself when writing this section:
- _If the algorithms chosen require preprocessing steps like feature selection or feature transformations, have they been properly documented?_
- _Based on the **Data Exploration** section, if there were abnormalities or characteristics that needed to be addressed, have they been properly corrected?_
- _If no preprocessing is needed, has it been made clear why?_

### Implementation
In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:
- _Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?_
- _Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?_
- _Was there any part of the coding process (e.g., writing complicated functions) that should be documented?_

### Refinement
In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:
- _Has an initial solution been found and clearly reported?_
- _Is the process of improvement clearly documented, such as what techniques were used?_
- _Are intermediate and final solutions clearly reported as the process is improved?_


## IV. Results
_(approx. 2-3 pages)_

### Model Evaluation and Validation
In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
- _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?_

### Justification
In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
- _Are the final results found stronger than the benchmark result reported earlier?_
- _Have you thoroughly analyzed and discussed the final solution?_
- _Is the final solution significant enough to have solved the problem?_


## V. Conclusion
_(approx. 1-2 pages)_

### Free-Form Visualization
In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Reflection


The final model and solution does fit my expectation.

In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_

### Improvement

I would have liked to do the following in my project.

- Allow the user the flexibility to select a certain stock and do the analysis presented in the project notebook on the stock. There is one immediate concern that comes to my mind. Since Broadcom is a semiconductor company and is a big Apple supplier, i decided to use Apple stock and the SMH ETF data for my analysis. These 2 will not be applicable to lets say Southwest Airlines. I will also need to code up more general user defined functions that can take a stock symbol and dataframe and perform all the necessary analysis.

- I would loved to have used neural networks and try out LTSM. The mian reason for not doing so is becuase I am not yet fully aware of LTSM and i didn't want to use something that i didn't ahve a good grasp of. If I had a good grasp of LTSM, i would defintely have tried using it in my project. LTSM should give good results for time series data. I intedn to try this out at a later date as time permits.

- Machine learning for stock trading and prediction is getting more and more complicated and fancy. I see sentiment analysis using tweet dat also being used. I think a better solution will defintely exist if we set up or analysis using more enhanced data points with data from tweets and using perhaps LTSM. Thus if my final solution were used as the new benchmark, i do believe an even better solution exists.

-----------





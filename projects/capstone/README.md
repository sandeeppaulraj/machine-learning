# Machine Learning Engineer Nanodegree

## Project: Stock Price Prediction

**Note**

There are no real special considerations to run this project.
I have provided all the necessary csv files in the project repository itself.

I am using Python 3 for my capstone project and there are several well known packages that need to be imported. These are imported in the first code cell in the ipython notebook. These are documented below as well.

```sh
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import cv2
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
from sklearn.svm import SVR
import random
```

The capstone proposal review can be seen at

[Capstone Proposal Review](https://review.udacity.com/#!/reviews/935978)

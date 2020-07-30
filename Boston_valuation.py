from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

import numpy as np
import pandas as pd

boston_dataset = load_boston()
data = pd.DataFrame(data=boston_dataset.data,columns=boston_dataset.feature_names)
# data.head()
features = data.drop(['INDUS','AGE'],axis=1)
log_prices = np.log(boston_dataset.target)
# log_prices.shape (converting to (506,1) shape)
target = pd.DataFrame(log_prices,columns=['PRICES'])
# target.shape


CRIM_IDX= 0
ZN_IDX=1
CHAS_IDX=2
NOX_IDX=3
RM_IDX=4
DIS_IDX=5
RAD_IDX=6
TAX_IDX=7
PTRATIO_IDX=8
B_IDX=9
LSTAT_IDX=10

property_stats = features.mean().values.reshape(1,11) #it acts like a template



regressor = LinearRegression()
regressor.fit(features,target)

fitted_values = regressor.predict(features)

#calculating MSE and RMSE

MSE = mean_squared_error(target,fitted_values)
RMSE = np.sqrt(MSE)


def get_log_estimate(nr_of_rooms,students_per_classroom,next_to_river=False,high_confidence=True):
    
    #configure property features
    property_stats[0][RM_IDX] = nr_of_rooms
    property_stats[0][PTRATIO_IDX] = students_per_classroom
    
    #checking requirement for next to river
    if next_to_river:
        property_stats[0][CHAS_IDX] = 1
    else:
        property_stats[0][CHAS_IDX] = 0
        
    # make prediction
    log_estimate = regressor.predict(property_stats)[0][0]
    
    #checking confidence high confidence(95%)
    if high_confidence:
        upperbound = log_estimate + 2*RMSE
        lowerbound = log_estimate - 2*RMSE
        interval = 95
    else:
        upperbound = log_estimate + RMSE
        lowerbound = log_estimate - RMSE
        interval = 68
    return log_estimate,upperbound,lowerbound,interval





# CREATING FUNCTOIN TO GET DOLLAR ESTIMATE

def get_dollar_estimate(rm, ptratio, chas=False ,high=True):
    
    '''
    keyword arguments:
    rm : number of rooms in house
    ptratio : ratio of nmber of students to teacher in a class
    chas :True if house near to river otherwise false
    high : True for 95% prediction interval false for 65% prediction interval
        
    '''
    
    
    if rm<1 or ptratio < 1:
        print('one or more parameter is unrealistic. Try providing correct paramaters')
        return
    
    CURRENT_MEDIAN_PRICE = 583.3
    SCALE_FACTOR = CURRENT_MEDIAN_PRICE/np.median(boston_dataset.target)
    SCALE_FACTOR

    log_estim,upper,lower,conf = get_log_estimate(nr_of_rooms=rm,students_per_classroom=ptratio,next_to_river=chas,high_confidence=high)

    # Converting them to dollar price as of todays prices
    dollar_price = np.e**log_estim  * 1000 * SCALE_FACTOR
    dollar_upper = np.e**upper  * 1000 * SCALE_FACTOR
    dollar_lower = np.e**lower  * 1000 * SCALE_FACTOR


    # Round the dollar value to nearest thousands

    rounded_dollar_price = np.around(dollar_price,-3)
    rounded_upper_price = np.around(dollar_upper,-3)
    rounded_lower_price = np.around(dollar_lower,-3)



    print(f'Estimated property value is :$ {rounded_dollar_price}')
    print(f'At {conf}% the valuation range is : ')
    print(f'USD {rounded_lower_price} is estimated lower price and USD {rounded_upper_price} is estimated higher price')
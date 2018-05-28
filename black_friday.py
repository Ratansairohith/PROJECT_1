# -*- coding: utf-8 -*-
"""
Created on Fri May 25 09:14:26 2018

@author: LENOVO
"""

#importing libraries
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

#creating dataframe
dataframe = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
#missing values
def missing_values_table(df):
        mis_val = df.isnull().sum()
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        # Return the dataframe with missing information
        return mis_val_table_ren_columns
missing_values_table(dataframe)
missing_values_table(test)
dataframe['Product_Category_3']=dataframe['Product_Category_3'].fillna(np.mean(dataframe['Product_Category_3']))
dataframe['Product_Category_2']=dataframe['Product_Category_2'].fillna(np.mean(dataframe['Product_Category_2']))
# Find all correlations and sort 
"""correlations_data = dataframe.corr()['Purchase'].sort_values()
# Print the most negative correlations
print(correlations_data.head(10), '\n')
#predicting the missing values
#from sklearn.linear_model import LinearRegression
#regressor = LinearRegression()
#data_with_null = dataframe[[]]
#exploratory data analysis
sns.distplot(dataframe['Purchase'],bins = 20)
#dataframe['Product_Category_1'] = np.log[dataframe['Product_Category_1']]
#sns.distplot(np.log[dataframe['Product_Category_1']], bins = 20)
#dataframe['Product_Category_1'] = dataframe['Product_Category_1'].apply(np.log)
sns.boxplot(dataframe['Purchase'])"""
data = [dataframe,test]
gender = {'M':1,'F':0}
for dataset in data:
    dataset['Gender'] = dataset['Gender'].map(gender)
#now lets see which gender brought with product highly
"""product_data = ['Product_Category_1','Product_Category_2','Product_Category_3']  
for i in product_data :
    plt.figure(i)
    sns.barplot(x = 'Gender', y=i, data = dataframe.sort_values(i).dropna())
    plt.figure(i)
    factor = sns.factorplot(x = 'Age', y=i, data = dataframe.sort_values(i).dropna(),size=5, aspect=.8) 
sns.barplot(x = 'Age', y= 'Marital_Status', data = dataframe.sort_values('Marital_Status').dropna())"""
#now we will divide age into lower age and upper age
dataframe['lower_age'] = " "
dataframe['upper_age'] = " "
test['lower_age'] = " "
test['upper_age'] = " "
data = [dataframe,test]
for dataset in data:
    dataset.loc[dataset.Age == '0-17',['lower_age','upper_age']] = 0,17
    dataset.loc[dataset.Age == '26-35',['lower_age','upper_age']] = 26,35
    dataset.loc[dataset.Age == '36-45',['lower_age','upper_age']] = 36,45
    dataset.loc[dataset.Age == '51-55',['lower_age','upper_age']] = 51,55
    dataset.loc[dataset.Age == '46-50',['lower_age','upper_age']] = 46,50
    dataset.loc[dataset.Age == '18-25',['lower_age','upper_age']] = 18,25
    dataset.loc[dataset.Age == '55+',['lower_age','upper_age']] = 55,65
    dataset.loc[dataset.City_Category == 'A',['City_Category']] = 1
    dataset.loc[dataset.City_Category == 'B',['City_Category']] = 2
    dataset.loc[dataset.City_Category== 'C',['City_Category']] = 3
    dataset.loc[dataset.Stay_In_Current_City_Years == '4+',['Stay_In_Current_City_Years']] = 5
dataframe = dataframe.drop(['Age'],axis = 1)   
test = test.drop(['Age'],axis = 1) 
sns.barplot(x ='City_Category',y = 'Product_Category_1',hue = 'Gender',data = dataframe.dropna())     
sns.barplot(x ='City_Category',y = 'Product_Category_2',hue = 'Gender',data = dataframe.dropna()) 
sns.barplot(x ='City_Category',y = 'Product_Category_3',hue = 'Gender',data = dataframe.dropna())
sns.barplot(x ='City_Category',y = 'Purchase',hue = 'Gender',data = dataframe.dropna())
sns.barplot(x ='Occupation',y = 'Purchase',hue ='City_Category' ,data = dataframe.dropna())
#generating heatmap for whole data
corr = dataframe.corr()
sns.heatmap(corr, xticklabels=corr.columns.values,yticklabels=corr.columns.values)
#function to drop features that are highly correlated to each other .otherthan target
def corr_df(x, corr_val):
    y = x['Purchase']
    x = x.drop(columns = ['Purchase'])
      
    corr_matrix = x.corr()
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = [] 
    for i in iters:
        for j in range(i):
            item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
            col = item.columns
            row = item.index
            val = item.values
            if val >= corr_val:
                print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                drop_cols.append(i)

    drops = sorted(set(drop_cols))[::-1]
    for i in drops:
        col = x.iloc[:, (i+1):(i+2)].columns.values
        x = x.drop(col, axis=1)

    x['Purchase']=y
    return x
dataframe = corr_df(dataframe, 0.6);
#features_test = corr_df(test, 0.7);
dataframe.shape

#chanding the order of dataframe columns
dataframe = dataframe[['User_ID', 'Product_ID', 'Gender', 'Occupation', 'City_Category',
       'Stay_In_Current_City_Years', 'Marital_Status', 'Product_Category_1','Product_Category_2', 'Product_Category_3','lower_age','upper_age','Purchase']]
features = dataframe.drop(columns=['Purchase','User_ID', 'Product_ID'])
target = dataframe['Purchase']
#split the training dataset into train and test
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size = 0.2, random_state = 42)
#now astablish the baseline score and function for rmse
#from sklearn.metrics import mean_squared_error
#from math import sqrt
#def rmse(y_true, y_pred):
 #   return(sqrt(mean_squared_error(y_test, baseline_guess)))
baseline_guess = np.median(y_train)    
print('The baseline guess is a score of %0.2f' % baseline_guess)
#print("Baseline Performance on the test set: RMSE =",rmse)  
#feature_importances = pd.DataFrame(rf.feature_importances_,index = X_train.columns,columns=['importance']).sort_values('importance',ascending=False)   
#fitting the models liner regression
#we will first fit the basic linear regression model
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(x_train,y_train)
y_pred1 = linear_regressor.predict(x_test)
acc_linear_resgressor_train = round(linear_regressor.score(x_train, y_train) * 100, 2)
rmse_linear_regressor = mean_squared_error(y_test, y_pred1)
#now we will fir the knn
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=16)
knn.fit(x_train,y_train)
y_pred2 = knn.predict(x_test)
acc_knn_train = round(knn.score(x_train, y_train) * 100, 2)
rmse_knn = mean_squared_error(y_test, y_pred2)

#we fit  ensemble algorithms
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
randomforest = RandomForestRegressor(n_estimators = 500)
randomforest.fit(x_train,y_train)
y_pred3 = randomforest.predict(x_test)
acc_randomforest_train = round(randomforest.score(x_train, y_train) * 100, 2)
rmse_randomforest = mean_squared_error(y_test, y_pred3)

gradientboost = GradientBoostingRegressor(n_estimators = 300)
gradientboost.fit(x_train,y_train)
y_pred4 = gradientboost.predict(x_test)
acc_gradientboost_train = round(gradientboost.score(x_train, y_train) * 100, 2)
rmse_gradientboost = mean_squared_error(y_test, y_pred4)

#creating new dataframes to visualising the error test acc and train acc
exam_dataframe = pd.DataFrame({'model': ['linear_regressor','knn','randomforest','gradientboost'],
                              'train accuracy':[acc_linear_resgressor_train,acc_knn_train,acc_randomforest_train,acc_gradientboost_train],
                               'error':[rmse_linear_regressor,rmse_knn,rmse_randomforest,rmse_gradientboost]})
result_df = exam_dataframe.sort_values(by='train accuracy', ascending=False)
result_df = result_df.set_index('train accuracy')
result_df.head()
feature_importances = pd.DataFrame(randomforest.feature_importances_,index = x_train.columns,columns=['importance']).sort_values('importance',ascending=False)
feature_importances = pd.DataFrame(gradientboost.feature_importances_,index = x_train.columns,columns=['importance']).sort_values('importance',ascending=False)
# improving model
from sklearn.cross_validation import cross_val_score
scores_knn = cross_val_score(knn, features,target, cv=10, scoring='mean_squared_error')
print(scores_knn)
mse_scores_knn = -scores_knn
rmse_scores_knn = np.sqrt(mse_scores_knn)
print(rmse_scores_knn.mean())

scores_grad = cross_val_score(gradientboost, features,target, cv=10, scoring='mean_squared_error')
print(scores_grad)
mse_scores_grad = -scores_grad
rmse_scores_grad = np.sqrt(mse_scores_grad)
print(rmse_scores_grad.mean())

scores_rf = cross_val_score(randomforest, features,target, cv=10, scoring='mean_squared_error')
print(scores_rf)
mse_scores_rf = -scores_rf
rmse_scores_rf = np.sqrt(mse_scores_rf)
print(rmse_scores_rf.mean())

        
#its time for hyperparameter tuning
# Loss function to be optimized
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
loss = ['ls', 'lad', 'huber']

# Number of trees used in the boosting process
n_estimators = [100, 500, 900, 1100, 1500]

# Maximum depth of each tree
max_depth = [2, 3, 5, 10, 15]

# Minimum number of samples per leaf
min_samples_leaf = [1, 2, 4, 6, 8]

# Minimum number of samples to split a node
min_samples_split = [2, 4, 6, 10]

# Maximum number of features to consider for making splits
max_features = ['auto', 'sqrt', 'log2', None]

# Define the grid of hyperparameters to search
hyperparameter_grid = {'loss': loss,
                       'n_estimators': n_estimators,
                       'max_depth': max_depth,
                       'min_samples_leaf': min_samples_leaf,
                       'min_samples_split': min_samples_split,
                       'max_features': max_features}

# Set up the random search with 4-fold cross validation
random_cv = RandomizedSearchCV(estimator=gradientboost,
                               param_distributions=hyperparameter_grid,
                               cv=10, n_iter=15, 
                               scoring = 'neg_mean_absolute_error',
                               n_jobs = -1, verbose = 1, 
                               return_train_score = True,
                               random_state=42)
random_cv.fit(features,target)
    
    
    



#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from helpers.eda import *
from helpers.data_prep import *


# In[3]:


import pickle
from warnings import filterwarnings
filterwarnings("ignore")


# In[4]:


from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from sklearn.model_selection import GridSearchCV,cross_validate,train_test_split,validation_curve
from sklearn.metrics import mean_absolute_error,mean_squared_error

from sklearn.decomposition import PCA


# In[5]:


######################################
# Modeling
######################################


# In[6]:


df = pickle.load(open("House_Price.pkl","rb"))
for col in df.columns:
    df[col] = df[col].astype(float)
    
cat_cols,num_cols,cat_but_car = grab_col_names(df,details=True)


# In[7]:


df.head()


# In[8]:


scaler_num_cols = ['MSSubClass',
 'LotFrontage',
 'LotArea',
 'YearBuilt',
 'YearRemodAdd',
 'Exterior1st',
 'Exterior2nd',
 'MasVnrArea',
 'BsmtFinSF1',
 'BsmtFinSF2',
 'BsmtUnfSF',
 'TotalBsmtSF',
 '1stFlrSF',
 '2ndFlrSF',
 'LowQualFinSF',
 'GrLivArea',
 'GarageYrBlt',
 'GarageArea',
 'WoodDeckSF',
 'OpenPorchSF',
 'EnclosedPorch',
 '3SsnPorch',
 'ScreenPorch',
 'PoolArea',
 'PoolQC',
 'MiscVal',
 'MoSold',
 'GarageCarSize',
 'HomeRepairYear',
 'TotalSF',
 'SqFtPerRoom',
 'TotalPorchSF',
 'ConditionScore',
 'TotalBsmScore',
 'TotalQual',
 'QualGr']


# In[9]:


scaler_num_cols == [col for col in num_cols if col not in ["Id","SalePrice"]]


# In[10]:


print("DataFrame Shape : {}".format(df.shape))


# In[11]:


train_df = df[df["SalePrice"].notnull()]
test_df = df[df["SalePrice"].isnull()].drop("SalePrice", axis = 1)


# In[12]:


X = train_df.drop(["Id","SalePrice"], axis = 1)
y = np.log1p(train_df["SalePrice"])


# In[13]:


ridge_params = {"solver" : ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
                "alpha" : [1e-15,1e-8,1e-3,1,5,10,30,50,100] , 
                "max_iter" : [500,700,900,]}

cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

rf_params = {"max_depth": [ 8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [8, 15, 20],
             "n_estimators": [ 500, 1000]}

gbm_params = {"learning_rate": [0.01, 0.1],
              "max_depth": [3, 8],
              "n_estimators": [500, 1000],
              "subsample": [1, 0.5, 0.7]}

#gbm_params = {"learning_rate": [ 0.1, 0.2 ,0.3 ,0.4],
#              "n_estimators": [ 300, 400, 500],
#              "max_depth": [ 5, 8, 12],
#              "subsample":[0.6 ,0.7 ,0.8 ,0.9]}


lightgbm_params = {"boosting_type":['gbdt','dart','goss'],
                   "learning_rate": [0.05, 0.07, 0.1, 0.2],
                   "n_estimators": [ 300, 400, 500],
                   "max_depth": [3, 5, 8],
                   "colsample_bytree": [ 0.5, 0.7, 0.8, 1]}

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8, 12, 20],
                  "n_estimators": [100, 200],
                  "colsample_bytree": [0.5, 0.8, 1]}

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500, 1500],
                   "colsample_bytree": [0.5, 0.7, 1]}

knn_params = {"n_neighbors": range(2, 50)}

regressors = [   
                
                #("Ridge" , Ridge(),ridge_params),
                #("Lasso" , Lasso()),
                #("ElasticNet", ElasticNet()),
                #("KNN", KNeighborsRegressor(),knn_params),
                ("CART" , DecisionTreeRegressor(),cart_params),
                #("RF",RandomForestRegressor(),rf_params),
                #("SVR", SVR()),
                ("GBM" , GradientBoostingRegressor(),gbm_params),
                ("XGBoost" , XGBRegressor(objective = "reg:squarederror"),xgboost_params),
                ("LightGBM" , LGBMRegressor(),lightgbm_params)
            
            ]


# In[14]:


#########################
# Base Model
#########################


# In[34]:


def base_model_regressors(regressors, X , y, cv =5):
    
    data = pd.DataFrame()
    index = 0
    
    for name,regressor,params in regressors:
        
        cv_results = cross_validate(estimator=regressor,
                                    X = X,
                                    y = y,
                                    cv = cv,
                                    scoring = "neg_mean_squared_error")
        
        rmse = np.sqrt(-cv_results["test_score"].mean())
        fit_time = cv_results["fit_time"].mean()
        score_time = cv_results["score_time"].mean()
    
        data.loc[index, "NAME"] = name
        data.loc[index, "RMSE"] = rmse
        data.loc[index, "FIT__TIME"] = fit_time
        data.loc[index, "SCORE_TIME"] = score_time
        index+=1
    
    data = data.sort_values(by = "RMSE")
    data = data.set_index("NAME")
    data = data.T
    
    return data


# In[16]:


base_model_regressors(regressors,X,y,cv = 5)


# In[17]:


#########################
# HyperParameter Optimization Model
#########################


# In[35]:


def hyperparameter_optimization_model(regressors,X,y,cv=5):
    
    index = 0
    data = pd.DataFrame()
    models_dict = {}
    
    
    for name, regressor, params in regressors:
        
        cv_results = cross_validate(estimator=regressor,
                                    X = X,
                                    y = y, 
                                    cv = cv,
                                    n_jobs=-1,
                                    verbose=0,
                                    scoring = "neg_mean_squared_error")
        
        rmse = np.sqrt(-cv_results["test_score"].mean())
        
        print("".center(50,"#"),end = "\n\n")
        
        print(f"For {type(regressor).__name__.upper()} Model", end = "\n\n")
        
        print(f"Before Grid Search :\n\nRMSE : {rmse}", end = "\n\n")
              
        best_grid = GridSearchCV(estimator=regressor,
                                 param_grid=params,
                                 cv = cv,
                                 n_jobs=-1,
                                 scoring="neg_mean_squared_error",
                                 verbose=0).fit(X,y)
              
        print(f"Best Grid : {best_grid.best_params_}",end = "\n\n")
        
        final_model = regressor.set_params(**best_grid.best_params_)
        
        models_dict[name] = final_model
        
        final_cv_results = cross_validate(estimator = final_model,
                                          X = X,
                                          y = y, 
                                          cv = cv,
                                          scoring = "neg_mean_squared_error",
                                          n_jobs = -1,
                                          verbose = 0)
        
        final_rmse = np.sqrt(-final_cv_results["test_score"].mean())
        
        print(f"After Grid Search:\n\nRMSE : {final_rmse}",end = "\n\n")
        
        data.loc[index, "NAME"] = name.upper()
        data.loc[index, "BEFORE_RMSE"] = rmse
        data.loc[index, "AFTER_RMSE"] = final_rmse
        index+=1
        
        
    data = data.sort_values(by = "AFTER_RMSE")
    data = data.set_index("NAME")
    data = data.T
    
    return data,models_dict


# In[19]:


regressors = [   
                
                ("Ridge" , Ridge(),ridge_params),
                #("Lasso" , Lasso()),
                #("ElasticNet", ElasticNet()),
                ("KNN", KNeighborsRegressor(),knn_params),
                ("CART" , DecisionTreeRegressor(),cart_params),
                #("RF",RandomForestRegressor(),rf_params),
                #("SVR", SVR()),
                #("GBM" , GradientBoostingRegressor(),gbm_params),
                ("XGBoost" , XGBRegressor(objective = "reg:squarederror"),xgboost_params),
                ("LightGBM" , LGBMRegressor(),lightgbm_params)
            
            ]


# In[20]:


data , models_dict = hyperparameter_optimization_model(regressors, X , y, cv = 5)


# In[21]:


data


# In[22]:


models_dict


# In[23]:


for name,regressor,params in regressors:
    
    models_dict[name].fit(X,y)
    
    pd.to_pickle(models_dict[name], open("Final_"+name.upper()+"_Model.pkl","wb"))


# In[24]:


#########################
# VotingRegressor Model
#########################


# In[25]:


from sklearn.ensemble import VotingRegressor


# In[36]:


def votingregressor_model(estimators,X,y,cv = 5):
    
    data = pd.DataFrame()
    index = 0
    
    voting_regressor = VotingRegressor(estimators=estimators,
                                       n_jobs=-1,
                                       verbose=False)
    
    voting_regressor = voting_regressor.fit(X,y)
    
    cv_results = cross_validate(estimator=voting_regressor,
                                X = X,
                                y = y,
                                cv = cv,
                                n_jobs=-1,
                                verbose=0,
                                scoring = "neg_mean_squared_error")
    
    data.loc[index,"NAME"] = ["VotingRegressor"]
    data.loc[index,"RMSE"] = np.sqrt(-cv_results["test_score"].mean())
    data.loc[index,"FIT_TIME"] = cv_results["fit_time"].mean()
    data.loc[index,"SCORE_TIME"] = cv_results["score_time"].mean()
    
    data = data.set_index("NAME")
    
    return data


# In[27]:


estimators = [ (name,models_dict[name]) for name,regressor,params in regressors if name not in ["Ridge","KNN"]]
estimators


# In[28]:


voting_data = votingregressor_model(estimators,X,y, cv = 5)


# In[29]:


data


# In[30]:


voting_data


# In[31]:


################################################
# Feature Importance
################################################


# In[37]:


def feature_importance(models,X,num=10,save = False):
    
    data = pd.DataFrame({"RATIO" : models.feature_importances_,
                         "FEATURES" : X.columns}).sort_values(by = "RATIO", ascending = False)[0:num]
    
    plt.figure(figsize = (10,7))
    sns.barplot(y = data["FEATURES"], x = data["RATIO"])
    plt.title("FEAURES")
    plt.show()
    
    if save:
        plt.savefig(type(models).__name__+"importance.png")


# In[33]:


feature_importance(models_dict["LightGBM"],X,num = 15, save = True)


# In[34]:


################################################
# Analyzing Model Complexity with Learning Curves (BONUS)
################################################


# In[35]:


models_dict


# In[36]:


new_lightgbm_params = {"learning_rate" : [0.01, 0.1, 0.2, 0.3],
                       "n_estimators" : [300,500,1500],
                       "colsample_bytree": [0.5, 0.7, 1]}
new_lightgbm_params


# In[38]:


def val_curve_params(estimator, X ,y , param_name, param_range, scoring, cv = 5 ):
    
    train_scores ,test_scores = validation_curve(estimator = estimator, 
                                                 X = X,
                                                 y = y,
                                                 cv = cv,
                                                 param_name = param_name,
                                                 param_range = param_range,
                                                 n_jobs = -1,
                                                 verbose = 0,
                                                 scoring = scoring)
    
    mean_train_scores = np.mean(train_scores, axis = 1)
    mean_test_scores = np.mean(test_scores, axis = 1)
    
    plt.plot(param_range, mean_train_scores, label = " TRANING SCORES ", color = "g")
    plt.plot(param_range, mean_test_scores, label = " VALIDATION SCORES ", color = "r")
    
    plt.title(f"Validation Curve For {type(estimator).__name__.upper()}")
    plt.xlabel(f"Number Of {param_range} : {param_name}")
    plt.ylabel(f"{scoring}")
    plt.legend(loc = "best")
    plt.show()


# In[38]:


for i in new_lightgbm_params:
    print(i,":",models_dict["LightGBM"].get_params()[i])


# In[39]:


for i in new_lightgbm_params:
    
    val_curve_params(estimator = models_dict["LightGBM"], 
                     X = X, 
                     y = y, 
                     param_name = i,
                     param_range = new_lightgbm_params[i],
                     scoring="neg_mean_squared_error")


# In[40]:


######################################################
# Prediction for a New Observation
######################################################


# In[41]:


random_select = X.sample(1, random_state = 14)
random_target = y.loc[random_select.index].values
random_target


# In[42]:


models_dict["LightGBM"].predict(random_select)


# In[43]:


######################################
# Actual Values vs Estimated Value
######################################


# In[44]:


y_pred = models_dict["LightGBM"].predict(X)
rmse = np.sqrt(mean_squared_error(y,y_pred))
rmse = round(rmse,4)


plt.figure(figsize = (15,7))
plt.plot(y,y, color = "g")
plt.scatter(y_pred,y_pred, color = "r")
plt.title(f"For Train Actual Vs Estimated (RMSE : {rmse})")
plt.show()


# In[45]:


######################################
#Loading Results
######################################


# In[46]:


submission_df = pd.DataFrame()
submission_df["Id"] = test_df["Id"].astype(int)


# In[47]:


y_pred_sub = models_dict["LightGBM"].predict(test_df.drop("Id",axis = 1))
y_pred_sub = np.expm1(y_pred_sub)

submission_df["SalePrice"] = y_pred_sub
submission_df.to_csv("submission.csv",index = False)


# In[48]:


submission_df


# In[ ]:





# In[49]:


################################
# Principal Component Analysis
################################


# In[14]:


cat_cols, num_cols, cat_but_car =grab_col_names(df,details=True)


# In[15]:


scaler_num_cols == [col for col in num_cols if col not in ["Id","SalePrice"]]


# In[16]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[scaler_num_cols] = scaler.fit_transform(df[scaler_num_cols])

pca = PCA()
pca_fit = pca.fit_transform(df[scaler_num_cols])
pca_fit


# In[17]:


np.cumsum(pca.explained_variance_ratio_)


# In[18]:


################################
# Optimum Number of Components
################################


# In[21]:


number_size = len(np.cumsum(pca.explained_variance_ratio_))


# In[22]:


plt.figure(figsize = (15,10))
plt.plot(np.cumsum(pca.explained_variance_ratio_)* 100,"--",color = "g")
plt.plot(15,84,"o",color = "r", markersize = 20)

plt.axhline(84,linestyle = "--",color ="purple",linewidth = .5)
plt.axvline(15,linestyle = "--",color ="purple",linewidth = .5)

plt.xticks(range(0,number_size))
plt.yticks(range(22,100))

size = 15
plt.title("PCA Graph",fontsize = size)
plt.xlabel("Number of Components" , fontsize = size)
plt.ylabel("Cumulative Rate of Variance" , fontsize = size)

plt.xlim([10,25])
plt.ylim([65,100])
sns.set()

plt.tight_layout()
plt.show()


# In[23]:


################################
# Creation of Final PCA
################################


# In[24]:


final_pca = PCA(n_components=15)
final_pca_fit = final_pca.fit_transform(df[scaler_num_cols])

np.cumsum(final_pca.explained_variance_ratio_)


# In[25]:


print("Final PCA Shape : {}".format(final_pca_fit.shape))


# In[26]:


pca_columns = ["PCA_"+str(i) for i in range(1,16)]

pca_data = pd.DataFrame(data = final_pca_fit, columns = pca_columns)
pca_data


# In[27]:


print("DataFrame Shape : {}".format(df.shape))


# In[28]:


drop_df = df.drop(scaler_num_cols, axis = 1)
pca_df = pd.concat([drop_df,pca_data],axis = 1)

print("PCA DataFrame Shape : {}".format(pca_df.shape))


# In[29]:


pca_df.head()


# In[30]:


pca_train_df = pca_df[pca_df["SalePrice"].notnull()]
pca_test_df = pca_df[pca_df["SalePrice"].isnull()].drop("SalePrice",axis = 1)


# In[31]:


pca_X = pca_train_df.drop(["Id","SalePrice"],axis = 1)
pca_Y = np.log1p(pca_train_df["SalePrice"])


# In[32]:


################################
# Base Models
################################


# In[39]:


base_model_regressors(regressors,pca_X,pca_Y,cv = 5)


# In[186]:


#########################
# HyperParameter Optimization Model
#########################


# In[40]:


regressors = [   
                
                #("Ridge" , Ridge(),ridge_params),
                #("Lasso" , Lasso()),
                #("ElasticNet", ElasticNet()),
                #("KNN", KNeighborsRegressor(),knn_params),
                ("CART" , DecisionTreeRegressor(),cart_params),
                #("RF",RandomForestRegressor(),rf_params),
                #("SVR", SVR()),
                #("GBM" , GradientBoostingRegressor(),gbm_params),
                ("XGBoost" , XGBRegressor(objective = "reg:squarederror"),xgboost_params),
                ("LightGBM" , LGBMRegressor(),lightgbm_params)
            
            ]


# In[41]:


PCA_data, PCA_models_dict = hyperparameter_optimization_model(regressors,pca_X,pca_Y,cv=5)


# In[43]:


PCA_data


# In[44]:


PCA_models_dict


# In[42]:


for name,regressor,params in regressors:
    
    PCA_models_dict[name].fit(pca_X,pca_Y)
    pd.to_pickle(PCA_models_dict[name], open("PCA_Final_"+name+"_Model.pkl", "wb"))


# In[45]:


################################################
# Feature Importance
################################################


# In[46]:


feature_importance(PCA_models_dict["LightGBM"],pca_X,num = 20, save = True)


# In[47]:


################################################
# Analyzing Model Complexity with Learning Curves (BONUS)
################################################


# In[51]:


for i in lightgbm_params:
    print(i, ":", PCA_models_dict["LightGBM"].get_params()[i])


# In[52]:


for i in lightgbm_params:
    val_curve_params(PCA_models_dict["LightGBM"], 
                     pca_X, 
                     pca_Y, 
                     param_name=i,
                     param_range = lightgbm_params[i],
                     scoring = "neg_mean_squared_error",
                     cv = 5)


# In[53]:


######################################################
# Prediction for a New Observation
######################################################


# In[57]:


random_select_pca = pca_X.sample(1,random_state = 42)
random_pca_values = pca_Y.loc[random_select_pca.index].values
random_pca_values


# In[59]:


PCA_models_dict["LightGBM"].predict(random_select_pca)


# In[60]:


######################################
# Actual Values vs Estimated Values
######################################


# In[61]:


y_pred_pca = PCA_models_dict["LightGBM"].predict(pca_X)

plt.figure(figsize = (10,7))
plt.plot(pca_Y, pca_Y, color = "g",label = "Actual_Values")
plt.scatter(y_pred_pca,y_pred_pca, color = "r", label = "Estimated_Values")

plt.title("Actual Values & Estimated Values")

plt.show()


# In[62]:


######################################
#Loading Results
######################################


# In[63]:


submission_df_pca = pd.DataFrame()
submission_df_pca["Id"] = test_df["Id"].astype(int)


# In[68]:


y_pred_pca_test = PCA_models_dict["LightGBM"].predict( pca_test_df.drop("Id",axis = 1) )

submission_df_pca["SalePrice"] = np.expm1(y_pred_pca_test)

submission_df_pca.to_csv("Submission_PCA.csv",index = False)


# In[69]:


submission_df_pca.head()


# In[ ]:





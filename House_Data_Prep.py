#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import pickle


# In[3]:


from helpers.eda import *
from helpers.data_prep import *


# In[4]:


from warnings import filterwarnings
filterwarnings("ignore")


# In[5]:


pd.set_option("display.max_columns" , None)
pd.set_option("display.float_format", lambda x : "%.4f" %x)
pd.set_option("display.width" , 200)


# In[6]:


train_path = "/Users/gokhanersoz/Desktop/VBO_Dataset/house_prices/train.csv"
test_path = "/Users/gokhanersoz/Desktop/VBO_Dataset/house_prices/test.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
df = train.append(test).reset_index(drop =True)

print("Train Shape : {}".format(train.shape))
print("Test Shape : {}".format(test.shape))
print("DataFrame Shape : {}".format(df.shape))


# In[7]:


check_df(df)


# In[8]:


df.SalePrice.nunique()


# In[9]:


def na_values(dataframe, plot = False):
    
    na_values = dataframe.isnull().sum()
    na_values = na_values[na_values > 0 ]
    na_values = pd.DataFrame(na_values)
    na_values.columns = ["NA_Values"]
    na_values["Ratio"] = round(na_values["NA_Values"] / len(dataframe) * 100 , 5)
    na_values = na_values.sort_values("Ratio" , ascending = False)
    
    if plot :
        
        x = na_values.index
        x = [str(i)+"_"+str(j) for i,j in zip(x,na_values.Ratio)]
        y = na_values["Ratio"]

        size = 15
        plt.figure(figsize = (15,7))
        sns.barplot(x, y)
        plt.xticks(rotation = 90,fontsize = size)
        plt.ylabel("RATIO" , fontsize = size)
        plt.title("NA VALUES", fontsize = size)
        plt.show()
        
    return na_values


# In[10]:


na_ = na_values(df,plot = True)
na_.T


# In[11]:


def high_correlated_cols(dataframe, plot = False, corr_th = .90):
    
    corr = dataframe.corr()
    corr_matrix = corr
    upper_triangle_matrix = corr_matrix.where(np.tril(np.ones(corr_matrix.shape),k = -1).astype(np.bool))
    drop_list = [col for col in upper_triangle_matrix.columns if (any(upper_triangle_matrix[col] > corr_th))]
    
    if plot :
        
        plt.figure(figsize = (20,10))
        sns.heatmap(upper_triangle_matrix, cmap = "viridis", fmt = ".1f", annot = True, linewidths=.1)
        plt.show()
        
    return drop_list


# In[12]:


drop_list = high_correlated_cols(df,plot = True)
drop_list


# In[13]:


# PoolQC : Pool Quality
df["PoolQC"] = df["PoolArea"].fillna("None")

# MiscFeature : Various features not covered in other categories
df["MiscFeature"] = df["MiscFeature"].fillna("None")

# Alley : Alley access type
df["Alley"] = df["Alley"].fillna("None")

# Fence : Fence quality
df["Fence"] = df["Fence"].fillna("None")

# FireplaceQu : Fireplace quality
df["FireplaceQu"] = df["FireplaceQu"].fillna("None")

for col in ["PoolQC","MiscFeature","Alley","Fence","FireplaceQu"]:
    print(f"{col} : {df[col].dtype}")

#GarageType: Garage location
#GarageFinish: Garage interior finish
#GarageQual: Garage quality
#GarageCond: Garage status

for col in ["GarageType","GarageFinish","GarageQual","GarageCond"]:
    print(f"{col} : {df[col].dtype}")
    df[col] = df[col].fillna("None")
    
#BsmtFinType1: Quality of basement finished area
#BsmtFinType2: Quality of the second finished field (if any)
#BsmtQual: The height of the basement
#BsmtCond: General condition of the basement
#BsmtExposure: Walk-out or garden level basement walls

for col in ["BsmtFinType1","BsmtFinType2","BsmtQual","BsmtCond","BsmtExposure"]:
    print(f"{col} : {df[col].dtype}")
    df[col] = df[col].fillna("None")


# In[14]:


#GarageYrBlt: The year the garage was built
#GarageArea: The size of the garage in square meters
#GarageCars: The size of the garage in terms of vehicle capacity

for col in ["GarageYrBlt","GarageArea","GarageCars"]:
    print(f"{col} : {df[col].dtype}")
    df[col] = df[col].fillna(0)
    

#BsmtFinSF1: Type 1 finished square feet
#BsmtFinSF2: Type 2 finished square feet
#BsmtUnfSF: Unfinished square foot basement space
#TotalBsmtSF: Total square meters of basement area
#BsmtFullBath: Basement bathrooms
#BsmtHalfBath: Basement half baths
    
for col in ["BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF","BsmtFullBath","BsmtHalfBath"]:
    print(f"{col} : {df[col].dtype}")
    df[col] = df[col].fillna(0)


# In[15]:


#MasVnrType: Wall cladding type
#MasVnrArea: Wall covering area in square meters

df["MasVnrType"] = df["MasVnrType"].fillna("None")
df["MasVnrArea"] = df["MasVnrArea"].fillna(0)

for col in ["MasVnrType","MasVnrArea"]:
    print(f"{col} : {df[col].dtype}") 


# In[16]:


# Functional : Home functionality degree
df["Functional"] = df["Functional"].fillna(df["Functional"].mode()[0])

# Electrical : Electrical System
df["Electrical"] = df["Electrical"].fillna(df["Electrical"].mode()[0])

# KitchenQual
df["KitchenQual"] = df["KitchenQual"].fillna(df["KitchenQual"].mode()[0])

#Exterior1st: The exterior of the house
#Exterior2nd: House siding
df["Exterior1st"] = df["Exterior1st"].fillna(df["Exterior1st"].mode()[0])
df["Exterior2nd"] = df["Exterior2nd"].fillna(df["Exterior2nd"].mode()[0])

#SaleType : Sales status
df["SaleType"] = df["SaleType"].fillna(df["SaleType"].mode()[0])

# Utilities
df["Utilities"] = df["Utilities"].fillna(df["Utilities"].mode()[0])

for col in ['Electrical', 'Exterior1st', 'Exterior2nd', 'Functional', 'KitchenQual','SaleType', 'Utilities']:
    print(f"{col} : {df[col].dtype}")


# In[17]:


# LotFrontage : Linear feet of the street connected to the property

df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(lambda x : x.fillna(x.median()))


# In[18]:


# MSZoning: General zoning classification
#MSSubClass: Building class

df["MSZoning"] = df.groupby("MSSubClass")["MSZoning"].transform(lambda x : x.fillna(x.mode()[0]))


# In[19]:


df[["MSSubClass","YrSold","MoSold"]].head()


# In[20]:


for col in ["MSSubClass","YrSold","MoSold"]:
    print(f"{col} : {df[col].dtype}")
    df[col] = df[col].astype(str)


# In[21]:


na_values(df,plot = True)


# In[22]:


cat_cols,num_cols,cat_but_car = grab_col_names(df, details= True)


# In[23]:


def rare_encoder(dataframe, rare_perc, cat_cols):
    """
    
    Object sınıfı alınıyor unutma !!!!
    
    """
    
    temp_df = dataframe.copy()
    
    rare_columns = [col for col in cat_cols if temp_df[col].dtype == "object" 
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis = None)]
    
    #rare_columns = [col for col in cat_cols \
    #                    if (temp_df[col].value_counts() / len(temp_df) < 0.01).sum() > 1]
    
    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels) , "Rare" , temp_df[var])
    
    return temp_df


# In[24]:


def boxplot_cat(dataframe , cat_cols, target ):
    
    i = 1
    plt.figure(figsize = (30,100))
    
    for col in cat_cols:
        plt.subplot(18,3,i)
        
        order = dataframe.groupby(col)[target].median().sort_values(ascending = False)
        sns.boxplot(x = dataframe[col], y = dataframe[target], order = order.index)
        plt.xticks(rotation = 45,fontsize = 15)
        plt.xlabel(col,fontsize = 15)
        plt.ylabel("SalePrice", fontsize = 15)
        i+=1
        plt.tight_layout()
        
    plt.show()


# In[25]:


def scatterplot_num(dataframe , num_cols, target):
    
    i = 1
    plt.figure(figsize = (30,80))
    size = 20
    
    for num in num_cols:
        plt.subplot(15,3,i)
        sns.scatterplot(x = dataframe[num], y = dataframe[target])
        plt.xticks(rotation = 45, fontsize = size)
        plt.xlabel(num, fontsize = size)
        plt.ylabel(target, fontsize = size)
        plt.tight_layout()
        
        i+=1
    plt.show()
        


# In[26]:


def dtypes_nunique(dataframe):
    data = pd.DataFrame()
    data["Name"] = [col for col in dataframe.columns]
    data["Dtype"] = [dataframe[col].dtype for col in dataframe.columns]
    data["Nunique"] = [dataframe[col].nunique() for col in dataframe.columns]
    data = data.set_index("Name")
    data = data.T
    return data


# In[27]:


boxplot_cat(df, cat_cols, "SalePrice")


# In[28]:


rare_analyser(df,"SalePrice" ,cat_cols)


# In[29]:


dictionary = {"MSZoning" : {"FV" : 3 ,"RL": 3 ,"RH": 2 , "RM": 2, "C (all)" : 1 },
 "LotShape" : {"IR2": 3, "IR3": 3, "IR1": 2, "Reg": 1},
 "LandContour" : {"HLS": 4,"Low": 3,"Lvl": 2,"Bnk": 1},
 "Electrical" : {"Mix" : 1, "FuseP" : 1, "FuseF" : 1, "FuseA" :2, "SBrkr" : 3},
 "LotConfig" : {"CulDSac" : 3, "FR3" : 3 , "Corner" : 2 ,"FR2" : 1,"Inside" : 1} ,
 "LandSlope" : {"Sev" : 2, "Mod" : 2,"Gtl" : 1 },
 "BldgType" : {"1Fam" : 2, "TwnhsE" : 2, "Twnhs" : 1, "Duplex": 1, "2fmCon": 1},
 "RoofStyle" : {"Shed" : 4, "Hip" : 4, "Flat" : 3,"Mansard" : 2, "Gable" : 2,"Gambrel" : 1 },
 "RoofMatl" : {"WdShngl" : 4, "Membran" : 3,"WdShake" : 3, "Tar&Grv" : 2 , "Metal" : 2, "CompShg": 2,"ClyTile" : 1,"Roll" : 1},
 "MasVnrType" : {"Stone" : 3, "BrkFace": 2, "None" : 1, "BrkCmn" : 1},
 "ExterQual" :  {"Ex" : 4, "Gd" : 3, "TA" : 2, "Fa" : 1},
 "ExterCond" : {"Ex" : 5, "TA" : 4, "Gd" : 3, "Fa" : 2, "Po" : 1},
 "Foundation" : {"PConc" : 5, "Wood" : 4, "Stone" : 4,"CBlock" : 3, "BrkTil" : 2, "Slab" : 1}, 
 "BsmtQual" : {'None': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
 "BsmtCond" : {'Po': 1, 'None': 2, 'Fa': 3, 'TA': 4, 'Gd': 5},
 "BsmtExposure" : {"Gd": 5 , "Av" : 4, "Mn" : 3, "No" : 2, "None" : 1},
 "BsmtFinType1" : {'None': 1, 'Rec': 2,'BLQ': 2 ,'LwQ': 2,'ALQ': 3,'Unf': 4,'GLQ': 5 },
 "BsmtFinType2" : {"None" : 1, "BLQ" : 2,"LwQ" : 2,"Rec" : 2, "GLQ" : 3,"Unf" : 3,"ALQ" : 4},
 "Heating" : {"GasA" : 5, "GasW" : 4, "OthW" : 3, "Wall" : 2, "Grav" : 1,"Floor" : 1},
 "HeatingQC" : {"Ex" : 4, "Gd" : 3, "TA" : 3,"Fa" : 2, "Po" : 1},
 "CentralAir" : {"Y" : 1, "N" : 0},
 "KitchenQual" : {"Ex" : 4, "Gd" : 3 , "TA" : 2,"Fa" :1},
 "FireplaceQu" : {"Ex" : 6, "Gd" : 5,"TA" : 4, "Fa" : 3,"None" : 2,"Po":1},
 "GarageType" : {"BuiltIn" : 5, "Attchd" : 4, "Basment" : 3, "2Types" : 3, "Detchd" : 2 ,"CarPort" : 1,"None" : 1},
 "GarageFinish" : {"Fin" : 4,"RFn" : 3,"Unf" : 2,"None" : 1},
 "GarageQual" : {"Ex" : 5,"Gd" : 4,"TA" : 3,"Fa" : 2,"None" : 1,"Po" :1 },
 "GarageCond" : {"TA" : 5,"Gd" : 4,"Ex" : 3,"Fa" : 2,"Po" : 1,"None" : 1 },
 "PavedDrive" : {"Y" : 3 , "P" : 2, "N" : 1},
 "Fence" : {"None" : 2, "GdPrv" : 2,"MnPrv" : 1,"GdWo" : 1,"MnWw" : 1},
 "Condition1" : {"PosA":7, "PosN":7, "RRNn":6, "RRNe":5, "Norm":4 ,"RRAn":3 ,"Feedr":2 ,"RRAe":1,"Artery":1},
 "Condition2" : {"PosA":7, "PosN":6, "RRAe":5, "Norm":4, "RRAn":3 ,"Feedr":3 ,"Artery":2 ,"RRNn":1},
 
 "Neighborhood" : { 'MeadowV': 1,'IDOTRR': 1,'BrDale': 1,
                    'BrkSide': 2,'OldTown': 2,'Edwards': 2,
                    'Sawyer': 3,'Blueste': 3,'SWISU': 3,'NPkVill': 3,'NAmes': 3,
                    'Mitchel': 4,
                    'SawyerW': 5,'NWAmes': 5,'Gilbert': 5,'Blmngtn': 5,'CollgCr': 5,
                    'ClearCr': 6,'Crawfor': 6,
                    'Veenker': 7,'Somerst': 7,
                    'Timber': 8,
                    'StoneBr': 9,
                    'NridgHt': 10,'NoRidge': 10}
} 


# In[30]:


orders = ["MSZoning",  "LotShape", "LandContour", "LotConfig", "LandSlope","Electrical" ,
         "BldgType", "RoofStyle", "RoofMatl", "MasVnrType", "ExterQual", "ExterCond", "Foundation", 
         "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "Heating", 
         "HeatingQC","CentralAir", "KitchenQual", "FireplaceQu", "Condition1","Condition2",
         "GarageQual", "GarageCond", "PavedDrive", "Fence" , "Neighborhood" ,"GarageType", "GarageFinish"]


# In[31]:


for col in orders:
    df[col] = df[col].map(dictionary[col]).astype(int)


# In[32]:


# Having Feature

df["HasPool"] = df["PoolArea"].apply(lambda x : 1 if x > 0 else 0)
df["HasGarage"] = df["GarageCars"].apply(lambda x : 1 if x > 0 else 0)
df["HasFirePlace"] = df["Fireplaces"].apply(lambda x : 1 if x > 0 else 0)
df["Has2ndFloor"] = df["2ndFlrSF"].apply(lambda x : 1 if x > 0 else 0)
df["HasBsmt"] = df["TotalBsmtSF"].apply(lambda x : 1 if x > 0 else 0 )


#Another Feature

df["GarageCarSize"] = df["GarageArea"] / df["GarageCars"]

df["HomeRepairYear"] = df["YearRemodAdd"] - df["YearBuilt"]

df["TotalSF"] = df["BsmtFinSF1"] + df["BsmtFinSF2"] + df["1stFlrSF"] + df["2ndFlrSF"]

df["SqFtPerRoom"] = df["GrLivArea"] / (df["TotRmsAbvGrd"]+
                                       df["FullBath"] +
                                       df["HalfBath"] + 
                                       df["KitchenAbvGr"])

df["TotalPorchSF"] = (df["OpenPorchSF"] + df["3SsnPorch"] + 
                      df["EnclosedPorch"] + df["ScreenPorch"] + df["WoodDeckSF"])

df["TotalBathrooms"] = (df["FullBath"] + (0.5*df["HalfBath"])) + df["BsmtFullBath"] + (0.5*df["BsmtHalfBath"])

# Score

df["ConditionScore"] = df["Condition1"] + df["Condition2"]

df["GarageScore"] = df["GarageCond"] + df["GarageQual"]

df["TotalHomeScore"] = df["OverallCond"] + df["OverallQual"]

df["ExterScore"] = df["ExterCond"] + df["ExterQual"]

df["TotalBsmScore"] = df["BsmtCond"] + df["BsmtQual"] + df["BsmtFinType1"] + df["BsmtFinType2"]

df['TotalQual'] = df['OverallQual'] + df['ExterCond'] +                   df['TotalBsmScore'] + df['GarageScore'] +                   df['ConditionScore'] + df["KitchenQual"] + df['HeatingQC']
        
###

df["QualGr"] = df["TotalQual"] * df["GrLivArea"]


# In[33]:


na_values(df)


# In[34]:


df["GarageCarSize"] = df["GarageCarSize"].fillna(0)
na_values(df)


# In[35]:


df_new = df.copy()
df = df_new.copy()


# In[36]:


new_features = ["TotalBsmScore","ExterScore","TotalHomeScore","GarageScore","ConditionScore",
                "TotalBathrooms","TotalPorchSF","SqFtPerRoom","TotalSF","HomeRepairYear","GarageCarSize",
                "HasPool","HasGarage","HasFirePlace","Has2ndFloor","HasBsmt","TotalQual","QualGr"]

print(len(new_features))
dtypes_nunique(df[new_features])


# In[37]:


print("Old Num Cols : \n\n", num_cols,end =  "\n\n")
print("Old Cat Cols : \n\n ", cat_cols,end =  "\n\n")
print("Old Cat But Car : \n\n ", cat_but_car,end =  "\n\n")


# In[38]:


cat_cols,num_cols,cat_but_car = grab_col_names(df,details = True)


# In[39]:


print("Num Cols : \n\n", num_cols,end =  "\n\n")
print("Cat Cols : \n\n ", cat_cols,end =  "\n\n")
print("Cat But Car : \n\n ", cat_but_car,end =  "\n\n")


# In[40]:


dtypes_nunique(df[num_cols])


# In[41]:


dtypes_nunique(df[cat_cols])


# In[42]:


df = rare_encoder(df,0.01,cat_cols)


# In[43]:


for col in cat_cols:
    if df[col].dtype == "object":
        label_encoder(df,col)


# In[44]:


dtypes_nunique(df[cat_cols])


# In[45]:


(df.dtypes == "object").sum()


# In[46]:


useless_columns = [col for col in cat_cols if df[col].nunique() == 1 or 
                   (df[col].nunique() == 2 and (df[col].value_counts() / len(df) <= 0.01).any(axis = None))]


# In[47]:


useless_columns


# In[48]:


for col in useless_columns:
    df.drop(col, axis = 1 ,inplace = True)


# In[49]:


cat_cols = [col for col in cat_cols if col not in useless_columns]
len(cat_cols)


# In[50]:


##################
# One-Hot Encoding
##################


# In[51]:


def select_one_columns(dtypes_nunique_):
    
    dtypes_nunique_ = dtypes_nunique_.T.Nunique
    dtypes_nunique_ = dtypes_nunique_[dtypes_nunique_ <= 5].index.tolist()
    return dtypes_nunique_

one_columns = select_one_columns(dtypes_nunique(df))
len(one_columns)


# In[52]:


df = one_hot_encoder(dataframe=df, categorical_cols=one_columns,drop_first=True)


# In[53]:


df.head()


# In[54]:


cat_cols,num_cols,cat_but_car = grab_col_names(df,details = True)


# In[55]:


print("Num Cols : \n\n", num_cols,end =  "\n\n")
print("Cat Cols : \n\n ", cat_cols,end =  "\n\n")
print("Cat But Car : \n\n ", cat_but_car,end =  "\n\n")


# In[56]:


dtypes_nunique(df[num_cols])


# In[57]:


useless_cols_new = [col for col in cat_cols if (df[col].value_counts() / len(df) <= 0.01).any(axis = None) ]
print(useless_cols_new)


# In[58]:


#for col in useless_cols_new:
#    df.drop(col, axis = 1 , inplace = True)


# In[59]:


##################
# Outliers
##################


# In[60]:


na_values(df)


# In[61]:


edited_num_cols = [col for col in num_cols if col not in "Id"]
for col in edited_num_cols:
    print(f" {col.upper()} : {check_outliers(df, col)}")


# In[62]:


scatterplot_num(df, edited_num_cols, "SalePrice")


# In[63]:


exclude = ['BsmtFinSF2',
           'LowQualFinSF',
           'EnclosedPorch',
           '3SsnPorch',
           'ScreenPorch',
           'PoolArea',
           'PoolQC',
           'MiscVal',
           'ConditionScore',
           'SalePrice']


# In[64]:


for col in edited_num_cols:
    if col not in exclude:
        replace_with_thresholds(df, col)


# In[65]:


(dtypes_nunique(df[edited_num_cols]).T.Nunique == 1).sum()


# In[66]:


scatterplot_num(df, edited_num_cols, "SalePrice")


# In[67]:


for col in edited_num_cols:
    print(f" {col.upper()} : {check_outliers(df, col)}")


# In[68]:


len(num_cols),len(cat_cols),len(cat_but_car)


# In[69]:


new_label = ['Neighborhood','OverallQual','TotRmsAbvGrd','TotalBathrooms','TotalHomeScore']

dtypes_nunique(df[new_label])


# In[70]:


for col in new_label:
    label_encoder(df, col)


# In[71]:


num_cols = [col for col in num_cols if col not in new_label]
cat_cols = cat_cols + new_label
len(num_cols),len(cat_cols),len(cat_but_car)


# In[72]:


dtypes_nunique(df[num_cols])


# In[73]:


(dtypes_nunique(df[num_cols]).T.Nunique < 10).sum()


# In[74]:


(dtypes_nunique(df[cat_cols]).T.Nunique >= 10).sum()


# In[75]:


pd.to_pickle(df,open("House_Price.pkl","wb"))


# In[76]:


######################################
# Standart Scaler
######################################


# In[77]:


scaler_num_cols = [col for col in num_cols if col not in ["Id","SalePrice"]]
len(scaler_num_cols)


# In[78]:


scaler_num_cols


# In[79]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df[scaler_num_cols] = scaler.fit_transform(df[scaler_num_cols])


# In[80]:


df[scaler_num_cols].head()


# In[81]:


print("Final DataFrame Shape : {}".format(df.shape))


# In[82]:


pd.to_pickle(df , open("Scaler_House_Price.pkl","wb"))


# In[ ]:





# In[ ]:





# In[ ]:





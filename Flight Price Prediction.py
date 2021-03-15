#!/usr/bin/env python
# coding: utf-8

# ### Flight Price Prediction

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')


# In[2]:


os.chdir('E:\\prasad\\SMIIT\\Working Project\\Flight Price Prediction')


# In[3]:


pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)


# In[4]:


df=pd.read_excel('Data_Train.xlsx')


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.head(3)


# #### Data Cleaning of Numeric Value

# In[8]:


df['Journey_day']=pd.to_datetime(df['Date_of_Journey'],format='%d/%m/%Y').dt.day


# In[9]:


df.head(2)


# In[10]:


df['Journey_month']=pd.to_datetime(df['Date_of_Journey'],format='%d/%m/%Y').dt.month


# In[11]:


df.drop('Date_of_Journey',axis=1,inplace=True)


# In[12]:


df.head(2)


# In[13]:


df.isnull().sum()


# In[14]:


df.dropna(inplace=True)


# In[15]:


df['Dep_hour']=pd.to_datetime(df.Dep_Time).dt.hour


# In[16]:


df['Dep_min']=pd.to_datetime(df.Dep_Time).dt.minute


# In[17]:


df.drop('Dep_Time',axis=1,inplace=True)


# In[18]:


df.head(2)


# In[19]:


df['Arrival_hour']=pd.to_datetime(df.Arrival_Time).dt.hour


# In[20]:


df['Arrival_min']=pd.to_datetime(df.Arrival_Time).dt.minute


# In[21]:


df.drop('Arrival_Time',axis=1,inplace=True)


# In[22]:


df.head(2)


# In[23]:


# Time taken by plane to reach destination is called Duration
# It is the differnce betwwen Departure Time and Arrival time


# Assigning and converting Duration column into list
duration = list(df["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:    # Check if duration contains only hour or mins
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute
        else:
            duration[i] = "0h " + duration[i]           # Adds 0 hour

duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extracts only minutes from duration


# In[24]:


df['Duration_hours']=duration_hours
df['Duration_mins']=duration_mins


# In[25]:


df.head(3)


# In[26]:


df.drop('Duration',axis=1,inplace=True)


# In[27]:


df.head(2)


# ### Handling Categorical Data

# In[28]:


df['Airline'].value_counts()


# In[29]:


sns.catplot(x='Airline',y='Price',data=df.sort_values('Price',ascending = False),kind="boxen", height = 6,aspect=3)
plt.show()


# In[30]:


df.head(2)


# In[31]:


Airline=df[['Airline']]


# In[32]:


Airline


# In[33]:


Airline=pd.get_dummies(Airline,drop_first=True)
Airline.head()


# In[34]:


df['Source'].value_counts()


# In[35]:


sns.catplot(x='Source',y='Price',data=df.sort_values('Price',ascending=False),kind='boxen',height=6,aspect=3)
plt.show()


# In[36]:


Source=df[['Source']]


# In[37]:


Source


# In[38]:


Source=pd.get_dummies(Source,drop_first=True)
Source.head()


# In[39]:


df.head(2)


# In[40]:


df['Destination'].value_counts()


# In[41]:


Destination=df[['Destination']]


# In[42]:


Destination


# In[43]:


Destination=pd.get_dummies(Destination,drop_first=True)
Destination.head()


# In[44]:


df.head(2)


# In[45]:


df.drop(['Route','Additional_Info'],axis=1,inplace=True)


# In[46]:


df.head(1)


# In[47]:


df['Total_Stops'].value_counts()


# In[48]:


df['Total_Stops'].unique()


# In[49]:


stops_map={'non-stop':0,'1 stop':1,'2 stops':2,'3 stops':3,'4 stops':4}


# In[50]:


df['Total_Stops']=df['Total_Stops'].map(stops_map)


# In[51]:


df['Total_Stops'].unique()


# In[52]:


df.head()


# In[53]:


train_data=pd.concat([df,Airline,Source,Destination],axis=1)


# In[54]:


train_data.head()


# In[55]:


train_data.shape


# In[56]:


train_data.drop(['Airline','Source','Destination'],axis=1,inplace=True)


# In[57]:


train_data.shape


# In[58]:


train_data.head(1)


# In[59]:


train_data.columns


# In[60]:


X=train_data.loc[:,['Total_Stops','Journey_day', 'Journey_month', 'Dep_hour',
       'Dep_min', 'Arrival_hour', 'Arrival_min', 'Duration_hours',
       'Duration_mins', 'Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo',
       'Airline_Jet Airways', 'Airline_Jet Airways Business',
       'Airline_Multiple carriers',
       'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',
       'Airline_Trujet', 'Airline_Vistara', 'Airline_Vistara Premium economy',
       'Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',
       'Destination_Cochin', 'Destination_Delhi', 'Destination_Hyderabad',
       'Destination_Kolkata', 'Destination_New Delhi']]


# In[61]:


X.shape


# In[62]:


X.head()


# In[63]:


y=train_data.iloc[:,1]


# In[64]:


# Finds correlation between Independent and dependent attributes

plt.figure(figsize = (18,18))
sns.heatmap(df.corr(), annot = True, cmap = "RdYlGn")

plt.show()


# In[65]:


# Important feature using ExtraTreesRegressor

from sklearn.ensemble import ExtraTreesRegressor
selection = ExtraTreesRegressor()
selection.fit(X, y)


# In[66]:


print(selection.feature_importances_)


# In[67]:


#plot graph of feature importances for better visualization

plt.figure(figsize = (12,8))
feat_importances = pd.Series(selection.feature_importances_, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()


# In[68]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[69]:


from sklearn.ensemble import RandomForestRegressor
reg_rf = RandomForestRegressor()
reg_rf.fit(X_train, y_train)


# In[70]:


y_pred = reg_rf.predict(X_test)


# In[71]:


reg_rf.score(X_train, y_train)


# In[72]:


reg_rf.score(X_test, y_test)


# In[73]:


sns.distplot(y_test-y_pred)
plt.show()


# In[74]:


plt.scatter(y_test, y_pred, alpha = 0.5)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()


# ## Hyperparameter Tuning

# In[75]:


from sklearn.model_selection import RandomizedSearchCV


# In[76]:


#Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]


# In[77]:


# Create the random grid

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}


# In[78]:


# Random search of parameters, using 5 fold cross validation, 
# search across 100 different combinations
rf_random = RandomizedSearchCV(estimator = reg_rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)


# In[79]:


rf_random.fit(X_train,y_train)


# In[80]:


rf_random.best_params_


# In[81]:


prediction = rf_random.predict(X_test)


# In[82]:


plt.figure(figsize = (8,8))
sns.distplot(y_test-prediction)
plt.show()


# In[83]:


plt.figure(figsize = (8,8))
plt.scatter(y_test, prediction, alpha = 0.5)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()


# ## Save the model to reuse it again

# In[84]:


import pickle,joblib


# In[85]:


pickle.dump(reg_rf,open('flight.pkl','wb'))


# In[86]:


joblib.dump(reg_rf,'flight.jbl')


# ### Load Pickle Model

# In[87]:


model_pkl=pickle.load(open('flight.pkl','rb'))


# In[88]:


model_pkl.score(X_test,y_test)


# In[89]:


model_pkl.score(X_train,y_train)


# In[90]:


from sklearn.model_selection import cross_val_score


# In[91]:


cv=cross_val_score(RandomForestRegressor(),X_train,y_train)
print('n_split',cv)
print('Average',np.average(cv))


# ### Load Joblib Model

# In[92]:


model_jbl=joblib.load('flight.jbl')


# In[93]:


model_jbl.score(X_train,y_train)


# In[94]:


model_jbl.score(X_test,y_test)


# In[95]:


cv=cross_val_score(RandomForestRegressor(),X_train,y_train)
print('n_split',cv)
print('Average',np.average(cv))


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency


# In[22]:


data = pd.read_csv("C:\\Users\\Uk\\Downloads\\Churn\\E Commerce Dataset.csv")


# In[3]:


data.shape


# In[24]:


data.head()


# In[25]:


data.pop("CustomerID")

data


# ### Understanding columns

# <table>
# 	<tr>
# 		<td><b>Variable</b></td>
# 		<td><b>Description</b></td>
# 	</tr>
# 	<tr>
# 		<td>CustomerID</td>
# 		<td>Unique customer ID</td>
# 	</tr>
# 	<tr>
# 		<td>Churn</td>
# 		<td>Churn Flag</td>
# 	</tr>
# 	<tr>
# 		<td>Tenure</td>
# 		<td>Tenure of customer in organization</td>
# 	</tr>
# 	<tr>
# 		<td>PreferredLoginDevice</td>
# 		<td>Preferred login device of customer</td>
# 	</tr>
# 	<tr>
# 		<td>CityTier</td>
# 		<td>City tier</td>
# 	</tr>
# 	<tr>
# 		<td>WarehouseToHome</td>
# 		<td>Distance in between warehouse to home of customer</td>
# 	</tr>
# 	<tr>
# 		<td>PreferredPaymentMode</td>
# 		<td>Preferred payment method of customer</td>
# 	</tr>
# 	<tr>
# 		<td>Gender</td>
# 		<td>Gender of customer</td>
# 	</tr>
# 	<tr>
# 		<td>HourSpendOnApp</td>
# 		<td>Number of hours spend on mobile application or website</td>
# 	</tr>
# 	<tr>
# 		<td>NumberOfDeviceRegistered</td>
# 		<td>Total number of deceives is registered on particular customer</td>
# 	</tr>
# 	<tr>
# 		<td>PreferedOrderCat</td>
# 		<td>Preferred order category of customer in last month</td>
# 	</tr>
# 	<tr>
# 		<td>SatisfactionScore</td>
# 		<td>Satisfactory score of customer on service</td>
# 	</tr>
# 	<tr>
# 		<td>MaritalStatus</td>
# 		<td>Marital status of customer</td>
# 	</tr>
# 	<tr>
# 		<td>NumberOfAddress</td>
# 		<td>Total number of added added on particular customer</td>
# 	<tr>
# 		<td>Complain</td>
# 		<td>Any complaint has been raised in last month</td>
# 	</tr>
# 	<tr>
# 		<td>OrderAmountHikeFromlastYear</td>
# 		<td>Percentage increases in order from last year</td>
# 	</tr>
# 	<tr>
# 		<td>CouponUsed</td>
# 		<td>Total number of coupon has been used in last month</td>
# 	</tr>
# 	<tr>
# 		<td>OrderCount</td>
# 		<td>Total number of orders has been places in last month</td>
# 	</tr>
# 	<tr>
# 		<td>DaySinceLastOrder</td>
# 		<td>Day Since last order by customer</td>
# 	</tr>
# 	<tr>
# 		<td>CashbackAmount</td>
# 		<td>Average cashback in last month</td>
# 	</tr>
# </table>

# ### Data Types

# In[26]:


data.dtypes


# ### Handle missing values

# In[7]:


data.isnull().sum()


# In[27]:


for i in data.columns:
    if data[i].isnull().sum() > 0:
        data[i].fillna(data[i].mean(),inplace=True)


# In[28]:


data["Churn"].unique()


# ### For Categorical Values

# In[29]:


for col in data.columns:
    if data[col].dtypes == 'object':
        print(col)
        value_counts = data[col].value_counts()
        plt.figure(figsize=(10, 5))
        sns.barplot(x = value_counts.index, y = value_counts)
        
        plt.show()


# ### For Continuous Variables

# In[30]:


for col in data.columns:
    if data[col].dtypes != 'object' and col != "Churn":
        print(col)
        sns.boxplot(data=data[col])
        plt.figure(figsize=(10, 5))        
        plt.show()


# In[11]:


for col in data.columns:
    if data[col].dtypes != 'object' and col != "Churn":
        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)
        iqr=q3-q1
        l_w = q1 - (1.5 * iqr)
        u_w = q3 + (1.5 * iqr)
        data = data[data[col]< u_w]
        data = data[data[col]> l_w]
        # print(str(q1) + " " + str(q3) + " " + str(l_w) + " " + str(u_w))


# ### Standard Scaler

# In[31]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

tmp_data = data.copy()
data_corr = data[['Tenure','CityTier','WarehouseToHome','HourSpendOnApp','NumberOfDeviceRegistered','SatisfactionScore','NumberOfAddress','Complain','OrderAmountHikeFromlastYear','CouponUsed','OrderCount','DaySinceLastOrder','CashbackAmount']]
data_corr = sc.fit_transform(data_corr)
data = pd.DataFrame(data_corr)
data.columns = ['Tenure','CityTier','WarehouseToHome','HourSpendOnApp','NumberOfDeviceRegistered','SatisfactionScore','NumberOfAddress','Complain','OrderAmountHikeFromlastYear','CouponUsed','OrderCount','DaySinceLastOrder','CashbackAmount']
data['PreferredLoginDevice'] = tmp_data['PreferredLoginDevice']
data['PreferredPaymentMode'] = tmp_data['PreferredPaymentMode']
data['Gender'] = tmp_data['Gender']
data['PreferedOrderCat'] = tmp_data['PreferedOrderCat']
data['MaritalStatus'] = tmp_data['MaritalStatus']
data['Churn'] = tmp_data['Churn']


# In[33]:


data["Churn"].unique()


# ### Correlation

# In[34]:


data_corr = data[['Tenure','CityTier','WarehouseToHome','HourSpendOnApp','NumberOfDeviceRegistered','SatisfactionScore','NumberOfAddress','Complain','OrderAmountHikeFromlastYear','CouponUsed','OrderCount','DaySinceLastOrder','CashbackAmount']]


# In[35]:


plt.figure(figsize=(15,15))
sns.heatmap(data_corr.corr(),annot=True)


# ### Feature Engineering

# #### Chi Squared Test

# In[36]:


def doChiSquaredTest(column_name, target = 'Churn'):
    contigency_pct = pd.crosstab(data[column_name], data[target], normalize='index')
    # plt.figure(figsize=(12,8)) 
    # sns.heatmap(contigency_pct, annot=True, cmap="YlGnBu")
    # print(contigency_pct)
    c, p, dof, expected = chi2_contingency(contigency_pct) 
    #print(str(c) + " " + str(p) + " " + str(dof) + " " + str(expected))
    
    if p <= 0.95: 
        print(col + " - significant result, reject null hypothesis (H0), dependent.")
    if p > 0.95:
        print(col + " - not significant result, fail to reject null hypothesis (H0), independent.")


# In[37]:


for col in data.columns:
    if data[col].dtypes == 'object' and col != "Churn":
        doChiSquaredTest(col)


# Except <b>Gender</b> all other columns are independent. Hence we remove them from our features.

# In[38]:


data = data.drop(columns = ['PreferredLoginDevice', 'PreferredPaymentMode', 'PreferedOrderCat', 'MaritalStatus'], axis=1)


# In[39]:


tmp_data = data.copy()


# ### Column Labeling
# 
# Our data now contains only one column <b>Gender</b> that is categorical. So here we encode them into One Hot Encoding

# In[48]:


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()


# In[43]:


gender_data = data["Gender"]


# In[49]:


gender_encoded = label_encoder.fit_transform(gender_data)


# In[50]:


data["GenderEncoded"] = gender_encoded


# ### Train and Test Data

# In[56]:


data.columns

X = data[['Tenure', 'CityTier', 'WarehouseToHome', 'HourSpendOnApp','NumberOfDeviceRegistered', 'SatisfactionScore', 'NumberOfAddress',
       'Complain', 'OrderAmountHikeFromlastYear', 'CouponUsed', 'OrderCount','DaySinceLastOrder', 'CashbackAmount',
       'GenderEncoded']]

y = data[["Churn"]]


# In[60]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# In[63]:


y_train.shape


# ### K Means Clustering

# In[ ]:





# In[ ]:





# ### Random Forest Classifier

# In[64]:


#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)


# In[65]:


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[71]:


confusion_map = metrics.confusion_matrix(y_test, y_pred)


# In[74]:


sns.heatmap(confusion_map, square=True, annot=True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# # Rhode Island Police Activity Analysis part 1
# 
# 

# In[2]:


# Let's start with simple commands such as: 


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')

# Now, let's take a closer look on our dataset: 

ri = pd.read_csv('Desktop/rhode-island-2019-02.csv', low_memory=False)
ri.head()


# ## NaN values 

# In[12]:


# let's check if there are null values in our dataset
ri.isnull().head()


# ## Dropping NaN values

# In[13]:


# Now we take a look at what columns to drop 
# (or in case there are no columns to drop, it will help us to better understand our dataset, therefore this step is
# helpful)
ri.isnull().sum()


# In[14]:


# in order to know which columns to drop, we need to print the shape of our dataset: 

ri.shape


# In[ ]:


# in this case there is no column we can drop, because none of the columns is completely empty, but for the 
# purpose of this analysis we will drop three columns: 'contraband_alcohol', 'contraband_weapons', 'search_basis'


# ## Dropping empty columns and rows

# In[15]:


ri.drop('contraband_alcohol', axis='columns', inplace=True)


# In[16]:


ri.drop('contraband_weapons', axis='columns', inplace=True) 


# In[17]:


ri.drop('search_basis', axis='columns', inplace=True)


# In[86]:


# Let's make sure the columns were actually dropped by printing head of the dataset:

ri.head()


# In[19]:


# now we drop NaN rows from columns 'reason_for_search' and 'vehicle_model' to make the analysis easier for us 

ri.dropna(subset=['reason_for_search', 'vehicle_model'], inplace=True)


# In[20]:


ri.isnull().head()


# ## Fixing dtypes and correcting formats
# 

# In[21]:


# for smoother manipulation with our data, we need to change current data types of some columns in our dataset

ri.dtypes     # in this case, we start with printing the data types first


# In[22]:


# To get answers and draw conclusions from our data we choose to change the data types of 'arrest_made' and 
#'citation_issued' from 'object' to 'bool'(boolean)

ri['arrest_made'] = ri.arrest_made.astype('bool')


# In[23]:


ri['citation_issued'] = ri.citation_issued.astype('bool')


# In[24]:


#Combine date and time column in "combined" with space in between 


combined = ri.date.str.cat(ri.time, sep=' ')


# In[25]:


combined.head()


# In[26]:


#create a new column called date_and_time from combined, use pandas method to_datetime

ri['date_and_time'] = pd.to_datetime(combined)


# In[27]:


ri.head()


# In[28]:


# let's make sure the column we've created has also the desired data type 

ri.dtypes  


# ## Setting index

# In[29]:


#let's set the column date_and_time as an index column

ri.set_index('date_and_time', inplace=True)


# In[30]:


ri.head()  


# # 1. Most common outcome of a police stop 

# In[18]:


outcomes=ri.outcome.value_counts()
print(outcomes)

# citation or also known as a ticket, is the most common outcome of a police stop


# # 2. Most common traffic violation

# In[46]:


ri.reason_for_stop.value_counts()         # from 'Call for Service' named as 'other'


# In[47]:


# let's create a visual pie chart to see what are the most common traffic violations

labels = ['Speeding', 'Other Traffic Violation', 'Equipment/Inspection', 'Registration',
'Seatbelt', 'Special Detail/Directed Patrol', 'other']
sizes = [268744,90234,61252,19830,16327,13642,10576]
plt.axis('equal')
plt.pie(sizes, labels=labels, radius=2, autopct='%1.1f%%', shadow=True, startangle=220, explode=[0,0,0,0,0,0,0.5])
("")

# 'other' contains = Call for Service, Violation of City/Town Ordinance, Motorist Assist/Courtesy, APB, 
#  Suspicious Person and Warrant


# ## 3.Comparing violations by gender 

# In[61]:


# let's take a closer look at the difference between violations committed by men and violations committed by women, 
# is there a difference? 


female = ri[ri.subject_sex == 'female']
male = ri[ri.subject_sex == 'male']

resultfemale= female.reason_for_stop.value_counts(normalize=True)
print(resultfemale)

resultmale=male.reason_for_stop.value_counts(normalize=True)
print(resultmale)


# In[ ]:


# as we can see women are more likely to speed than men, on the other hand they are less likely to commit the
# other two most common violations 'Other Traffic Violation' and 'Equipment/Inspection Violation'


# ## 4. Does gender affect whether your car will be searched?

# In[83]:


female=ri[ri.subject_sex == 'female'].search_conducted.mean()     
print(female)                                                  
male=ri[ri.subject_sex == 'male'].search_conducted.mean()
print(male)                                                 
                                                                
                                                # Women seem to be less likely to have their car searched, but to
                                                # make sure our conclusions are correct, we have to find out the total
                                                # number of searches

femaletotal=ri[ri.subject_sex == 'female'].search_conducted.sum()
maletotal=ri[ri.subject_sex == 'male'].search_conducted.sum()

print(femaletotal)
print(maletotal)


# In[ ]:


# ^ We see clearly, that drawing conclusion based on one result can significantly distort overall understanding of out
# data, specifically in this case there were around 6.2 times more men having their car searched than women and 
# therefore the mean of women having their car searched was lower because the total number of women having 
# their car searched was 6.2 times less.


# In[16]:


# if we take a closer look on the connection between reason for stop, the search itself and gender we see that the
# most common reason why both genders had their car searched was "Speeding" and "Other Traffic Violation" and 
# eventually "Equipment/Inspection Violation"

#given the proportions there's no significant difference except, women were more likely to be stopped for 
# "Other Traffic Violation" than men

ri.groupby(['subject_sex', 'reason_for_stop']).search_conducted.sum()


# ## 5. Arrest rate

# In[79]:


ri.arrest_made.value_counts(normalize=True) 


#We can see the arrest rate is around 3,4 % 


# ## 6. Arrest rate by district 

# In[87]:


ri.zone.unique()

# Let's find out the names of the districts with the .unique() method, we need to know the names of the zones first
# to move on with our analysis


# In[11]:


# in order to find out which district has the highest arrest rate and which one the lowest we will filter the 
# dataset by each zone 

x3=ri[ri.zone == 'X3'].arrest_made.mean()
print(x3)

x4=ri[ri.zone == 'X4'].arrest_made.mean()
print(x4)

k3=ri[ri.zone == 'K3'].arrest_made.mean()
print(k3)

k2=ri[ri.zone == 'K2'].arrest_made.mean()
print(k2)

k1=ri[ri.zone == 'K1'].arrest_made.mean()
print(k1)

x1=ri[ri.zone == 'X1'].arrest_made.mean()
print(x1)


# as we can see, the highest arrest rate is in the zone X4, on the contrary the lowest arrest rate is in zone K1



all=ri.groupby('zone').arrest_made.sum()     # let's take into consideration the total sum of arrests made in each
                                             # zone
print(all)


# In[10]:


# ^ After printing the sum of arrests made we can see our conclusion was not completely correct, the average or the
# mean number of arrests is the lowest in zone K1 but the amount of arrests is the lowest in zone X1. 

#
# 

# To find out more information about the relation between gender and arrest we use the groupby method once again, 
# this time we add a second factor 'subject_sex'

ri.groupby(['zone', 'subject_sex']).arrest_made.sum()  

 


# In[ ]:


# Arrests of men seem to be approximately 4-5 times higher than arrests of women in all zones.


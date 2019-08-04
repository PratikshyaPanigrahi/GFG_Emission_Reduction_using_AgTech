
# coding: utf-8

# In[ ]:


#The following data was collected from faostat.org for a vineyard in california
#Observing the data set, it is clear that the assumption was to apply 60lbs of fertilizer per acre
#In this, we find the usual requirement for grape vine and find how much excess fertilizer has been applied
#Then we correlate excess fertilization with Nitrogen emission due to N Fertilizers in CA


# In[69]:



import os
#datapath = os.path.join("datasets", "lifesat", "")
datapath = os.path.join("Desktop","handson-ml","handson-ml-master","")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model
from scipy.interpolate import *

# Load the data
Nitrogen_Data = pd.read_csv(datapath + "N_Temp.csv", thousands=',',na_values="n/a")
print(Nitrogen_Data)
N_Required = 10 #for grape, required N = 10 units per acre
N_per_lb = 0.30 # 0.30 units of N per lb of fertilizer
N_Content = np.c_[Nitrogen_Data['N_Content']]
a = Nitrogen_Data['N_Content']
N_Use = np.c_[Nitrogen_Data['N_Use']]
b = Nitrogen_Data['N_Use']
N_Emission = np.c_[Nitrogen_Data['N_Emission']]
c = Nitrogen_Data['N_Emission']
N_Excess = (N_Use*N_per_lb)+(N_Content-(N_Required))
x = b*N_per_lb+a-N_Required
print(x)

plt.title("Nitrogen Emission because of excess fertilizers")
plt.xlabel("Excess Fertilizer Applied(lb/H)")
plt.ylabel("Nitrogen Emission")
plt.plot(N_Excess, N_Emission,'o')
p2 = np.polyfit(x, c,1)
print(p2)
plt.plot(x,np.polyval(p2,x),'r-')
plt.show()



# In[ ]:


#We learn the trends of remnant Nitrogen post-harvest and pre-seeding
#We use linear regression to predict required amount of nitrogen fertilizer 
#We then give suggestions with respect to commercial fertilizer Nitrogen content


# In[70]:



x1 = Nitrogen_Data['Excess'][0:9]
y1 = Nitrogen_Data['N_Content'][1:]

#print(y1)
#print(x1)
plt.title("Soil Nitrogen Content Prediction")
plt.xlabel("Excess Fertilizer Applied(lb/H)")
plt.ylabel("Nitrogen Content in Soil")
plt.plot(x1, y1,'o')
p3 = np.polyfit(x1, y1,1)
print(p3)
plt.plot(x1,np.polyval(p3,x1),'r-')
plt.show()


model1 = sklearn.linear_model.LinearRegression()
model1.fit(np.array(x1).reshape(-1,1),np.array(y1).reshape(-1,1))

test1 =[[9.10]] #Excess Value
print(model1.predict(test1))
next_content= model1.predict(test1)
suggested_fertilizer= (N_Required-next_content)/N_per_lb
print("Apply ",suggested_fertilizer," lbs of Ammonium Nitrate based fertilizer per acre:")
#print(suggested_fertilizer)


# In[ ]:


#We then predict the emission reduction by using recommended Nitrogen fertilizer


# In[71]:


model = sklearn.linear_model.LinearRegression()
model.fit(N_Excess,N_Emission)
test_data = ((suggested_fertilizer*N_per_lb)-N_Required)
print(model.predict(test_data))


# In[67]:


#For a vineyard in CA, we suggest the amount of water to be irrigated
#We use the simplified version of Penman-Monteith Equation
#This is as per research by Dr. Larry Williams

#Water Use = (ETo X Kc) divided by the irrigation system efficiency


# In[74]:


Ref =  pd.read_csv(datapath + "Monthly_CA.csv", thousands=',',na_values="n/a") #reference evapotranspiration for grapes(KTo) 
print(Ref)


# In[80]:


Coeff = pd.read_csv(datapath + "Irrigation.csv", thousands=',',na_values="n/a") #crop coefficient for grapes (Kc) 
print(Coeff)
#Grape season is from July to October in CA
#Average Kc for July to October
Kc_Jul = np.mean(Coeff['Kc'][0:3])
Kc_Aug = np.mean(Coeff['Kc'][4:7])
Kc_Sep = np.mean(Coeff['Kc'][8:11])
Kc_Oct = np.mean(Coeff['Kc'][12:14])
print("Mean values per month: ")
print("July: ",Kc_Jul)
print("August: ",Kc_Aug)
print("September: ",Kc_Sep)
print("October: ",Kc_Oct)


# In[91]:


Ref =  pd.read_csv(datapath + "Monthly_CA.csv", thousands=',',na_values="n/a") #reference evapotranspiration for grapes(KTo) 
eff= 0.9
W_Jul = Ref["Eto"][6]*Kc_Jul*0.85/eff
W_Aug = Ref["Eto"][7]*Kc_Aug*0.85/eff
W_Sep = Ref["Eto"][8]*Kc_Sep*0.85/eff
W_Oct = Ref["Eto"][9]*Kc_Oct*0.8/eff
#print(W_Jul,W_Aug,W_Sep,W_Oct,"inches required to be replenished")
Water = np.array([W_Jul,W_Aug,W_Sep,W_Oct])
print(Water)


# In[ ]:


#Desired irrigation rate = 1.04521666667 2.14703703704 2.448 2.0336 inch
#1 acre-inch = 27,152 gallons
#1 acre = 43,560 square feet
#Vine spacing = 4 feet, Row spacing = 8 feet
#Area per vine = (4 feet X 8 feet) = 32 square feet


# In[92]:


Gal_per_acre = Water*27152
Water_Required = Gal_per_acre*0.000735
print(Water_Required," Gallons per vine required for July, Aug, Sep and Oct")


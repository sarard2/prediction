import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import sklearn
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor , ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV , train_test_split
import seaborn as sns
df=pd.read_csv("https://raw.githubusercontent.com/sarard2/prediction/main/price.csv")
# From graph we can see that Jet Airways Business have the highest Price.
# Apart from the first Airline almost all are having similar median
# Airline vs Price
sns.catplot(y = "Price", x = "Airline", data = df.sort_values("Price", ascending = False), kind="boxen", height = 6, aspect = 3)


# Source vs Price

sns.catplot(y = "Price", x = "SourceCity", data = df.sort_values("Price", ascending = False), kind="boxen", height = 4, aspect = 3)


# Finds correlation between Independent and dependent attributes

plt.figure(figsize = (18,18))
sns.heatmap(df.corr(), annot = True, cmap = "RdYlGn")
plt.show()

X = df.drop(["Price"] , axis =1)
y = df.Price
X_train , X_test , y_train , y_test = train_test_split(X,y,random_state = 100 , test_size = 0.3)


#Converting Categorical Features into numerical form using LabelEncoder()
le = LabelEncoder()
#For train dataset
X_train["ArrivalTime"]=le.fit_transform(X_train["ArrivalTime"])
print(X_train["ArrivalTime"])
X_train["DepartureTime"]=le.fit_transform(X_train["DepartureTime"])
X_train["SourceCity"] = le.fit_transform(X_train["SourceCity"])
X_train["DestinationCity"] = le.fit_transform(X_train["DestinationCity"])
X_train["Class"]=le.fit_transform(X_train["Class"])
print(X_train.head())

#For test dataset
X_test["ArrivalTime"]=le.fit_transform(X_test["ArrivalTime"])
print(X_test["ArrivalTime"])
X_test["DepartureTime"]=le.fit_transform(X_test["DepartureTime"])
X_test["SourceCity"] = le.fit_transform(X_test["SourceCity"])
X_test["DestinationCity"] = le.fit_transform(X_test["DestinationCity"])
X_test["Class"]=le.fit_transform(X_test["Class"])
print(X_test.head())

#mapping
stop = {
    "AirAsia":1,
    "GOFIRST":2,
    "AirIndia":3,
    "Indigo":4,
    "SpiceJet":5 , "Vistara":6
}

X_train.loc[: , "Airline"] = X_train["Airline"].map(stop)
X_test.loc[: , "Airline"] = X_test["Airline"].map(stop)

#mapping no. of stops for
stop = {
    "zero":0,
    "one":1,
    "twoormore":2
}

X_train.loc[: , "Stops"] = X_train["Stops"].map(stop)
print(X_train.head())
print(X_train.info())

X_test.loc[: , "Stops"] = X_test["Stops"].map(stop)

##create model
lr = LinearRegression()
rfr = RandomForestRegressor()
dt = DecisionTreeRegressor()
print(lr.fit(X_train,y_train))
print(r2_score(lr.predict(X_train) , y_train))
prediction=lr.predict(X_test)
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))


#Moving it into pickle file to be used in streamlit
import pickle
file = open(r"C:\Users\Sara\Desktop\logisticmodel.pkl", "wb")
pickle.dump(lr , file)
file.close()
model = open(r"C:\Users\Sara\Desktop\logisticmodel.pkl", "rb")
forest = pickle.load(model)

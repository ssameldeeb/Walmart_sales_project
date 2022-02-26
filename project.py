import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

data = pd.read_csv("Walmart.csv")

print(data.shape)
print(data.dtypes)
print(data.columns.values)
print(data.isnull().sum())

data["Date"] = pd.to_datetime(data["Date"])
print(data.dtypes)
print(data["Date"].head())

data["year"] = data["Date"].dt.year
data["day"] = data["Date"].dt.day
data["month"] = data["Date"].dt.month

data = data.drop("Date", axis=1)
print(data.head())
print(data.dtypes)

sns.heatmap(data.corr(), annot=True, cmap="hot")
plt.show()

sns.distplot(data["Fuel_Price"])
plt.show()

plt.figure(figsize=(9,5))
sns.countplot(data["day"])
plt.show()
print(data.groupby("day")["Fuel_Price"].sum().sort_values(ascending=False))

print("_"*100)
print(data.groupby("year")["Fuel_Price"].sum().sort_values(ascending=False))


x = data.drop("Fuel_Price", axis=1)
y = data["Fuel_Price"]
print(y[:5])

ss = StandardScaler()
x = ss.fit_transform(x)
print(x[:5])

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=44, shuffle =True)
print(X_train.shape)
print(y_train.shape)

Ln = LinearRegression()
Ln.fit(X_train, y_train)

print(Ln.score(X_train, y_train))
print(Ln.score(X_test, y_test))
print("_"*100)

# for x in range(2,20):
#     Dt = DecisionTreeRegressor(max_depth=x,random_state=33)
#     Dt.fit(X_train, y_train)

#     print("x = ", x)
#     print(Dt.score(X_train, y_train))
#     print(Dt.score(X_test, y_test))
#     print("_"*100)

Dt = DecisionTreeRegressor(max_depth=15,random_state=33)
Dt.fit(X_train, y_train)

print(Dt.score(X_train, y_train))
print(Dt.score(X_test, y_test))

y_pred = Dt.predict(X_test)
print(y_test[:5])
print(y_pred[:5])

ma = mean_absolute_error(y_test,y_pred)
print(ma)

plt.scatter(X_test[:,1], y_test, color="g")
plt.plot(X_test[:,1], y_pred, color="b")
plt.show()

autput = pd.DataFrame({"y_test":y_test, "y_pred":y_pred})
autput.to_csv("autput.csv",index=False)

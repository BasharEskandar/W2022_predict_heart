import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv('./heart.csv')
# df.info()
# print(df.duplicated().sum())

show_steps = True


# -----------------labels encoding--------------
def encode_label(column):
    from sklearn import preprocessing
    label_encoder = preprocessing.LabelEncoder()
    df[column] = label_encoder.fit_transform(df[column])
    df[column].unique()


encode_label("Sex")
encode_label("ChestPainType")
encode_label("RestingECG")
encode_label("ExerciseAngina")
encode_label("ST_Slope")
if show_steps:
    print(df.head())
#  --------------------feature scaling--------------
scaler = StandardScaler()
df1 = df.drop('HeartDisease', axis=1)
scaler.fit(df1)

scaled_features = scaler.transform(df1)
df_scaled = pd.DataFrame(scaled_features, columns=df1.columns[:])
if show_steps:
    print(df_scaled.head())
#  ----------------determining k -------------------
# split into train-test data subsets
X = df_scaled
y = df['HeartDisease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

error_rate_euclidian = []
for i in range(2, 20):
    knn = KNeighborsClassifier(n_neighbors=i, p=2)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate_euclidian.append(np.mean(pred_i != y_test))

error_rate_manhattan = []
for i in range(2, 20):
    knn = KNeighborsClassifier(n_neighbors=i, p=1)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate_manhattan.append(np.mean(pred_i != y_test))

# k = 9 with manhattan distance calculation gives the best accuracy
if show_steps:
    k_euc = np.argmin(error_rate_euclidian) + 2
    k_man = np.argmin(error_rate_manhattan) + 2
    print(k_euc)
    print(k_man)
    print(error_rate_euclidian[k_euc - 2])
    print(error_rate_manhattan[k_man - 2])

# plot error rates per k tried, for both euclidian and manhattan distance methods
if show_steps:
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, 20), error_rate_euclidian, color='blue', linestyle='--', marker='o', markerfacecolor='blue',
             markersize=10, label="euclidian")
    plt.plot(range(2, 20), error_rate_manhattan, color='red', linestyle='--', marker='o', markerfacecolor='red',
             markersize=10, label="manhattan")
    plt.legend()
    plt.title('Error Rate vs K')
    plt.xticks(ticks=range(0, 20))
    plt.xlabel('K')
    plt.ylabel('Error Rate')
    plt.show()
# ----------- model performance-----------------
knn = KNeighborsClassifier(n_neighbors=9, p=1)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)

print(confusion_matrix(y_test, pred))

if __name__ == '__main__':
    pass
# no null values, no duplicates
# gender dist. highly skewed towards Female,
# Heart Disease Distribution : mean : 0.553377  std: 0.497414   skewness: -0.215086

# missclassification rate: 0.15 0.85 accuracy
# true positive : 131/150 = 0.87
# true negative  : 104/126 = 0.82
# false positive: 22/126 = 0.17
# false negative:  19/150 = 0.13
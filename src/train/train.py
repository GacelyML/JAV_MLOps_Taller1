# imports
import pickle

from pandas import read_csv
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler

# load data
filename = "penguins_size.csv"
data = read_csv(filename)

# clean data
data.replace({"sex": {".": None}}, inplace=True)
data.dropna(inplace=True)

# transform data
target_variable = "species"
numerical_features = [
    "culmen_length_mm",
    "culmen_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]
categorical_features = ["island", "sex"]

label_encoder = LabelEncoder().fit(data[target_variable])
y = label_encoder.transform(data[target_variable])
X = data.drop(columns=[target_variable])

scaler = StandardScaler().fit(X[numerical_features])
X[numerical_features] = scaler.transform(X[numerical_features])

values_encoder = OrdinalEncoder().fit(X[categorical_features])
X[categorical_features] = values_encoder.transform(X[categorical_features])

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1337)

# build models
lr_model = LogisticRegression()
lda_model = LinearDiscriminantAnalysis()
knn_model = KNeighborsClassifier()

# train models
lr_model.fit(X_train, y_train)
lda_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)

# evaluate models
print(f"Logistic regression: {lr_model.score(X_test, y_test):.2f} test accuracy")
print(f"Linear discriminant: {lda_model.score(X_test, y_test):.2f} test accuracy")
print(f"K nearest neighbors: {knn_model.score(X_test, y_test):.2f} test accuracy")

# save models
with open("label_encoder.pkl", "wb") as pklfile:
    pickle.dump(label_encoder, pklfile, pickle.HIGHEST_PROTOCOL)
with open("standard_scaler.pkl", "wb") as pklfile:
    pickle.dump(scaler, pklfile, pickle.HIGHEST_PROTOCOL)
with open("variable_encoder.pkl", "wb") as pklfile:
    pickle.dump(values_encoder, pklfile, pickle.HIGHEST_PROTOCOL)
with open("lr_model.pkl", "wb") as pklfile:
    pickle.dump(lr_model, pklfile, pickle.HIGHEST_PROTOCOL)
with open("lda_model.pkl", "wb") as pklfile:
    pickle.dump(lda_model, pklfile, pickle.HIGHEST_PROTOCOL)
with open("knn_model.pkl", "wb") as pklfile:
    pickle.dump(knn_model, pklfile, pickle.HIGHEST_PROTOCOL)

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split


import joblib

#Loading the data
digits = datasets.load_digits()
#Transform data to a dataframe
df_digits = pd.DataFrame(digits.data)


X, y = digits.data, digits.target 


#Dataset separation
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20, random_state = 70)


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 7).fit(X_train, y_train) #n_neighbors : le K
'''predicted = knn.predict(X_test)
accuracy = knn.score(X_test, y_test) 
print("Précision du test:", accuracy)'''

# Save the trained model to a file
joblib.dump(knn, 'model_digits.pkl')
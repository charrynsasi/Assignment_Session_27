#K-nearest neighbors

# Importing the libraries
import pandas as pd

# Importing the dataset
nba = pd.read_csv('nba_2013.csv')

##Feature selection##

nba.drop(['player','season_end','season','bref_team_id'],axis=1,inplace=True)

#Split our dependant and independant feature
X=nba.drop(['pts'],axis=1)
y=nba.iloc[:,26]

##Feature Engineering##

 #Encoding categorical feature
poss=pd.get_dummies(X['pos'], drop_first=True)

 #Drop the columns
X.drop(['pos'],axis=1,inplace=True) 

 #Concat the dummy variables
X=pd.concat([X,poss], axis=1)

 #Handle Nan Values
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer=imputer.fit(X.iloc[:,:])
X.iloc[:,:]=imputer.transform(X.iloc[:,:])

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Fitting K-NN to the Training set
from  sklearn.neighbors import KNeighborsRegressor
regressor = KNeighborsRegressor(n_neighbors=5, weights='distance',metric='minkowski', p=2)
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

#measure accuracy
from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv = 10)
accuracies=accuracies.mean()




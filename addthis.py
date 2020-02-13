from sklearn.metrics import mean_squared_error 

from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics


#Predict using cross validation with 10 folds
predictions = cross_val_predict(Lregression, X, y, cv=10)

#Print the predictions
print(predictions)

#Print mean score fo the cross validation 
scores = cross_val_score(Lregression, X, y, cv=2)
print(np.mean(scores))

'''
The score of accuracy averages at about 0.704 which is lower than when we dont
use cross validation. When I increasee the fold size the accuracy goes down. 

'''

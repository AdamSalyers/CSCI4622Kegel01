from sklearn.metrics import accuracy_score
class AdaBoost:
    def __init__(self, n_learners=20, base=DecisionTreeClassifier(max_depth=3), random_state=1234):
        """
        Create a new adaboost classifier.
        
        Args:
            N (int, optional): Number of weak learners in classifier.
            base (BaseEstimator, optional): Your general weak learner 
            random_state (int, optional): set random generator.  needed for unit testing. 

        Attributes:
            base (estimator): Your general weak learner 
            n_learners (int): Number of weak learners in classifier.
            alpha (ndarray): Coefficients on weak learners. 
            learners (list): List of weak learner instances. 
        """
        
        np.random.seed(random_state)
        
        self.n_learners = n_learners 
        self.base = base
        self.alpha = np.zeros(self.n_learners)
        self.learners = []
        
    def fit(self, X_train, y_train):
        """
        Train AdaBoost classifier on data. Sets alphas and learners. 
        
        Args:
            X_train (ndarray): [n_samples x n_features] ndarray of training data   
            y_train (ndarray): [n_samples] ndarray of data 
        """

        # =================================================================
        # TODO 

        # Note: You can create and train a new instantiation 
        # of your sklearn decision tree as follows 

        # w = np.ones(len(y_train))
        # h = clone(self.base)
        # h.fit(X_train, y_train, sample_weight=w)
        # =================================================================
        
        # YOUR CODE HERE
        
        w = np.ones(len(y_train))
        h = clone(self.base)
        
        for i in range(self.n_learners):
            
            h.fit(X_train, y_train, sample_weight = w)
            y_pred = h.predict(X_train)
            
            notFitted = (y_pred != y_train)
            error = self.error_rate(y_train, y_pred, w)
            self.alpha[i] = (0.5)*np.log((1-error)/error)
            
            w *= np.exp(self.alpha[i] * notFitted * ((w>0) | (self.alpha[i] < 0)))
            
            w /= sum(w)
            
            self.learners.append(h)
        
        #raise NotImplementedError()
        
            
    def error_rate(self, y_true, y_pred, weights):
        # =================================================================
        # TODO 

        # Implement the weighted error rate
        # =================================================================
        # YOUR CODE HERE
        
        sumWeights = 0.001
        for i in range(len(y_true)):
            if y_pred[i] != y_true[i]:
                sumWeights += weights[i]
        return sumWeights
        
        
        #raise NotImplementedError()
        
    def predict(self, X):
        """
        Adaboost prediction for new data X.
        
        Args:
            X (ndarray): [n_samples x n_features] ndarray of data 
            
        Returns: 
            yhat (ndarray): [n_samples] ndarray of predicted labels {-1,1}
        """

        # =================================================================
        # TODO
        # =================================================================
        yhat = np.zeros(X.shape[0])
        
        # YOUR CODE HERE
        #h = clone(self)
        
        counter = [0] * len(X)
        
        for i in range(self.n_learners):
            
            pred = self.learners[i].predict(X)
            for j in range(len(X)):
                counter[j] += ((self.alpha[i]) * (pred[j]))
                
        for k in range(len(X)):
            if counter[k] >= 0:
                yhat[k] = 1
            else:
                yhat[k] = -1
        
        return yhat
        
        #raise NotImplementedError()
    
    def score(self, X, y):
        """
        Computes prediction accuracy of classifier.  
        
        Args:
            X (ndarray): [n_samples x n_features] ndarray of data 
            y (ndarray): [n_samples] ndarray of true labels  
            
        Returns: 
            Prediction accuracy (between 0.0 and 1.0).
        """
        
        # YOUR CODE HERE
        
        
        y_pred = self.predict(X)
        score = accuracy_score(y, y_pred)
        return score
        
        
        #raise NotImplementedError()
    
    def staged_score(self, X, y):
        """
        Computes the ensemble score after each iteration of boosting 
        for monitoring purposes, such as to determine the score on a 
        test set after each boost.
        
        Args:
            X (ndarray): [n_samples x n_features] ndarray of data 
            y (ndarray): [n_samples] ndarray of true labels  
            
        Returns: 
            scores (ndarary): [n_learners] ndarray of scores 
        """

        scores = []
        
        
         # YOUR CODE HERE
        
        
        for i in range(self.n_learners):
            scores.append(self.learners[i].score(X, y))
        
        
       
        #raise NotImplementedError()
        
        return np.array(scores)        
        

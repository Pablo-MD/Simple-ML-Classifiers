import numpy as np
from scipy.special import expit

class NaiveBayesClassifier():
    def __init__(self, k=1):
        # Variables to store model parameters
        self.k = k
        self.PC = None
        self.PAC = None
        self.trained = False
    
    def train(self, X, y):
        self.trained = True
        # We will store both the probabilities of the labels and attribute probabilities given the label in dictionaries
        y_unique=np.unique(y)
        self.PAC = dict.fromkeys(y_unique)
        self.PC = dict.fromkeys(y_unique)
        N = len(y)
        
        for c in self.PC:
            # Calculate PC as the count of a given label divided by the total of instances and create the PA list of length
            # equal to the number of attributes.
            nc = np.count_nonzero(y == c)
            self.PC[c] = nc / N
            PA = np.empty(len(X[0])).tolist()
            
            for i in range(len(PA)):
                # Get possible values an attribute can take and create an empty dictionary at index i of the PA list.
                A = np.unique(X[:, i])
                PA[i] = dict.fromkeys(A)
                
                for a in A:
                    # Count the number of values with a given value 'a' and label 'c' using np.logical_and()
                    # Assign the dictionary values in the PA list using the formula with Laplace's smoothing.
                    Pac = np.count_nonzero(np.logical_and(y == c, X[:, i] == a))
                    PA[i][a] = (Pac + self.k) / (nc + self.k * len(A))
            
            # Store the result in the PAC class dictionary with the corresponding label.
            self.PAC[c] = PA
    
    def classify_prob(self, example):
        if not self.trained:
            raise ClassifierNotTrained("The model has not been trained")
        
        # Create an empty list for probabilities. Check if the input example is a 1-dimensional list or a numpy array
        # to ensure consistent results.
        probabilities = []
        
        if len(example.shape) == 1:
            example = np.array([example])
        
        # Iterate through all instances in the example and assign a dictionary with the labels from PC.
        for row in example:
            probability = dict.fromkeys(np.unique(list(self.PC.keys())))
            
            # Iterate through the labels, where 's' is a sum (since we're taking logarithms) and retrieve the values
            # stored in PC and PAC.
            for c in self.PC:
                s = np.log(self.PC[c])
                
                for i in row:
                    s += np.log(self.PAC[c][np.where(np.array(row) == i)[0][0]][i])
                
                probability[c] = np.exp(s)
            
            # Assign the sum values to the probability dictionary with label 'c'. Once all label values are iterated,
            # append them to the list and return.
            probabilities.append(probability)
        
        return probabilities
    
    def classify(self, example):
        # Create a list for class predictions. First, execute the class function that returns a list of probabilities
        # for each label.
        class_predictions = []
        probabilities = self.classify_prob(example)
        
        for probability in probabilities:
            # Take the maximum value from the corresponding dictionary values for each instance and return the
            # corresponding labels.
            class_predictions.append(max(probability, key=probability.get))
        
        return class_predictions

class ClassifierNotTrained(Exception):
    pass


class LogisticRegressionMiniBatch:
    # Constructor that stores the variable values throughout the class
    def __init__(self, classes=[0, 1], normalization=False, rate=0.1, rate_decay=False, batch_size=64):
        self.classes = classes
        self.normalization = normalization
        self.rate = rate
        self.rate0 = rate
        self.rate_decay = rate_decay
        self.batch_size = batch_size
        self.mean = None
        self.std = None
        self.weights = None

    # Create a function to calculate gradient descent
    def __gradient_descent(self, X, y, n_epochs):
        # Assign the number of features and instances using the shape function
        n_samples, n_features = X.shape
        
        # For each epoch, obtain a random batch. Perform the dot product of weights w*x using .dot and
        # apply the sigmoid function from scipy. Update the weights with the result and calculate the new rate.
        for i in range(n_epochs):
            indices = np.random.choice(n_samples, self.batch_size, replace=False)
            X_batch = X[indices]
            y_batch = y[indices]
            
            y_prediction = expit(self.weights.T.dot(X_batch.T))
            gradients = X_batch.T.dot(y_batch - y_prediction)
            self.weights += self.rate * gradients

        if self.rate_decay:
            self.rate = self.rate0 * (1 / (1 + i))

    def train(self, X, y, n_epochs, reset_weights=False, initial_weights=None):
        # Normalize the data and store the mean and standard deviation to apply them to the prediction set
        if self.normalization:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
            X = (X - self.mean) / self.std
        
        # Add a new column to the data that corresponds to w_0 (bias)
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

        if reset_weights or self.weights is None:
            self.weights = np.random.uniform(-1, 1, size=X.shape[1])
        else:
            self.weights = initial_weights
        
        # Call the gradient descent function
        self.__gradient_descent(X, y, n_epochs)
        
    def classify_prob(self, example):
        probabilities = []
        if len(example.shape) == 1:
            example = np.array([example])
        if self.normalization:
            example = (example - self.mean) / self.std
        for row in example:
            # Normalize using the stored mean and standard deviation, and add the column with 1s (w_0) by reshaping
            # the data from the example.
            row = row.reshape(1, -1)
            row = np.concatenate((np.ones((row.shape[0], 1)), row), axis=1)
            row = row.reshape(-1)
            probabilities.append(expit(row.dot(self.weights)))
        # Add the probabilities to the list as the sigmoid of the dot product h*x and return
        return probabilities

    def classify(self, example):
        # Call the classify_prob function to obtain the list with sigmoid values,
        # and distinguish between one classification or another based on a threshold of > 0.5
        probabilities = self.classify_prob(example)
        return [self.classes[1] if p > 0.5 else self.classes[0] for p in probabilities]
    

class RL_OvR:
    # Variables stored for use in the class
    def __init__(self, classes, rate=0.1, normalization=False, rate_decay=False, batch_size=64):
        self.classes = classes
        self.normalization = normalization
        self.rate = rate
        self.rate_decay = rate_decay
        self.batch_size = batch_size
        self.classifiers = None
    
    def train(self, X, y, n_epochs):
        # Create an empty list that will contain as many classifiers as there are classes
        self.classifiers = []
        # Iterate over all labels, assigning 1 to the matching value and 0 to the rest,
        # creating binary classification where we can use LogisticRegressionMiniBatch
        for label in self.classes:
            binary_classes = np.where(y == label, 1, 0)
            classifier = LogisticRegressionMiniBatch(rate=self.rate, rate_decay=self.rate_decay, normalization=self.normalization, batch_size=self.batch_size)
            classifier.train(X, binary_classes, n_epochs)
            self.classifiers.append(classifier)
            
    def classify(self, example):
        probabilities, predictions = [], []
        # In this case, it won't be necessary to distinguish the size of the example since it is already handled
        # by the logistic classifiers by default.
        # Apply all the previously declared classifiers to the example data
        for classifier in self.classifiers:
            probabilities.append(np.array(classifier.classify_prob(example)))
        # Create an ndarray where each row contains the values returned by the sigmoid functions
        probabilities = np.column_stack([array for array in probabilities])
        # For each row, find the maximum value and return the corresponding class for that value
        for probability in probabilities:
            predictions.append(self.classes[np.argmax(probability)])
        return predictions

#Simple function that can be used to calculate the performance of any of the previous models
def performance(model, X, y):
    hit, miss= 0, 0
    classification=model.clasifica(X)
    for row in range(0,len(X)):
        if(classification[row]==y[row]):
           hit+=1
        else:
           miss+=1
    return hit/(hit+miss)

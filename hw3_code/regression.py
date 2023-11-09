import numpy as np
from typing import Tuple, List


class Regression(object):
    def __init__(self):
        pass

    def rmse(self, pred: np.ndarray, label: np.ndarray) -> float:  # [5pts]
        """
        Calculate the root mean square error.

        Args:
            pred: (N, 1) numpy array, the predicted labels
            label: (N, 1) numpy array, the ground truth labels
        Return:
            A float value
        """

        #calculate the rmse
        rmse = np.sqrt(np.mean((pred - label)**2))
        
        return rmse

    def construct_polynomial_feats(
        self, x: np.ndarray, degree: int
    ) -> np.ndarray:  # [5pts]
        """
        Given a feature matrix x, create a new feature matrix
        which is all the possible combinations of polynomials of the features
        up to the provided degree

        Args:
            x:
                1-dimensional case: (N,) numpy array
                D-dimensional case: (N, D) numpy array
                Here, N is the number of instances and D is the dimensionality of each instance.
            degree: the max polynomial degree
        Return:
            feat:
                For 1-D array, numpy array of shape Nx(degree+1), remember to include
                the bias term. feat is in the format of:
                [[1.0, x1, x1^2, x1^3, ....,],
                 [1.0, x2, x2^2, x2^3, ....,],
                 ......
                ]
        Hints:
            - For D-dimensional array: numpy array of shape N x (degree+1) x D, remember to include
            the bias term.
            - Example:
            For inputs x: (N = 3 x D = 2) and degree: 3,
            feat should be:

            [[[ 1.0        1.0]
                [ x_{1,1}    x_{1,2}]
                [ x_{1,1}^2  x_{1,2}^2]
                [ x_{1,1}^3  x_{1,2}^3]]

                [[ 1.0        1.0]
                [ x_{2,1}    x_{2,2}]
                [ x_{2,1}^2  x_{2,2}^2]
                [ x_{2,1}^3  x_{2,2}^3]]

                [[ 1.0        1.0]
                [ x_{3,1}    x_{3,2}]
                [ x_{3,1}^2  x_{3,2}^2]
                [ x_{3,1}^3  x_{3,2}^3]]]

        """
        if x.ndim == 1:
            x = x.reshape(-1, 1)  # Convert 1-D array to a 2-D array with one column
        
        N, D = x.shape
        feat = np.ones((N, degree + 1, D))
        
        for i in range(1, degree + 1):
            feat[:, i, :] = x ** i
        
        return feat

    def predict(self, xtest: np.ndarray, weight: np.ndarray) -> np.ndarray:  # [5pts]
        """
        Using regression weights, predict the values for each data point in the xtest array

        Args:
            xtest: (N,1+D) numpy array, where N is the number
                    of instances and D is the dimensionality
                    of each instance with a bias term
            weight: (1+D,1) numpy array, the weights of linear regression model
        Return:
            prediction: (N,1) numpy array, the predicted labels
        """
        prediction = xtest.dot(weight)
        
        return prediction
    
    # =================
    # LINEAR REGRESSION
    # =================

    def linear_fit_closed(
        self, xtrain: np.ndarray, ytrain: np.ndarray
    ) -> np.ndarray:  # [5pts]
        """
        Fit a linear regression model using the closed form solution

        Args:
            xtrain: (N,1+D) numpy array, where N is number
                    of instances and D is the dimensionality
                    of each instance with a bias term
            ytrain: (N,1) numpy array, the true labels
        Return:
            weight: (1+D,1) numpy array, the weights of linear regression model
        Hints:
            - For pseudo inverse, you can use numpy linear algebra function (np.linalg.pinv)
        """
        # #(X^T * X)^-1 * X^T * y
        # weight = np.matmul(np.linalg.pinv(np.matmul(xtrain.T, xtrain)), np.matmul(xtrain.T, ytrain))
        # #weight = np.linalg.pinv(xtrain.T.dot(xtrain)).dot(xtrain.T).dot(ytrain)
        
        # return weight
    
        # Compute the Moore-Penrose pseudoinverse of X
        X_pinv = np.linalg.pinv(xtrain)
        
        # Compute the weights (coefficients)
        weight = X_pinv.dot(ytrain)
        
        return weight


    def linear_fit_GD(
        self,
        xtrain: np.ndarray,
        ytrain: np.ndarray,
        epochs: int = 5,
        learning_rate: float = 0.001,
    ) -> Tuple[np.ndarray, List[float]]:  # [5pts]
        """
        Fit a linear regression model using gradient descent.
        Although there are many valid initializations, to pass the local tests
        initialize the weights with zeros.

        Args:
            xtrain: (N,1+D) numpy array, where N is number
                    of instances and D is the dimensionality
                    of each instance with a bias term
            ytrain: (N,1) numpy array, the true labels
        Return:
            weight: (1+D,1) numpy array, the weights of linear regression model
            loss_per_epoch: (epochs,) list of floats, rmse of each epoch
        """

        # N, D = xtrain.shape
        # weight = np.zeros((D, 1))  # Initialize weights to zeros
        # loss_per_epoch = []

        # for epoch in range(epochs):
        #     # Calculate predictions
        #     preds = self.predict(xtrain, weight)
        #     # Calculate the error
        #     errors = preds - ytrain
        #     # Compute the gradient
        #     gradient = np.dot(xtrain.T, errors) / N
        #     # Update weights
        #     weight -= learning_rate * gradient
        #     # Calculate the loss for this epoch
        #     loss = np.sum(errors ** 2) / (2 * N)
        #     loss_per_epoch.append(loss)

        # return weight, loss_per_epoch
        N, D = xtrain.shape
        weight = np.zeros((D, 1))  # Initialize weights to zeros
        loss_per_epoch = []

        for epoch in range(epochs):
            preds = self.predict(xtrain, weight)
            errors = preds - ytrain
            gradient = np.dot(xtrain.T, errors) / N
            weight -= learning_rate * gradient
            loss = self.rmse(preds, ytrain)
            loss_per_epoch.append(loss)

        return weight, loss_per_epoch

    def linear_fit_SGD(
        self,
        xtrain: np.ndarray,
        ytrain: np.ndarray,
        epochs: int = 100,
        learning_rate: float = 0.001,
    ) -> Tuple[np.ndarray, List[float]]:  # [5pts]
        """
        Fit a linear regression model using stochastic gradient descent.
        Although there are many valid initializations, to pass the local tests
        initialize the weights with zeros.

        Args:
            xtrain: (N,1+D) numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance with a bias term
            ytrain: (N,1) numpy array, the true labels
            epochs: int, number of epochs
            learning_rate: float, value of regularization constant
        Return:
            weight: (1+D,1) numpy array, the weights of linear regression model
            loss_per_step: (N*epochs,) list of floats, rmse calculated after each update step

        NOTE: For autograder purposes, iterate through the dataset SEQUENTIALLY, NOT stochastically.


        Note: Keep in mind that the number of epochs is the number of
        complete passes through the training dataset. SGD updates the
        weight for one datapoint at a time, but for each epoch, you'll
        need to go through all of the points.
        """
        # N, D = xtrain.shape
        # weight = np.zeros((D, 1))  # Initialize weights to zeros
        # loss_per_step = []

        # for epoch in range(epochs):
        #     for i in range(N):
        #         xi = xtrain[i, :].reshape(1, -1)
        #         yi = ytrain[i].reshape(1, -1)
        #         pred = self.predict(xi, weight)
        #         error = pred - yi
        #         # Compute the gradient for a single sample
        #         gradient = np.dot(xi.T, error)
        #         # Update weights
        #         weight -= learning_rate * gradient
        #         # Calculate the loss after this update
        #         loss = np.sum(error ** 2) / 2
        #         loss_per_step.append(loss)

        # return weight, loss_per_step
    
        N, D = xtrain.shape
        weight = np.zeros((D, 1))  # Initialize weights to zeros
        loss_per_step = []

        for epoch in range(epochs):
            for i in range(N):
                xi = xtrain[i, :].reshape(1, -1)
                yi = ytrain[i].reshape(1, -1)
                pred = self.predict(xi, weight)
                error = pred - yi
                gradient = np.dot(xi.T, error)
                weight -= learning_rate * gradient
                loss = error**2 / 2  # No need to divide by N since it's a single sample
                loss_per_step.append(loss[0][0])  # Extract the scalar value

        # For SGD, loss reported per step is the sum of losses divided by the number of steps
        #average_loss_per_epoch = [np.sqrt(np.mean(loss_per_step[i * N:(i + 1) * N])) for i in range(epochs)]
    
        return weight, loss_per_step

    # =================
    # RIDGE REGRESSION
    # =================

    def ridge_fit_closed(
        self, xtrain: np.ndarray, ytrain: np.ndarray, c_lambda: float
    ) -> np.ndarray:  # [5pts]
        """
        Fit a ridge regression model using the closed form solution

        Args:
            xtrain: (N,1+D) numpy array, where N is
                    number of instances and D is the dimensionality
                    of each instance with a bias term
            ytrain: (N,1) numpy array, the true labels
            c_lambda: float value, value of regularization constant
        Return:
            weight: (1+D,1) numpy array, the weights of ridge regression model
        Hints:
            - For pseudo inverse, you can use numpy linear algebra function (np.linalg.pinv)
            - You should adjust your I matrix to handle the bias term differently than the rest of the terms
        """

        # The number of features, D, includes the bias term, so subtract 1 for the identity matrix size.
        D = xtrain.shape[1] - 1
        I = np.eye(D + 1)
        I[0, 0] = 0  # To not regularize the bias term

        ridge_weight = np.matmul(np.linalg.pinv(np.matmul(xtrain.T, xtrain) + c_lambda * I), np.matmul(xtrain.T, ytrain))
        
        #weight = np.linalg.pinv(xtrain.T.dot(xtrain)).dot(xtrain.T).dot(ytrain)
        
        return ridge_weight

    def ridge_fit_GD(
        self,
        xtrain: np.ndarray,
        ytrain: np.ndarray,
        c_lambda: float,
        epochs: int = 500,
        learning_rate: float = 1e-7,
    ) -> Tuple[np.ndarray, List[float]]:  # [5pts]
        """
        Fit a ridge regression model using gradient descent.
        Although there are many valid initializations, to pass the local tests
        initialize the weights with zeros.

        Args:
            xtrain: (N,1+D) numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance with a bias term
            ytrain: (N,1) numpy array, the true labels
            c_lambda: float value, value of regularization constant
            epochs: int, number of epochs
            learning_rate: float
        Return:
            weight: (1+D,1) numpy array, the weights of linear regression model
            loss_per_epoch: (epochs,) list of floats, rmse of each epoch
        Hints:
            - You should avoid applying regularization to the bias term in the gradient update
        """
        N, D = xtrain.shape
        weight = np.zeros((D, 1))  # Initialize weights with zeros
        loss_per_epoch = []  # List to keep track of loss per epoch

        for epoch in range(epochs):
            predictions = xtrain.dot(weight)  # Calculate predictions
            errors = ytrain - predictions  # Compute error vector
            
            # Compute the gradient
            regularized_weight = np.copy(weight)
            regularized_weight[0, 0] = 0  # Exclude bias term from regularization
            gradient = -(xtrain.T.dot(errors)) / N + c_lambda * regularized_weight / N
            
            # Update weights
            weight -= learning_rate * gradient

            # Compute the Ridge loss (without considering the regularization term for loss tracking)
            loss = np.mean(errors**2) / 2
            loss_per_epoch.append(loss)

        return weight, loss_per_epoch

    def ridge_fit_SGD(
        self,
        xtrain: np.ndarray,
        ytrain: np.ndarray,
        c_lambda: float,
        epochs: int = 100,
        learning_rate: float = 0.001,
    ) -> Tuple[np.ndarray, List[float]]:  # [5pts]
        """
        Fit a ridge regression model using stochastic gradient descent.
        Although there are many valid initializations, to pass the local tests
        initialize the weights with zeros.

        Args:
            xtrain: (N,1+D) numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance with a bias term
            ytrain: (N,1) numpy array, the true labels
            c_lambda: float, value of regularization constant
            epochs: int, number of epochs
            learning_rate: float
        Return:
            weight: (1+D,1) numpy array, the weights of linear regression model
            loss_per_step: (N*epochs,) list of floats, rmse calculated after each update step

        NOTE: For autograder purposes, iterate through the dataset SEQUENTIALLY, NOT stochastically.

        Hints:
            - Keep in mind that the number of epochs is the number of complete passes
            through the training dataset. SGD updates the weight for one datapoint at
            a time. For each epoch, you'll need to go through all of the points.
            - You should avoid applying regularization to the bias term in the gradient update
        """
        raise NotImplementedError

    def ridge_cross_validation(
        self, X: np.ndarray, y: np.ndarray, kfold: int = 10, c_lambda: float = 100
    ) -> List[float]:  # [5 pts]
        """
        For each of the kfolds of the provided X, y data, fit a ridge regression model
        and then evaluate the RMSE. Return the RMSE for each kfold

        Args:
            X : (N,1+D) numpy array, where N is the number of instances
                and D is the dimensionality of each instance with a bias term
            y : (N,1) numpy array, true labels
            kfold: int, number of folds you should take while implementing cross validation.
            c_lambda: float, value of regularization constant
        Returns:
            loss_per_fold: list[float], RMSE loss for each kfold
        Hints:
            - np.concatenate might be helpful.
            - Use ridge_fit_closed for this function.
            - Look at 3.5 to see how this function is being used.
            - If kfold=10:
                split X and y into 10 equal-size folds
                use 90 percent for training and 10 percent for test
        """
        fold_size = X.shape[0] // kfold
        loss_per_fold = []  # Initialize list to store the RMSE for each fold
    
        # Iterate over each fold
        for i in range(kfold):
            # Generate indices to slice out validation and training sets
            start_val, end_val = i * fold_size, (i + 1) * fold_size
        
            # Create validation set and training sets
            X_val, y_val = X[start_val:end_val], y[start_val:end_val]
            X_train = np.concatenate((X[:start_val], X[end_val:]), axis=0)
            y_train = np.concatenate((y[:start_val], y[end_val:]), axis=0)
        
            # Fit model using the training set
            weight = self.ridge_fit_closed(X_train, y_train, c_lambda)
        
            # Predict on the validation set
            #pred_val = X_val.dot(weight)
            pred_val = self.predict(X_val, weight)
        
            # Calculate RMSE for the validation set and store it
            rmse = self.rmse(pred_val, y_val)
            loss_per_fold.append(rmse)
        
        return loss_per_fold

    def hyperparameter_search(
        self, X: np.ndarray, y: np.ndarray, lambda_list: List[float], kfold: int
    ) -> Tuple[float, float, List[float]]:
        """
        FUNCTION PROVIDED TO STUDENTS

        Search over the given list of possible lambda values lambda_list
        for the one that gives the minimum average error from cross-validation

        Args:
            X : (N, 1+D) numpy array, where N is the number of instances and
                D is the dimensionality of each instance with a bias term
            y : (N,1) numpy array, true labels
            lambda_list: list of regularization constants (lambdas) to search from
            kfold: int, Number of folds you should take while implementing cross validation.
        Returns:
            best_lambda: (float) the best value for the regularization const giving the least RMSE error
            best_error: (float) the average RMSE error achieved using the best_lambda
            error_list: list[float] list of average RMSE loss for each lambda value given in lambda_list
        """

        best_error = None
        best_lambda = None
        error_list = []

        for lm in lambda_list:
            err = self.ridge_cross_validation(X, y, kfold, lm)
            mean_err = np.mean(err)
            error_list.append(mean_err)
            if best_error is None or mean_err < best_error:
                best_error = mean_err
                best_lambda = lm

        return best_lambda, best_error, error_list

import numpy as np
from knn import KNN

############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """
    assert len(real_labels) == len(predicted_labels)
    
    # tp = true-positive
    # fp = false positive
    # fn = false negative
    tp = float(sum(1 for x,y in zip(real_labels, predicted_labels) if (x == 1 and y ==1)))
    fp = float(sum(1 for x,y in zip(real_labels, predicted_labels) if (x == 0 and y ==1)))
    fn = float(sum(1 for x,y in zip(real_labels, predicted_labels) if (x == 1 and y ==0)))
    
    # define precision and recall
    if ((tp+fp)==0) or ((tp+fn)==0):
        precision = 0
        recall = 0
    else:
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
    
    # define and return f1-score
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * ((precision*recall)/(precision + recall))
    return f1

class Distances:
    @staticmethod
    def canberra_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        #distance = sum((abs(x - y))/(abs(x) + abs(y)) for x,y in zip(point1, point2))
        distance = 0
        for x,y in zip(point1, point2):
            if (abs(x)+abs(y)) > 0:
                distance += (abs(x - y))/(abs(x) + abs(y))
        return distance

    @staticmethod
    def minkowski_distance(point1, point2):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        distance = (sum((abs(x - y))**3 for x,y in zip(point1,point2)))**(1/3)
        return distance

    @staticmethod
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        sumsquares = sum((x - y)**2 for x,y in zip(point1, point2))
        distance = np.sqrt(sumsquares)
        return distance
        
    @staticmethod
    def inner_product_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        distance = sum(x*y for x,y in zip(point1, point2))
        return distance

    @staticmethod
    def cosine_similarity_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        inner_xx = sum(x*x for x in point1)
        inner_yy = sum(y*y for y in point2)
        inner_xy = sum(x*y for x,y in zip(point1, point2))
        if (inner_xx == 0) or (inner_yy == 0):
            cos_sim = 0
        else: 
            cos_sim = inner_xy/((np.sqrt(inner_xx))*np.sqrt(inner_yy))
        distance = 1 - cos_sim
        return distance
    
    @staticmethod
    def gaussian_kernel_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        sumsquares = sum((x - y)**2 for x,y in zip(point1, point2))
        distance = -np.exp(-sumsquares/2) 
        return distance


class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you should try different distance function you implemented in part 1.1, and find the best k.
        Use k range from 1 to 30 and increment by 2. Use f1-score to compare different models.

        :param distance_funcs: dictionary of distance functions you must use to calculate the distance.
            Make sure you loop over all distance functions for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val:  List[List[int]] Validation data set will be used on your KNN predict function to produce
            predicted labels and tune k and distance function.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_function and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function and self.best_model respectively.
        NOTE: self.best_scaler will be None

        NOTE: When there is a tie, choose model based on the following priorities:
        Then check distance function  [canberra > minkowski > euclidean > gaussian > inner_prod > cosine_dist]
        If they have same distance function, choose model which has a smaller k.
        """
        # range of k values
        last_k = min(len(x_train)+1, 30)
        k_array = np.arange(1, last_k, 2)
        
        # initialize local variables
        best_f1 = -1
        best_k = 1
        best_distance_function = 'canberra'
        best_model = None
        
        # loop through k values
        #for k in reversed(k_array):
        for key in distance_funcs:
        
            
            # loop through distance functions
            #for key in distance_funcs:
            for k in k_array:
                
                # run KNN
                knn_instance = KNN(k, distance_funcs[key]) # initiate knn object
                knn_instance.train(x_train, y_train) # train model
                y_val_predicted = knn_instance.predict(x_val) # returns predicted labels
                
                # compute f1 score for this knn instance
                this_f1 = f1_score(y_val, y_val_predicted)
                   
                if (this_f1 > best_f1):
                    best_f1 = this_f1 # update best f1 score 
                    best_k = k # update best k
                    best_distance_function = key # update best distance function
                    best_model = knn_instance # update best knn model
                    #print(f'updated k: {k}, updated distance function: {best_distance_function}')
        
        # You need to assign the final values to these variables
        self.best_k = best_k
        self.best_distance_function = best_distance_function
        self.best_model = best_model
        #print(f'best k: {best_k}, best distance function: {best_distance_function}, from model:{best_model.k}, \
        #{best_model.distance_function}')

    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        This part is similar to Part 1.3 except that before passing your training and validation data to KNN model to
        tune k and distance function, you need to create the normalized data using these two scalers to transform your
        data, both training and validation. Again, we will use f1-score to compare different models.
        Here we have 3 hyperparameters i.e. k, distance_function and scaler.

        :param distance_funcs: dictionary of distance funtions you use to calculate the distance. Make sure you
            loop over all distance function for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param scaling_classes: dictionary of scalers you will use to normalized your data.
        Refer to test.py file to check the format.
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val: List[List[int]] validation data set you will use on your KNN predict function to produce predicted
            labels and tune your k, distance function and scaler.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_funtion, scaler and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function, self.best_scaler and self.best_model respectively

        NOTE: When there is a tie, choose model based on the following priorities:
        For normalization, [min_max_scale > normalize];
        Then check distance function  [canberra > minkowski > euclidean > gaussian > inner_prod > cosine_dist]
        If they have same distance function, choose model which has a less k.
        """
        # range of k values
        last_k = min(len(x_train)+1, 30)
        k_array = np.arange(1, last_k, 2)
        
        # initialize local variables
        best_f1 = -1
        best_k = 1
        best_distance_function = 'canberra'
        best_model = None
        best_scaler = 'min_max_scale'
        
        # loop through k values
        #for k in reversed(k_array):
        for scaler in scaling_classes:
            
            # scale data 
            scaler_function = scaling_classes[scaler]
            scaler_obj = scaler_function()
            x_train = scaler_obj(x_train)
            x_val = scaler_obj(x_val)

            # loop through distance functions
            for key in distance_funcs:
                
                # loop through scalers
                #for scaler in scaling_classes: 
                for k in k_array:
                    
                    # run KNN
                    knn_instance = KNN(k, distance_funcs[key]) # initiate knn object
                    knn_instance.train(x_train, y_train) # train model
                    y_val_predicted = knn_instance.predict(x_val) # returns predicted labels

                    # compute f1 score for this knn instance
                    this_f1 = f1_score(y_val, y_val_predicted)

                    if this_f1 > best_f1:
                        best_f1 = this_f1 # update best f1 score 
                        best_k = k # update best k
                        best_distance_function = key # update best distance function
                        best_model = knn_instance # update best knn model
                        best_scaler = scaler # update scaler
                        #print(f'updated k: {k}, updated distance function: {best_distance_function}\
                        #, updated scaler: {scaler}, updated f1: {best_f1}')
        
        # You need to assign the final values to these variables
        self.best_k = best_k
        self.best_distance_function = best_distance_function
        self.best_scaler = best_scaler
        self.best_model = best_model
        #print(f'best k: {best_k}, best distance function: {best_distance_function}, from model:{best_model.k}, \
        #{best_model.distance_function}')

class NormalizationScaler:
    def __init__(self):
        pass

    def __call__(self, features):
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        scaled_features = []
        for feature in features:
            if sum(abs(x) for x in feature) == 0:
                scaled = feature
            else:
                norm = np.sqrt(sum(x*x for x in feature))
                scaled = feature/norm
                scaled = scaled.tolist()
            scaled_features.append(scaled)
            
        return scaled_features


class MinMaxScaler:
    """
    Please follow this link to know more about min max scaling
    https://en.wikipedia.org/wiki/Feature_scaling
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
    will be the training set.

    Hints:
        1. Use a variable to check for first __call__ and only compute
            and store min/max in that case.

    Note:
        1. You may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler1 = MinMaxScale()
        train_features_scaled = scaler1(train_features)
        # train_features_scaled should be equal to [[0, 1], [1, 0]]

        test_features_scaled = scaler1(test_features)
        # test_features_scaled should be equal to [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]

    """
    
    def __init__(self):
        self.counter = 0

    def __call__(self, features):
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        # convert to np array
        features = np.array(features)
        
        # first call to __call__ 
        if self.counter == 0:
            # find min/max of attributes (columns)
            self.min = features.min(axis=0)
            self.max = features.max(axis=0)
            
        # scale by min/max
        columns = features.shape[1]
        for i in range(columns):
            denom = self.max[i]-self.min[i]
            if denom != 0:
                features[:,i] = (features[:,i]-self.min[i])/(denom)
            
        # increase counter
        self.counter += 1
            
        scaled_features = features.tolist()
        return scaled_features

        
        
        
        
        
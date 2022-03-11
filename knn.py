import numpy as np
from collections import Counter


class KNN:
    def __init__(self, k, distance_function):
        """
        :param k: int
        :param distance_function
        """
        self.k = k
        self.distance_function = distance_function

    def train(self, features, labels):
        """
        In this function, features is simply training data which is a 2D list with float values.
        For example, if the data looks like the following: Student 1 with features age 25, grade 3.8 and labeled as 0,
        Student 2 with features age 22, grade 3.0 and labeled as 1, then the feature data would be
        [ [25.0, 3.8], [22.0,3.0] ] and the corresponding label would be [0,1] 

        For KNN, the training process is just loading of training data. Thus, all you need to do in this function
        is create some local variable in KNN class to store this data so you can use the data in later process.
        :param features: List[List[float]]
        :param labels: List[int]
        """
        self.features = features
        self.labels = labels
        
    def get_k_neighbors(self, point):
        """
        This function takes one single data point and finds k-nearest neighbours in the training set.
        You already have your k value, distance function and you just stored all training data in KNN class with the
        train function. This function needs to return a list of labels of all k neighbors.
        :param point: List[float]
        :return:  List[int]
        """
        # create/allocate distances array
        dist = []
        
        # loop through each training point
        for training_point in self.features:
            tmp_dist = self.distance_function(point, training_point)
            dist.append(tmp_dist)
            
        # find index (indices) of k closest neighbors
        dist_arr = np.array(dist)
        ind = np.argpartition(dist_arr,self.k)[:self.k]

        # return list of labels of k closest neighbors
        list_labels = np.array(self.labels)[ind] 
        list_labels = list_labels.tolist()
        return list_labels    
        
    def predict(self, features):
        """
        This function takes 2D list of test data points, similar to those from train function. Here, you need to process
        every test data point, reuse the get_k_neighbours function to find the nearest k neighbours for each test
        data point, find the majority of labels for these neighbours as the predict label for that testing data point.
        Thus, you will get N predicted label for N test data point.
        This function need to return a list of predicted labels for all test data points.
        :param features: List[List[float]]
        :return: List[int]
        """
        predicted_labels = []
        # loop through each data point in input features
        for data_point in features:
            nn_labels = self.get_k_neighbors(data_point) # this returns the labels of k closest neighbors
            c = Counter(nn_labels) # creates count dictionary
            the_label = c.most_common(1)[0][0] # chooses majority count
            predicted_labels.append(the_label) # insert into return array
        
        return predicted_labels    


if __name__ == '__main__':
    print(np.__version__)

    
    
    
    
    
    
    
    
    
    
    
    
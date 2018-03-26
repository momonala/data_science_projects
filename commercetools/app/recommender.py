import pandas as pd 
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy.sparse.linalg import svds

def rmse(prediction, ground_truth):
    '''
    Compute root mean squared error loss metric between prediction and ground truth 
    
    Returns: 
        RMSE value 
    '''

    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))


class Recommender(): 
    def __init__(self, df, k = 20, test_size=0.2): 
        self.test_size = test_size
        self.k = k
        self.df = df

        self.train_pivot = None
        self.test_pivot = None
        self.train_rmse = None
        self.test_rmse = None
        self.preds = None
        self.preds_df = None 

        self.train_test_split()

    def create_pivot(self):
        '''create pivot table for matrix '''

        pivot = self.df.pivot(index='CustomerID', columns='StockCode', values='ones')
        pivot = pivot.fillna(0)
        A = pivot.as_matrix()

        #demean the matrix 
        A_mean = np.mean(A, axis=1)
    
        return pivot, A, A_mean


    def train_test_split(self): 
        '''
        Create training and testing matrices 
        
        Params: 
        -------
        df : pandas data frame - input data
        test_size : float - about of data to mask out for training 
        
        Returns: 
        --------
        training matrix : pandas df - matrix of ground truth with {test size} random indices masked out 
        testing matrix : pandas df - ground truth data 
        
        '''
        
        self.test_pivot, mat, man_mean = self.create_pivot() #create pivot table (ground truth test data)
        mask  = np.random.rand(*self.test_pivot.shape) < self.test_size #create mask 
        self.train_pivot = self.test_pivot.mask(mask).fillna(0) #mask the ground truth (train data)

        return 1


    def computeSVD(self, verbose=True): 
        '''
        Perform singular value decomposition
        
        Params: 
        k : int - number of features to decompose matrix into 
        train_data : np.array - data to compute SVD on 
        test_data: np.array - ground truth data to compute loss on 
        
        Returns: 
        train_rmse : float - RMSE on training data and predictions 
        test_rmse : float - RMSE on testing data and predictions 
        preds : np.array - prediction matrix - reconstruction of low-rank approximation 
        
        '''
        
        U, sigma, Vt = svds(self.train_pivot.values, k = self.k)
        sigma = np.diag(sigma)

        # reconstruct the low-rank matrix for predictions
        self.preds = np.dot(np.dot(U, sigma), Vt) 
        
        self.train_rmse = rmse(self.preds, self.train_pivot.values)
        self.test_rmse = rmse(self.preds, self.test_pivot.values)

        self.preds_df = pd.DataFrame(self.preds,
                                    columns = self.train_pivot.columns,
                                    index=self.train_pivot.index)

        if verbose: 
            print ('Model with {0} components acheived train RMSE of: {1}, test RMSE of {2}'.format(self.k, self.train_rmse, self.test_rmse))

        return 1

    def get_product_description(self, stockcode):
        '''given a product stock code, return its  most common product description'''
        
        all_names = self.df[self.df['StockCode'] == stockcode]['Description']
        true_desc = pd.value_counts(all_names).index[0] #pick the most popular 
        return true_desc


    def show_recommendations(self, customer_id, num_recom = 5, simpleID = True):
        '''generate a set of recommendations for a user, given the low rank matrix'''

        if simpleID: 
            # use reindexed IDs instead of True ones, for simplicty 
            customer_id = self.df.CustomerID.unique()[customer_id]
                    
        #generate the top recommendations
        preds = self.preds_df.loc[customer_id].sort_values(ascending=False)
        already_bought = self.df[self.df.CustomerID == customer_id].StockCode.values
        recommendations = preds.drop(already_bought)
        
        #pick the top results 
        recom = recommendations[:num_recom].index
        bought = already_bought[:num_recom]

        bought_desc = [self.get_product_description(item) for item in bought] 
        recom_desc = [self.get_product_description(item) for item in recom]

        return bought_desc, recom_desc

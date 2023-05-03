import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
import pickle
import warnings
warnings.filterwarnings(action='ignore')
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import time
import random

class RecommenderSystem:
   
    def __init__(self):
        self.attribute = "default value"
        
    def recommender_eda(self, feature_file, user_file, featureId, userId, value, featureTitle):
    
        """ Organize raw data 
              Arg:
              	feature_file : file containing fature and the rating of the features
              	user_file : file containing user and their rating of the features
              	featureId : Id of the feature that identfy feature in both files
              	
              	NB: both file are pandas dataframe
              	
              return q_matrix, user_feature_matrix(na non fill), user_feature_matrix(na fill)
        """
        featureId = str(featureId)
        userId = str(userId)
        featureTitle = str(featureTitle)
        value = str(value)
        
        # merge the two file to create one
        feature_user = pd.merge(feature_file, user_file, on=featureId)
        
        # creaate a long format table user/feature
        feature_user = pd.pivot_table(feature_user, index= userId, columns=featureTitle, values=value)
        
        # extract feature list
        feature_list = list(feature_user.columns)
        
        #user list
        user_list = list(feature_user.index)
        
        # fillna 
        m = feature_user.mean()
        feature_user_fill = feature_user.fillna(value=m)
        
        
        #construct NMF table
        
        factorizer = NMF(n_components =50, init ='random', max_iter = 1000)
        
        factorizer.fit(feature_user_fill)
        
        # q matrix 
        q_matrix = factorizer.components_
        q_matrix = pd.DataFrame(q_matrix, columns=feature_list)
        
        # Save model 
        with open('factorize_model.pkl', 'wb') as recom_model:
    	    pickle.dump(factorizer, recom_model)
        
        # Save feature_user, q_matrix in local
        q_matrix.to_csv('q_matrix.csv')
        feature_user.to_csv('user_movie.csv')
         
        
        return feature_user, feature_user_fill, q_matrix
        
    
    def optimal_component(self, matrix, n):
        """
        Estimate the best number of components to construct NMF
        """
        # Load data and preprocess
        X = matrix # movie_user_fill ## np.loadtxt('data.txt')

        # Set range of number of components to evaluate
        n_components_range = range(1, n)

        # Initialize list to store AIC and BIC values
        aics = []
        bics = []
        err_ = []
        timer = 0
        n=1000

        # Loop over number of components
        for n_components in n_components_range:
            start_time = time.time()
        
            # Fit NMF model on training data
            nmf = NMF(n_components=n_components, init= 'random', random_state=42)
            W_train = nmf.fit_transform(X)
            H_train = nmf.components_
        
            # Compute reconstruction error on validation data
            X_val_reconstructed = np.dot(W_train, H_train)
         
       
            # summe of elt wise reconstruction error
            err1_= nmf.reconstruction_err_
        
            #print (f"--{rec}---{err1_}")
        
        
            # compute log likehood
            log_likelihood = np.sum(X * np.log(np.maximum(X_val_reconstructed, np.finfo(float).eps) - X_val_reconstructed))
        
            # Compute AIC and BIC 
            k = n_components * (X.shape[1] + X.shape[0])
            aic = -2*log_likelihood + 2 * k
            bic = -2*log_likelihood + np.log(X.size) * k
        
        
            # Append AIC, BIC and reconstruction error values to list
            aics.append(aic)
            bics.append(bic)
            err_.append(err1_)
        
            end_time = time.time()
            duration = end_time - start_time
        
        print(f"Iteration number:------- {n_components} ---- finish in : {round(duration, 3)} seconds")        
        timer +=duration
        
        #update error for minimum 
        
        if err1_ < n:
            n = err1_
            # best component for minimum error
            b_comp=n_components
                  
        
        timer = timer/60
        print (f"Time total of program execution : {round(timer, 3)} min")
    
        return aics, bics, err_, n, b_comp

    def recommender_NMF (self, query, model, q_matrix):
    
        """"
        Filter and recommends the top k movies for any given query 
    
            Arg : 
                query : initial list of features as guide to base the 
                recommendation on
                model : binary model containing the fitted model
                q_matrix : user_features matrix from the NMF factorizer
                        it is obtained after fitting NMF
                      (factorizer.components_)
    
           Return list of k movie ids
        """
        recommendations = []
        dict_ ={}
    
    
        # assign random rating to the given films
    
        for st in query:
            dict_[st] = random.randint(1,5)
    
        # load model
    
        #with open ('model.pkl', 'rb') as file_:
        #    fit_model = pickle.load(file_)
    
        # create dataframe to store movie and rating
        col = list(q_matrix.columns)
    
        user_rate = pd.DataFrame(dict_, index=['new_user'], columns= col)
    
        # fill missing value 
        user_rate = user_rate.fillna(value=random.randint(0,3))
    
        # user P matrix 
        p = model.transform(user_rate)
    
        # store user P matrix in dataframe 
        p = pd.DataFrame(p, index = ['new_user'])
    
        # Reconstruct user_movie matrix (user P matrix  x q)
        R_hat = np.dot(p, q_matrix)
    
        # store reconstruct matrix in dataframe
        R_hat = pd.DataFrame(R_hat, index= ['new_user'], columns=list(q_matrix.columns))
    
        #Transpose R_hat matrx
        R_hat_tranp = R_hat.T.sort_values(by='new_user', ascending=False)
    
        # make recommendations
        for movie in list(R_hat_tranp.index):
            if movie not in list(dict_.keys()):
                recommendations.append(movie)
    
        return recommendations
    
    
    

    def recommander_cos_similarity(self, user, matrix_na, expect_rate):

        """ function to make recommendation based on the rating of a choosen user 
        present in the data base. It compare the rating of the choosen user to other
        user and propose selection base on the minnimum rating defined
    
            Arg :
                Matrix contenant database features as column and user as index
                user : choosen user to base the recommendation on 
                expected rate : minimum rating the recommendation should fulfill
            
            return list of recommendation
    
        """
    
    
        movie_user_mat = matrix_na
    
        # fillna if exist whith mean mean value of rating of each movie
        if movie_user_mat.isnull().values.any():
            m = movie_user_mat.mean()
            movie_user_fill = movie_user_mat.fillna(value=m)
    
        # Transpose fill matrix to have user as column
    
        movie_user_fill = movie_user_fill.T
    
        # compute similarity matrix 
        movie_user_sim = cosine_similarity(movie_user_fill.T)
    
        # store similarity matrix in a dataframe (index and column = user)
        movie_user_sim = pd.DataFrame(movie_user_sim, index=movie_user_fill.columns, columns=movie_user_fill.columns)
    
        # explore unseen movie of a user from database
        # look to the initial matrix before filling nan
        # Transpose to have user as columns of the matrix
        movie_user_mat = movie_user_mat.T
    
        # unseen movie of a given user
        unseen_movie = movie_user_mat[movie_user_mat[user].isna()].index
    
        # Search for top five user of the movie database
        t_five = movie_user_sim[user].sort_values(ascending=False).index[1:6]

        dict_= {}
        dict_2 = []
        # Make recommendation based on the top five user
        for movie in unseen_movie:
            # ~ negate the boolean mask, hier return movie that not na
            # Applies a boolean mask to the columns, selecting only columns 
            # where the value for a specific row (specified by the movie variable) is not null (i.e., not NaN).   
            other_users = movie_user_mat.columns[~movie_user_mat.loc[movie].isna()]
    
            #set is a collection of unique elements, meaning that it only 
            # contains one instance of each distinct element
            others_users = set(other_users)
    
            num=0
            den=0
            ratings=0
        
    
            for users in other_users.intersection(set(t_five)):
                # calculate intersection of two set
                # return only elt contain in both set (t_five and other_user)
                rating = movie_user_fill[user][movie]
                similarity = movie_user_sim[user][users]
                num = num + (rating*similarity)
                den = den + similarity + 0.0001 #to avoid zero division
        
                ratings = num/den
     
                # print recommendation of movie if ratings is higher than a value 
                if ratings > expect_rate:
                    print(movie, round(ratings, 2))
                
                    dict_= {'movie':movie, 'rating':round(ratings,1)}
                    dict_2.append(dict_)
    
        # Convert each dictionary to a tuple and add it to a set to remove duplicates
        unique_tuples = set(tuple(sorted(d.items())) for d in dict_2)
    
        # Convert each unique tuple back to a dictionary and add it to a list
        unique_dicts = [dict(t) for t in unique_tuples]
            
            
        return unique_dicts
        
        
        
                
    
    




        

        
    
    
    
    
    
    
    

    
    
    


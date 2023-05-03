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
from recommender_system import RecommenderSystem 


#read file 
movie = pd.read_csv("ml-latest-small/movies.csv")
rating = pd.read_csv("ml-latest-small/ratings.csv")

# initiate an instance of the RecommenderSystem

rs = RecommenderSystem()

# Perfom matrix transformation and save NMF model
# need to be run once then load everything from local --> next command
#user_movie, user_movie_fill, q_matrix = rs.recommender_eda(movie, rating, "movieId", "userId", "rating", "title")

# load database save locally

user_movie = pd.read_csv('user_movie.csv', index_col=0)
q_matrix = pd.read_csv('q_matrix.csv', index_col=0)

#print(q_matrix.head())

# load fitted model 

with open ('factorize_model.pkl', 'rb') as file_:
        fit_model = pickle.load(file_)
        
def get_movie():
    # query for NMF recommendation
    movie = list(user_movie.columns)
    query = random.sample(movie, 5)
    
    return query
    
query = get_movie()

#query =  ["'71 (2014)",
# "'Hellboy': The Seeds of Creation (2004)",
# "'Round Midnight (1986)",
# "'Salem's Lot (2004)"]
 
 
# NMF recommendation



def get_recommender(query):

    tr = rs.recommender_NMF(query,fit_model, q_matrix)
    # print(tr[:])
    
    return tr
    
    
get_recommender(query)

#for it in tr:
#    print(it)
#print (len(tr))

# cosine similarity recommendation 
#print('This the cosine similarty movies')
#rs.recommander_cos_similarity(40, user_movie, 4)






from flask import Flask, render_template, request
import random
from recommender_system import RecommenderSystem
from main import get_recommender, get_movie


app = Flask(__name__)

#@app.route('/')
#def hello_world():
#    return 'Hello, world'

#@app.route('URL') ----- bind a URL to a Python function that will handle requests made to that URL.

@app.route('/')
def index():
    
    ##define movie to prompt to user for rating
    ### should be from user_movie.columns
    select_query = get_movie()
    
    ## how to change movie, movie2 ... to name from select_query
    movie = {title : title for title in select_query}
    
    
    return render_template('index.html', title='Welcome to your Movie Recommender', 
                                         movie_dict = movie)

@app.route('/recommender')
def recommender():
    # get input data from user(form)
    user_input_data = dict(request.args) # same as request.args.dict()
    ### if rating > 3 adding movie to the query of the user
    ### then get recommender from query of the user
    query = []
    for key in user_input_data:
        value= user_input_data[key]
        if value >= 3:
            query.append(key)
    
    movie = get_recommender(query)
    
    #print(user_input_data)
    
    return render_template('recommendation.html',
                            movies = movie)


if __name__ =="__main__":
    app.run(debug=True, port=5003)

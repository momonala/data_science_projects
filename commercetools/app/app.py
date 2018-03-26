import pandas as pd
import numpy as np 
from flask import Flask, render_template, request

import build_df 
from recommender import Recommender

from urllib.parse import quote
import urllib.request

K = 25
TEST_SIZE = 0.20

app = Flask(__name__)

#################
#  Preprocess data and perfrom SVD. Keep all SVD data in a Recommender Object 
filtered_df = build_df.setup_simple()
recom_engine = Recommender(filtered_df, K, TEST_SIZE)
recom_engine.computeSVD()

#################
# Some helper functions for Flask 
def get_img_urls(search_term):
    '''  A hacky solution to grab all image url's from Google Image Search.
     We will only use the first one for displaying.   '''

    url = 'https://www.google.com/search?q=' \
        + quote(search_term) \
        + '&espv=2&biw=1366&bih=667&site=webhp&source=lnms&tbm=isch&sa=X&ei=XosDVaCXD8TasATItgE&ved=0CAcQ_AUoAg'

    try:
        headers = {}
        headers['User-Agent'] = "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36"
        req = urllib.request.Request(url, headers=headers)
        resp = urllib.request.urlopen(req)
        respData = str(resp.read()) #raw HTML
        imgs = [s for s in respData.split('"') if '.jpg' in s]
        return respData, imgs

    except Exception as e:
        print(str(e))


class Item: 
    '''Item object to unpack in Jinja (HTML)'''
    def __init__(self): 
        img_url = None 
        name = None 


#################
# The Web Routing 

@app.route('/')
def my_form():
    return render_template('form.html')


@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']
    
    # Generate recommendations via the SVD matrix, gather relevant Google Images,
    # pack them into an Item object, and render it to HTML via Jinja 
    try: 
        _id = int(text)
        bought_names, recom_names = recom_engine.show_recommendations(_id, num_recom=12)

        bought = []
        for name in bought_names: 
            _, imgs = get_img_urls(name)
            new_item = Item()
            new_item.name = name
            new_item.img_url = imgs[0]
            bought.append(new_item)

        recom = []
        for name in recom_names: 
            _, imgs = get_img_urls(name)
            new_item = Item()
            new_item.name = name
            new_item.img_url = imgs[0]
            recom.append(new_item)

        return render_template('recommendations.html', 
                                userid = _id,
                                items_bought = bought,
                                items_recom = recom, 
                                )

    # If the user inputs an invalid UserID 
    except ValueError:
        error_message  = '{} not a Valid User ID'.format(text)
        return render_template('form.html',  error_message=error_message )


if __name__ == '__main__': 
    app.run(debug=True)
from flask import Flask, render_template, request
import pandas as pd
import model 
app = Flask('__name__')

# Read the dataset sample30.csv
reviews_df = pd.read_csv("dataset\sample30.csv")

# Create a list of users
VALID_USERID = reviews_df['reviews_username'].to_list()

@app.route('/')
def view():
    return render_template('index.html')

@app.route('/recommend',methods=['POST'])
def recommend_top5():
    user_name = request.form['User Name']
    
    if  user_name in VALID_USERID and request.method == 'POST':
        get_top5 = model.get_top5_sentiment_recommended_products(user_name)
        return render_template('index.html',column_names=get_top5.columns.values, row_data=list(get_top5.values.tolist()), zip=zip,text='Recommended products')
    elif not user_name in  VALID_USERID:
        return render_template('index.html',text='No Recommendation found for the user')
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.debug=False

    app.run()
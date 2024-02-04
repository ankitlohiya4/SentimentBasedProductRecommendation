# Import required libraries
import re
import pandas as pd
import numpy as np
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet

import warnings
warnings.filterwarnings("ignore") 


# Method to load pickle files
def load_pickle(pickleFile):
    with open(r"./pickle_file/" + pickleFile, "rb") as f:
        return pickle.load(f)


# Map pos to wordnet tag
def pos_tag_to_wordnet_tag(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:          
        return None


#  Method for initial data cleaning using regex pattern matching and lematizing
def get_clean_tokens(corpus):   
    
    # Load stopwords
    stop_words = set(stopwords.words('english'))

    # Regex pattern for all special characters
    special_character = re.compile(r"[`!@#$%^&*()+;:'\"/?.,<=>~^_\-\[\]\{\}\\\|]+")

    # Regex patter for non ASCII code
    non_ascii = re.compile("[^\x00-\x7f]")
    
    for i in range(len(corpus)):

        processPercent = round(((i/len(corpus)) * 100), 2)
        print("\rProcessing Status -> "+str(processPercent)+"%", end =' ', flush=True)

        # Convert text to lower case and remove all special characters
        corpus[i] = corpus[i].lower()        
        corpus[i] = re.sub(pattern=special_character, repl=' ', string=corpus[i])
    
        # Tokenize
        tokens = corpus[i].split()
        
        # POS tagging
        tokens = nltk.pos_tag(tokens)

        # Instantiate WordNet Lemmatizer
        lemmatizer = WordNetLemmatizer()

        lemma = []
        for t in tokens:  
            # Remove tokens of length <= 2 and stopwords and with non ASCII characters
            if(len(t[0])>2 and t[0] not in stop_words and not re.search(non_ascii,t[0])):   
                # Map pos tag to wordnet tag for each tokens  
                tag = pos_tag_to_wordnet_tag(t[1])
                # Remove all tokens which do not have wordnet tag
                if(tag != None):
                    # Lemmatize each tokens
                    lemma.append(lemmatizer.lemmatize(word=t[0], pos=tag))

        tokens = lemma
        corpus[i] = ' '.join(tokens)

    processPercent = round((((i+1)/len(corpus)) * 100), 2)
    print("\rProcessing Status -> "+str(processPercent)+"%", end =' ', flush=True)
    return corpus


# Method to get recommended products for a given user
def get_recommended_products(user):
    user_final_rating = load_pickle("user_final_rating.pk")
    user_recommendations  = user_final_rating.loc[user].sort_values(ascending=False)[0:20]
    user_recommendations_df = pd.DataFrame({'product_id': user_recommendations.index, 'similarity_score' : user_recommendations})
    user_recommendations_df.reset_index(drop=True, inplace=True)
    return user_recommendations_df


# Method to get sentiments for recommended products
def get_sentiments_for_recommended_products(recommended_products_df):
    # Load the dataset
    reviews_df = pd.read_csv("dataset\sample30.csv")

    # Filter recommended products
    recommended_reviews_df = reviews_df[reviews_df["id"].isin(recommended_products_df["product_id"].to_list())]
    recommended_reviews_df["reviews_title"] = recommended_reviews_df["reviews_title"].fillna('')

    # Combine reviews_text and reviews_title
    recommended_reviews_df["reviews_full_text"] = recommended_reviews_df[['reviews_title', 'reviews_text']].agg('. '.join, axis=1)

    # Pre-process text to get clean Reviews for recommended products
    corpus = list(recommended_reviews_df['reviews_full_text'])
    corpus = get_clean_tokens(corpus)
    recommended_reviews_df['reviews_clean_full_text'] = corpus

    # Create X from the dataset
    X = recommended_reviews_df['reviews_clean_full_text']

    # Load TF-IDF vectorizer
    vectorizer = load_pickle("tfidf_vectorizer.pk")

    # Transform X to vectors
    X = vectorizer.transform(X)
    print("X.shape = ",X.shape)

    # Load the Sentiment classification XG boost model
    xgb = load_pickle("sentiment_classification_xg_boost_model.pk")

    # Predict the sentiment
    recommended_reviews_df["predicted_sentiment"]= xgb.predict(X)

    # Filter id and predicted_sentiment
    temp = recommended_reviews_df[['id','predicted_sentiment']]

    # Group by id and compute Total Positive Review Count and Total Review Count
    temp_grouped = temp.groupby('id', as_index=False).count()
    temp_grouped["pos_review_count"] = temp_grouped.id.apply(lambda x: temp[(temp.id==x) & (temp.predicted_sentiment==1)]["predicted_sentiment"].count())
    temp_grouped["total_review_count"] = temp_grouped['predicted_sentiment']

    # Determine Positive Percentage and sort in Descending order
    temp_grouped['pos_sentiment_percent'] = np.round(temp_grouped["pos_review_count"]/temp_grouped["total_review_count"]*100,2)
    temp_grouped.sort_values('pos_sentiment_percent', ascending=False, inplace=True)
    return pd.merge(recommended_reviews_df, temp_grouped, on="id")[["name", "brand", "manufacturer", "pos_sentiment_percent"]].drop_duplicates().sort_values(['pos_sentiment_percent', 'name'], ascending=[False, True])


# Method to get top 5 recommended products based on sentiments
def get_top5_sentiment_recommended_products(user):
    recommended_products_df = get_recommended_products(user)
    sentiment_recommended_products_df = get_sentiments_for_recommended_products(recommended_products_df)
    return sentiment_recommended_products_df[:5]
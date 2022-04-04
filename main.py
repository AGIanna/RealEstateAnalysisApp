from models import naiveBayes as nb
from models import KNN, SVM
from models import decisionTree as dt

import pandas as pd

def get_integrated_dataset():
    # get final dataset
    final = pd.read_csv("data/realestatedata_final.csv")
    final['Date'] = pd.to_datetime(final[['Year', 'Month']].assign(DAY=1))

    # Insert news data
    # News without sentiment
    news_top = pd.read_csv("data/month_without_sentiment.csv")
    news_top["Date"] = pd.to_datetime(news_top["Date"]) 

    # final_news_top = final.merge(news_top, left_on='Date', right_on = 'Date')

    # News with sentiment
    # news_sen = pd.read_csv("data/month_with_sentiment.csv")
    # news_sen["Date"] = pd.to_datetime(news_sen["Date"])  

    # Merge news into final
    # df_sen = final.merge(news_sen, left_on='Date', right_on = 'Date') # merge news with sentiment
    df_top = final.merge(news_top, left_on='Date', right_on='Date') # merge news without sentiment
    return df_top
    
if __name__ == "__main__":
    df = get_integrated_dataset()
    features = ['ZipCode', 'Unemp Rate']
    
    knn_m, knn_pred, knn_conf, knn_score = KNN.KNN(df, features)
    print(knn_m)
    print(knn_conf)
    print(knn_score)
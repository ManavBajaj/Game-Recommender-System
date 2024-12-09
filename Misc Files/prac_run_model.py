import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import mysql.connector
import os

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import *

warnings.filterwarnings("ignore")

os.environ['KAGGLE_USERNAME'] = 'jayeshrmohanani'
os.environ['KAGGLE_KEY'] = '030582329579eb59561c5aeb1fd5f65e'

video_games_df = pd.read_csv("C:\\Users\jayes\OneDrive\Desktop\College\MSc\CHRIST Data Science\T2\Fullstack Web dev\CAC 2 - Project\games-dataset\Video_Games_Sales_as_at_22_Dec_2016.csv")

print(f"No. of records: {video_games_df.shape[0]}")
video_games_df.head(5)

video_games_filtered_df = video_games_df[['Game_Name', 'Platform', 'Genre', 'Critic_Score', 'User_Score', 'Rating']]
video_games_filtered_df.info()

video_games_filtered_df.isna().sum().sort_values(ascending=False)

# Remove missing values
video_games_filtered_df.dropna(subset=['Game_Name', 'Genre', 'Rating'], axis=0, inplace=True)
video_games_filtered_df = video_games_filtered_df.reset_index(drop=True)

video_games_filtered_df[['Game_Name', 'Genre', 'Rating']].isna().sum()

features = video_games_filtered_df[['Genre', 'Platform', 'Rating']].columns

for idx, feature in enumerate(features):
    plt.figure(figsize = (13,4))
    sns.countplot(data=video_games_filtered_df, x=feature)
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.title(" Data Distribution of Video Game " + feature + "s")
plt.show()

# Replace 'tbd' value to NaN
video_games_filtered_df['User_Score'] = np.where(video_games_filtered_df['User_Score'] == 'tbd', 
                                                 np.nan, 
                                                 video_games_filtered_df['User_Score']).astype(float)

# Group the records by Genre, then aggregate them calculating the average of both Critic Score and User Score
video_game_grpby_genre = video_games_filtered_df[['Genre', 'Critic_Score', 'User_Score']].groupby('Genre', as_index=False)
video_game_score_mean = video_game_grpby_genre.agg(Ave_Critic_Score = ('Critic_Score', 'mean'), Ave_User_Score = ('User_Score', 'mean'))

# Merge the average scores with the main dataframe
video_games_filtered_df = video_games_filtered_df.merge(video_game_score_mean, on='Genre')
video_games_filtered_df

video_games_filtered_df['Critic_Score_Imputed'] = np.where(video_games_filtered_df['Critic_Score'].isna(), video_games_filtered_df['Ave_Critic_Score'], video_games_filtered_df['Critic_Score'])

video_games_filtered_df['User_Score_Imputed'] = np.where(video_games_filtered_df['User_Score'].isna(), video_games_filtered_df['Ave_User_Score'], video_games_filtered_df['User_Score'])
video_games_filtered_df

video_games_filtered_df[['Critic_Score', 'Critic_Score_Imputed', 'User_Score', 'User_Score_Imputed']].describe()

video_games_final_df = video_games_filtered_df.drop(columns=['User_Score', 'Critic_Score', 'Ave_Critic_Score', 'Ave_User_Score'], axis=1)
video_games_final_df = video_games_final_df.reset_index(drop=True)

video_games_final_df = video_games_final_df.rename(columns={'Critic_Score_Imputed':'Critic_Score', 'User_Score_Imputed':'User_Score'})
video_games_final_df.info()

hist, bins = np.histogram(video_games_final_df['Critic_Score'], bins=10, range=(0, 100))

plt.figure(figsize = (8,4))
plt.bar(bins[:-1], hist, width=(bins[1]-bins[0]), align='edge')
plt.xlabel('Critic Score')
plt.ylabel('Frequency')
plt.title("Data Distribution of Critic Scores")
plt.show()

hist, bins = np.histogram(video_games_final_df['User_Score'], bins=10, range=(0, 10))

plt.figure(figsize = (8,4))
plt.bar(bins[:-1], hist, width=(bins[1]-bins[0]), align='edge')
plt.xlabel('User Score')
plt.ylabel('Frequency')
plt.title("Data Distribution of User Scores")
plt.show()

plt.figure(figsize=(8, 8))
ax = sns.regplot(x=video_games_final_df['User_Score'], y=video_games_final_df['Critic_Score'], line_kws={"color": "black"}, scatter_kws={'s': 4})
ax.set(xlabel ="User Score", ylabel = "Critic Score", title="User Scores vs. Critic Scores")

video_games_final_df.info()

categorical_columns = [name for name in video_games_final_df.columns if video_games_final_df[name].dtype=='O']
categorical_columns = categorical_columns[1:]

print(f'There are {len(categorical_columns)} categorical features:\n')
print(", ".join(categorical_columns))

video_games_df_dummy = pd.get_dummies(data=video_games_final_df, columns=categorical_columns)
video_games_df_dummy.head(5)

video_games_df_dummy.info()

features = video_games_df_dummy.drop(columns=['Game_Name'], axis=1)

scale = StandardScaler()
scaled_features = scale.fit_transform(features)
scaled_features = pd.DataFrame(scaled_features, columns=features.columns)

scaled_features.head(5)

model = NearestNeighbors(n_neighbors=11, metric='cosine', algorithm='brute').fit(scaled_features)
print(model)

vg_distances, vg_indices = model.kneighbors(scaled_features)

print("List of indexes and distances for the first 5 games:\n")
print(vg_indices[:5], "\n")
print(vg_distances[:5])

game_names = video_games_df_dummy['Game_Name'].drop_duplicates()
game_names = game_names.reset_index(drop=True)

vectorizer = TfidfVectorizer(use_idf=True).fit(game_names)
print(vectorizer)

game_title_vectors = vectorizer.transform(game_names)

print("List of game title vectors for the first 5 games:\n")
print(pd.DataFrame(game_title_vectors.toarray()).head(5))

def VideoGameRecommender(game_name):
    """
    Recommend games based on the input game name.
    
    Parameters:
    - game_name (str): Name of the game entered by the user.
    
    Returns:
    - List of recommended game titles.
    """
    if game_name in video_games_df['Game_Name'].values:
        genre = video_games_df.loc[video_games_df['Game_Name'] == game_name, 'Genre'].values[0]
        recommendations = video_games_df[video_games_df['Genre'] == genre] \
                              .sort_values(by='User_Score', ascending=False) \
                              .head(5)['Game_Name'].tolist()
        if recommendations:
            return recommendations
        else:
            return ["No recommendations available for this genre."]
    else:
        print(f"'{game_name}' not found in the database. Here are some random suggestions:")
        return video_games_df.groupby('Genre').apply(lambda x: x.sample(1))['Game_Name'].sample(5).tolist()
    

def UserInput():
    """
    Function to take at least 3 valid game inputs from the user 
    and pass them to VideoGameRecommender.
    """
    user_games = []
    
    # Collect 3 valid games
    while len(user_games) < 3:
        user_input = input(f"Enter game {len(user_games) + 1} (at least 3 required): ").strip()
        
        if not user_input:
            print("Input cannot be empty. Try again.")
            continue

        # Validate the input game name immediately
        if user_input not in video_games_df['Game_Name'].values:
            print(f"'{user_input}' not found. Try another game.")
            continue
        
        user_games.append(user_input)

    all_recommendations = set()  # To avoid duplicates
    
    # Collect recommendations for each game and add to the set
    for game in user_games:
        print(f"\nGetting recommendations for '{game}':")
        recommendations = VideoGameRecommender(game)
        
        # Add recommendations to the set to avoid duplicates
        all_recommendations.update(recommendations)

    # Limit to 5 distinct recommendations
    final_recommendations = list(all_recommendations)[:5]

    print("\nFinal Recommendations based on your inputs:")
    for rec in final_recommendations:
        print(f"- {rec}")
    
    return user_games

UserInput()
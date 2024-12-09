import pandas as pd
import numpy as np
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

# video_games_df = pd.read_csv("games-dataset/Video_Games_Sales_as_at_22_Dec_2016.csv")
video_games_df = pd.read_csv("C:\\Users\\manav\\OneDrive\\Desktop\\Projects\\Game Recommender System\\games-dataset\\Video_Games_Sales_as_at_22_Dec_2016.csv")

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
    """
    if game_name in video_games_final_df['Game_Name'].values:
        genre = video_games_final_df.loc[video_games_final_df['Game_Name'] == game_name, 'Genre'].values[0]

        # Get genre-based recommendations
        genre_recommendations = video_games_final_df[video_games_final_df['Genre'] == genre] \
                                .sort_values(by='User_Score', ascending=False)

        # Title-based similarity using partial matches
        title_recommendations = video_games_final_df[
            video_games_final_df['Game_Name'].str.contains(game_name.split()[0], case=False, na=False)
        ]

        # Combine title-based and genre-based recommendations
        combined_recommendations = pd.concat([title_recommendations, genre_recommendations]) \
                                     .drop_duplicates(subset='Game_Name') \
                                     .head(5)[['Game_Name', 'Genre']].to_dict(orient='records')

        print(f"Recommendations for {game_name}: {combined_recommendations}")  # Debugging line

        return combined_recommendations if combined_recommendations else [{"Game_Name": "No recommendations available.", "Genre": ""}]
    else:
        print(f"'{game_name}' not found in the database. Returning random suggestions.")  # Debugging line
        return video_games_final_df.groupby('Genre').apply(lambda x: x.sample(1)) \
                   [['Game_Name', 'Genre']].sample(5).to_dict(orient='records')

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
        if user_input not in video_games_final_df['Game_Name'].values:
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

app = Flask(__name__)

@app.route("/")
def homepage():
    return render_template("home.html")

@app.route("/recommendation", methods=["GET"])
def recommend():
    # Send distinct game names for dropdowns
    game_names = sorted(video_games_final_df['Game_Name'].unique())
    return render_template("recommendation.html", game_names=game_names, recommendations=None)

@app.route("/saveitem", methods=["GET"])
def save():
    # Retrieve selected games from the form
    selected_games = [request.args.get('game1'), request.args.get('game2'), request.args.get('game3')]

    # Validate that all games are selected
    if all(selected_games):
        # Get unique details of the selected games for the "Your Selection" table
        selected_games_details = video_games_final_df[
            video_games_final_df['Game_Name'].isin(selected_games)
        ][['Game_Name', 'Genre']].drop_duplicates(subset='Game_Name').to_dict(orient='records')

        # Generate recommendations
        all_recommendations = set()
        for game in selected_games:
            recommendations = VideoGameRecommender(game)
            for rec in recommendations:
                all_recommendations.add((rec['Game_Name'], rec['Genre']))

        # Exclude the selected games from recommendations
        all_recommendations = {rec for rec in all_recommendations if rec[0] not in selected_games}

        # Convert recommendations to the required format (list of dictionaries)
        final_recommendations = [{'Game_Name': name, 'Genre': genre} for name, genre in all_recommendations][:5]

        # Get distinct game names for the dropdowns
        game_names = sorted(video_games_final_df['Game_Name'].drop_duplicates())

        # Render the template with all data, retaining dropdown selections
        return render_template(
            "recommendation.html",
            game_names=game_names,
            selected_games_details=selected_games_details,  # Pass distinct selections
            recommendations=final_recommendations,
            selected_games=selected_games  # Retain dropdown selections
        )
    else:
        # If not all games are selected, return an error
        game_names = sorted(video_games_final_df['Game_Name'].drop_duplicates())
        return render_template(
            "recommendation.html",
            game_names=game_names,
            recommendations=[],
            error="Please select 3 games.",
            selected_games=selected_games,  # Retain dropdown selections even on error
            selected_games_details=None
        )

@app.route("/bioshock")
def bio():
    return render_template("bioshock.html")

@app.route("/call-of-duty-black-ops-6")
def cod():
    return render_template("codblackops.html")

@app.route("/cyberpunk")
def cyb():
    return render_template("cyberpunk.html")

@app.route("/dishonoured-2")
def dis():
    return render_template("dishonored.html")

@app.route("/far-cry-3")
def fc3():
    return render_template("fc3.html")

@app.route("/fc-25")
def fifa():
    return render_template("fifa25.html")

@app.route("/forza-horizon-5")
def forza():
    return render_template("forza.html")

@app.route("/ghost-of-tsushima")
def got():
    return render_template("got.html")

@app.route("/god-of-war")
def gow():
    return render_template("gow.html")

@app.route("/grand-theft-auto-5")
def gta5():
    return render_template("gta5.html")

@app.route("/it-takes-two")
def itt():
    return render_template("it_takes_two.html")

@app.route("/last-of-us-part-II")
def los():
    return render_template("los.html")

@app.route("/portal-2")
def por2():
    return render_template("portal2.html")

@app.route("/red-dead-redemption-2")
def rdd():
    return render_template("rdd.html")

@app.route("/resident-evil-4")
def re4():
    return render_template("resident.html")

@app.route("/sekiro-shadows-die-twice")
def sek():
    return render_template("sekiro.html")

@app.route("/marvel's-spider-man-2")
def spm2():
    return render_template("spiderman.html")

@app.route("/the-walking-dead")
def twd():
    return render_template("TWD.html")

@app.route("/uncharted-4-a-theif's-end")
def uc4():
    return render_template("uncharted4.html")

@app.route("/witcher-3-wild-hunt")
def witcher3():
    return render_template("witcher.html")

@app.route("/black-myth-wukong")
def wuk():
    return render_template("wukong.html")

# MySQL database configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Johnwick@5485',
    'database': 'FullStackProject'
}

@app.route('/play2gether')
def play2gether():
    try:
        # Connect to the database
        dbconn = mysql.connector.connect(**db_config)
        cursor = dbconn.cursor()

        # Query the database
        cursor.execute("SELECT id, name, description, poster FROM games")
        game = cursor.fetchall()
        print(game)

        # Close the connection
        cursor.close()
        dbconn.close()

        # Pass the data to the template
        return render_template('play2gether.html', game=game)
    except mysql.connector.Error as e:
        return f"Database error occurred: {str(e)}"
    except Exception as e:
        return f"An error occurred: {str(e)}"

@app.route('/<name>')
def game_list(name):
    try:
        # Connect to the database
        dbconn = mysql.connector.connect(**db_config)
        cursor = dbconn.cursor()

        # Query the database
        cursor.execute("SELECT * FROM games WHERE name = %s", (name,))
        games = cursor.fetchone()
        print(games)

        # Close the connection
        cursor.close()
        dbconn.close()

        # Pass the data to the template
        if games:
            return render_template('play2gethergames.html', games=games, platform=games[5].split(','))
        else:
            return "Game not found."
    except mysql.connector.Error as e:
        return f"Database error occurred: {str(e)}"
    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
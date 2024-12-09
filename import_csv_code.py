import pandas as pd
import mysql.connector

# Database connection details
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Johnwick@5485',
    'database': 'fullstackproject'
}

# Path to your CSV file
csv_file = "C:\\Users\\manav\\OneDrive\\Desktop\\Projects\\Game Recommender System\\games-dataset\\Video_Games_Sales_as_at_22_Dec_2016.csv"

# Load CSV into a DataFrame
df = pd.read_csv(csv_file)

# Connect to MySQL database
connection = mysql.connector.connect(**db_config)
cursor = connection.cursor()

# Insert data into the table
for _, row in df.iterrows():
    insert_query = """
    INSERT INTO video_games (Game_Name, Platform, Release_Year, Genre, Publisher, NA_Sales, EU_Sales, JP_Sales, Other_Sales, Global_Sales, Critic_Score, Critic_Count, User_Score, User_Count, Developer, Rating)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
    """
    cursor.execute(insert_query, tuple(row))

# Commit the transaction
connection.commit()

# Close the connection
cursor.close()
connection.close()

print("Data inserted successfully!")
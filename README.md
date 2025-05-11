# PlayNext   
**Find Your Next Favorite Game**

**PlayNext** is a dynamic full-stack game recommendation system designed to help gamers discover new titles based on their preferences. With a robust backend powered by Flask and SQL, and a stylish frontend built using HTML, CSS, and JavaScript, PlayNext creates a seamless user experience across its landing, recommendation, and multiplayer explorer pages.

##  Key Features

-  **Personalized Game Recommendations**  
  Suggests games based on a user’s top 3 favorite titles using genre and rating similarities.

-  **Dynamic Dropdown System**  
  Dropdowns in the recommendation page are populated directly from the SQL database for real-time updates.

-  **Content-Based Filtering**  
  Matches games using genre, ratings, and attributes for targeted suggestions.

-  **Play2gether: Multiplayer Game Explorer**  
  A curated, database-driven page listing top multiplayer games with visuals and game links.

-  **Interactive UI**  
  - Carousel of top-rated and trending games  
  - Hover effects and particle animations  
  - Game cards with detail buttons and external store redirects

##  Tech Stack

- **Frontend:**  
  HTML, CSS, JavaScript (with animations & hover effects)

- **Backend:**  
  Python Flask, Jinja2 Templates

- **Database:**  
  MySQL (Games table with attributes: name, genre, rating, platforms, poster, description)

- **Deployment:**  
  Local Flask server (can be deployed to cloud platforms like Heroku or Render)


##  How It Works

1. **Landing Page:**  
   Static entry point with a game carousel and category-based navigation (Top Rated, Popular, Trending).

2. **Game Details Pages:**  
   Dynamic routes display detailed descriptions and images for selected games.

3. **Recommendation Page:**  
   - Users select 3 favorite games  
   - Flask backend retrieves related games from SQL database  
   - Games displayed in a dynamic table (title, genre)

4. **Play2gether Page:**  
   - Fetches multiplayer games from DB  
   - Displays with game card layout and interactive navigation



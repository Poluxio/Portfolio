import streamlit as st
from PIL import Image
import requests
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from sklearn.preprocessing import RobustScaler

from streamlit import set_page_config

set_page_config(page_title="Movies Madness !", layout="wide")


@st.cache_data
def read_data():
    # Modify this line to specify the full path to the df_ML.csv file on your Windows computer
    df_ML = pd.read_csv('./df_ML.csv', sep=',')
    scaled_columns = pd.read_csv('./scaled_columns', sep=',')
    return df_ML, scaled_columns

df_ML, scaled_columns = read_data()


@st.cache_data
def fetch_poster(movie_title):
    # Replace YOUR_API_KEY with your TMDB API key
    api_key = "8265bd1679663a7ea12ac168da84d2e8"
    query = movie_title.replace(' ', '+')
    response = requests.get(f'https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={query}')
    data = response.json()
    try:
        poster_path = data['results'][0]['poster_path']
        return f"https://image.tmdb.org/t/p/w500/{poster_path}"
    except:
        return None

def reco_film_TEST(movies, names, genres):
    names_list = []
    for person in names:
        names_list.extend(df_ML.loc[df_ML['cast'].str.contains(person, na=False), 'title'].tolist())
        
    genres_list= []
    for genre in genres:
        genres_list.extend(df_ML.loc[df_ML['genres'].str.contains(genre, na=False), 'title'].tolist())

    L_movies = len(movies)    
    L_names_list = len(names_list)
    L_genres_list = len(genres_list)

    X_scaled = scaled_columns.drop(columns=['releaseDate', 'runtimeMinutes', 'averageRating', 'numVotes', 'WR'])

    model = NearestNeighbors(n_neighbors=50, metric='cosine', algorithm='auto', n_jobs=-1).fit(X_scaled)

    if movies and not df_ML.loc[df_ML['title'] == movies[0]].index.empty:
        index_movies = df_ML.loc[df_ML['title'] == movies[0]].index[0]
        X_sumMovies = X_scaled.loc[X_scaled.index == index_movies].sum(axis=0)
    else:
        X_sumMovies = 0

    if names_list and not df_ML.loc[df_ML['title'] == names_list[0]].index.empty:
        index_names = df_ML.loc[df_ML['title'] == names_list[0]].index[0]
        X_sumNames = X_scaled.loc[X_scaled.index == index_names].sum(axis=0)
    else:
        X_sumNames = 0

    if genres_list and not df_ML.loc[df_ML['title'] == genres_list[0]].index.empty:
        index_genres = df_ML.loc[df_ML['title'] == genres_list[0]].index[0]
        X_sumGenres = X_scaled.loc[X_scaled.index == index_genres].sum(axis=0)
    else:
        X_sumGenres = 0

    for movie in movies[1:]:
        index_movie = df_ML.loc[df_ML['title'] == movie].index[0]
        X_sumMovies += X_scaled.loc[X_scaled.index == index_movie].sum(axis=0)    
        
    for movie in names_list[1:]:
        index_names = df_ML.loc[df_ML['title'] == movie].index[0]
        X_sumNames += X_scaled.loc[X_scaled.index == index_names].sum(axis=0)
    
    for movie in genres_list[1:]:
        index_genres = df_ML.loc[df_ML['title'] == movie].index[0]
        X_sumGenres += X_scaled.loc[X_scaled.index == index_genres].sum(axis=0)    


    if L_movies != 0:
        X_sumMovies /= L_movies
    if L_names_list != 0:
        X_sumNames /= L_names_list
    if L_genres_list != 0:
        X_sumGenres /= L_genres_list
    
    X_final = (X_sumMovies*0.5 + X_sumNames + X_sumGenres)


    recommandation = model.kneighbors([X_final])[1][0]
    result = df_ML.iloc[recommandation][['title', 'genres', 'releaseDate', 'numVotes', 'averageRating', 'actor', 'actress', 'director']]
    result = result.loc[result['title'].isin(movies) == False]

    return result.head(20)

st.markdown("""
    <style>
        .stApp {
        background: url("https://static.wixstatic.com/media/49106a_cdcfbf47cec046f68ba90fed9de4b4c2~mv2.jpg");
        background-size: cover;
        }
    </style>""", unsafe_allow_html=True)





st.title("Des films qui vous vont bien !")
st.write("60% du temps, fonctionne à chaque fois.")


# Convert the 'cast' column to a list and remove any 'nan' values
names_list = df_ML['cast'].dropna().tolist()

# Split the names in the list and remove duplicates by converting to a set and back to a list
names_list = list(set([name for sublist in names_list for name in str(sublist).split(',')]))


# User input for movies, names and genres
movies = st.multiselect("Select Movies", df_ML['title'].unique(), key="movies")
names = st.multiselect("Select Names", names_list, key="names")
genres = st.multiselect("Select Genres", ["Action", "Adventure", "Animation", "Biography", "Comedy", "Crime", "Documentary", "Drama", "Family", "Fantasy", "Film-Noir", "History", "Horror", "Music", "Musical", "Mystery", "News", "Romance", "Sci-Fi", "Sport", "Thriller", "War", "Western"], key="genres")



# Button to run the reco_film_TEST function and generate recommendations
if st.button("Trouve les films qui correspondent le mieux !"):
    # Display a loading message while the recommendations are being generated
    with st.spinner("Recherche en cours..."):
        # Generate recommendations using the reco_film_TEST function
        recommendations = reco_film_TEST(movies, names, genres)



# Check if the recommendations variable is defined
if 'recommendations' in locals():
    # Display the recommendations as rows
    st.subheader("Top 10 Recommendations")
    for index, row in recommendations.iterrows():
        # Display title in large font
        st.markdown(f"## {row['title']}")
        
        # Create columns for image, text, and review
        col1, col2, col3 = st.columns([1, 2, 2])

        # Fetch and display movie poster from TMDB in left column
        poster_url = fetch_poster(row['title'])
        if poster_url:
            image = Image.open(requests.get(poster_url, stream=True).raw)
            col1.image(image, width=300)

        
        # Display text in middle column
        col2.write(f"#### Genres: {row['genres']}")
        col2.write(f"#### Release Date: {row['releaseDate']}")
        col2.write(f"#### Number of Votes: {row['numVotes']}")
        col2.write(f"#### Average Rating: {row['averageRating']}")

        # Replace 'nan' values with an empty space
        actor = row['actor'] if pd.notna(row['actor']) else 'No result'
        actress = row['actress'] if pd.notna(row['actress']) else 'No result'
        director = row['director'] if pd.notna(row['director']) else 'No result'
        
        col2.write(f"#### Actor: {actor}")
        col2.write(f"#### Actress: {actress}")
        col2.write(f"#### Director: {director}")


# Read the data
df_spotify = read_data()

# Create a selectbox to choose an artist
artist = st.selectbox('Select an artist:', df_spotify['track_artist'].unique())

# Run the reco_artistes function and get the recommendations
recommendations = reco_artistes(df_spotify, artist)

# Display the recommendations
st.write('Recommended artists:')
st.write(recommendations)

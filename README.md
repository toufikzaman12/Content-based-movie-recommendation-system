# Content-based-movie-recommendation-system  
Movie Recommendation System part 1 


## Overview

A content-based movie recommendation system that suggests movies similar to a given title based on genres and tags.
## Features

- **Content-Based Filtering**: Recommends movies by analyzing genres and tags.
- **No User Data Required**: Works solely on the movie's content.
- **Scalable**: Easily handles new movie additions.

## Dataset

- **`movies.csv`**: Contains movie titles and genres.
- **`tags.csv`**: Contains user-generated tags for movies.


## How It Works

1. **Data Preparation:**
   - **Load Data:** Import movie and tag data from CSV files.
   - **Merge Data:** Combine movie data with tags to create a unified dataset.

2. **Feature Extraction:**
   - **Combine Attributes:** Merge movie genres and tags into a single text column for each movie.

3. **Text Vectorization:**
   - **TF-IDF Vectorization:** Convert text data into numerical form using Term Frequency-Inverse Document Frequency (TF-IDF). This represents the importance of words in the context of the entire dataset.

4. **Similarity Calculation:**
   - **Compute Cosine Similarity:** Measure how similar each movie is to every other movie based on their TF-IDF representations.

5. **Recommendation Generation:**
   - **Find Similar Movies:** For a given movie, retrieve the top similar movies based on similarity scores.

## Code Example

Here’s a simplified version of the code:

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
movies = pd.read_csv('movies.csv')
tags = pd.read_csv('tags.csv')

# Merge tags with movies based on movieId
movies = movies.merge(tags.groupby('movieId')['tag']
                      .apply(lambda x: ' '.join(x)).reset_index(), on='movieId', how='left')

# Replace missing tags with an empty string
movies['tag'] = movies['tag'].fillna('')

# Combine genres and tags into a 'content' column
movies['content'] = movies['genres'] + ' ' + movies['tag']

# Create TF-IDF matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['content'])

# Compute cosine similarity between movies
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Map movie titles to indices
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# Recommendation function
def get_content_based_recommendations(title, cosine_sim=cosine_sim):
    if title not in indices:
        return "Movie not found in the dataset."
    
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    movie_indices = [i[0] for i in sim_scores[1:11]]
    return movies['title'].iloc[movie_indices]
print(get_content_based_recommendations('Die Hard (1988)'))  # Example 1
print(get_content_based_recommendations('Toy Story (1995)'))  # Example 2
```

## Advantages

- **Personalized Recommendations**: Provides suggestions based on specific movie attributes, making them relevant to the user’s interests.
- **No User Data Needed**: Operates without needing user ratings or history.
- **New Items**: Can recommend new movies as soon as they’re added to the dataset.

## Disadvantages

- **Lack of Diversity**: Recommendations may be too similar to the input item, potentially missing out on different genres or styles.
- **Content Quality**: The system's effectiveness depends on the accuracy and richness of the movie descriptions (genres and tags).
- **Cold Start Problem**: Struggles with new or niche items that have limited or generic content.
- **Scalability**: Becomes slower as the number of movies increases due to the complexity of similarity calculations.

## License

This project is licensed under the NIT Sikkim License. For more details, see the [LICENSE](LICENSE) file.

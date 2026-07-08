# Movie Recommender System

A content-based movie recommendation system built using **TF-IDF** and **cosine similarity**. Given a movie title, it recommends the 5 most similar movies based on their text metadata (overview, genres, and keywords).

## How It Works

The system does not "understand" movies — it compares **text**. Each movie's overview, genres, and keywords are combined into a single "tags" string. These tags are converted into numerical vectors using TF-IDF, and cosine similarity is used to measure how closely any two movies match. Movies with the most overlapping words are recommended.

**Pipeline:**

1. **Data cleaning** — Keep only `title`, `overview`, `genres`, `keywords`; drop rows with missing values.
2. **Parsing** — `genres` and `keywords` are stored as stringified lists of dictionaries. These are parsed with `ast.literal_eval` to extract clean name lists.
3. **Feature engineering** — Spaces inside multi-word terms are removed (e.g. `"Science Fiction"` → `"ScienceFiction"`) so TF-IDF treats them as single tokens. All text is combined into one lowercase `tags` column.
4. **Vectorization** — `TfidfVectorizer` (top 5000 words, English stop words removed) converts tags into vectors.
5. **Similarity** — Cosine similarity produces a movie-to-movie similarity matrix.
6. **Recommendation** — For a given movie, the function returns the top 5 movies with the highest similarity scores.

## Why TF-IDF?

TF-IDF was chosen over simple Bag-of-Words because it down-weights words that appear across many movies (common, low-signal words) and gives more importance to distinctive words unique to a movie. This makes the similarity scores more meaningful.

## Dataset

TMDB 5000 Movie Dataset (`movies.csv`) — approximately 4800 movies with metadata including overview, genres, and keywords.

Source: [TMDB 5000 Movie Dataset on Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

## Tech Stack

- Python
- pandas
- scikit-learn (TfidfVectorizer, cosine_similarity)
- seaborn / matplotlib (similarity heatmap)

## Usage

```python
recommend('Avatar')
```

**Output:**
```
Falcon Rising
Battle: Los Angeles
Apollo 18
Titan A.E.
Predators
```

## Limitation

This is a **content-based** recommender. It suggests movies based purely on shared text metadata — it has no knowledge of movie quality or user preferences. A collaborative filtering approach (using actual user ratings) would be needed to capture "people who liked X also liked Y" patterns.

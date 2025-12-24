import pandas as pd
import numpy as np
import torch
import re
from collections import defaultdict

def extract_year_bucket(title):
    match = re.search(r'\((\d{4})\)$', title.strip())
    if match:
        year = int(match.group(1))
        if year == 0: return 0   
        if year < 1970: return 1 
        if year < 1975: return 2
        if year < 1980: return 3
        if year < 1985: return 4
        if year < 1990: return 5
        if year < 1995: return 6
        if year < 2000: return 7
        if year < 2005: return 8
        if year < 2010: return 9
        if year < 2015: return 10
        if year < 2020: return 11
        return 12    
    return 0

def calculate_ar2_transitions(sequences):
    """Calculates 2nd-Order Markov transitions for the Cold Start fallback."""
    transitions = defaultdict(lambda: defaultdict(int))
    for seq in sequences:
        if len(seq) < 3: continue
        for i in range(len(seq) - 2):
            prev1, prev2, target = seq[i], seq[i+1], seq[i+2]
            state = (prev1, prev2)
            transitions[state][target] += 1
            
    sorted_transitions = {}
    for prev_tuple, next_dict in transitions.items():
        total = sum(next_dict.values())
        sorted_list = [(m, count/total) for m, count in next_dict.items()]
        sorted_list.sort(key=lambda x: x[1], reverse=True)
        sorted_transitions[prev_tuple] = sorted_list
    return sorted_transitions

def load_and_process_data(ratings_path, movies_path, min_movie_ratings=5, min_user_ratings=5):
    print("Loading data...")
    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)

    # Iterative Filtering
    while True:
        movie_counts = ratings['movieId'].value_counts()
        valid_movies = movie_counts[movie_counts >= min_movie_ratings].index
        ratings = ratings[ratings['movieId'].isin(valid_movies)]
        
        user_counts = ratings['userId'].value_counts()
        valid_users = user_counts[user_counts >= min_user_ratings].index
        ratings = ratings[ratings['userId'].isin(valid_users)]
        
        if len(ratings['movieId'].unique()) == len(valid_movies) and \
           len(ratings['userId'].unique()) == len(valid_users):
            break

    # Type casting
    ratings['movieId'] = ratings['movieId'].astype(str)
    ratings['userId'] = ratings['userId'].astype(str)
    movies['movieId'] = movies['movieId'].astype(str)
    
    # Feature Engineering
    movies['year_bucket'] = movies['title'].apply(extract_year_bucket)
    
    # Mappings
    movie_id_to_genres = {}
    unique_genres = set()
    movie_id_to_year = dict(zip(movies['movieId'], movies['year_bucket']))

    for idx, row in movies.iterrows():
        g_list = row['genres'].split('|')
        movie_id_to_genres[row['movieId']] = g_list
        unique_genres.update(g_list)

    all_movies = sorted(ratings['movieId'].unique())
    movie_to_idx = {mid: i+1 for i, mid in enumerate(all_movies)} 
    idx_to_movie = {i: m for m, i in movie_to_idx.items()}
    
    all_genres = sorted(list(unique_genres))
    genre_to_idx = {g: i+1 for i, g in enumerate(all_genres)}
    
    # Create Tensors
    num_movies = len(all_movies) + 1
    genre_lookup_array = np.zeros(num_movies, dtype=int)
    year_lookup_array = np.zeros(num_movies, dtype=int)

    for mid, idx in movie_to_idx.items():
        gs = movie_id_to_genres.get(mid, [])
        genre_lookup_array[idx] = genre_to_idx[gs[0]] if gs else 0
        year_lookup_array[idx] = movie_id_to_year.get(mid, 0)

    # Timelines
    ratings_sorted = ratings.sort_values(by=['userId', 'timestamp'])
    user_timeline_id = ratings_sorted.groupby('userId')['movieId'].apply(list).to_dict()
    
    # Split Train/Test
    train_timeline_int = {}
    test_timeline_int = {}
    all_train_sequences = [] # For AR(2) calculation

    for user, seq in user_timeline_id.items():
        if len(seq) > 1:
            train_seq = seq[:-1]
            train_timeline_int[user] = [movie_to_idx[m] for m in train_seq if m in movie_to_idx]
            test_timeline_int[user] = [movie_to_idx[m] for m in seq if m in movie_to_idx]
            all_train_sequences.append(train_seq)

    # Pre-calculate AR(2) transitions and Global Pop for Fallback
    print("Calculating AR(2) transitions...")
    ar2_transitions = calculate_ar2_transitions(all_train_sequences)
    global_pop = ratings['movieId'].value_counts().index.tolist()

    data_bundle = {
        'train_timeline': train_timeline_int,
        'test_timeline': test_timeline_int,
        'genre_tensor': torch.tensor(genre_lookup_array, dtype=torch.long),
        'year_tensor': torch.tensor(year_lookup_array, dtype=torch.long),
        'movie_to_idx': movie_to_idx,
        'idx_to_movie': idx_to_movie,
        'num_movies': num_movies,
        'num_genres': len(all_genres) + 1,
        'ar2_transitions': ar2_transitions,
        'global_pop': global_pop
    }
    
    return data_bundle
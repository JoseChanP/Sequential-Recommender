import pandas as pd
import numpy as np
import torch
import re

def extract_year_bucket(title):
    match = re.search(r'\((\d{4})\)$', title.strip())
    if match:
        year = int(match.group(1))
        if year < 1980: return 1 
        if year < 1990: return 2
        if year < 2000: return 3
        if year < 2010: return 4
        if year < 2020: return 5
        return 6    
    return 0

def load_data_and_mappings(ratings_path, movies_path):
    print(f"Loading raw data...")
    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)
    
    ratings['movieId'] = ratings['movieId'].astype(str)
    ratings['userId'] = ratings['userId'].astype(str)
    movies['movieId'] = movies['movieId'].astype(str)
    
    # 1. Feature Engineering (Years)
    movies['year_bucket'] = movies['title'].apply(extract_year_bucket)
    
    # 2. Mappings (ID -> Index)
    all_movies = sorted(ratings['movieId'].unique())
    movie_to_idx = {mid: i+1 for i, mid in enumerate(all_movies)} 
    idx_to_movie = {i: m for m, i in movie_to_idx.items()}
    
    # 3. Genre & Year Lookups
    movie_id_to_genres = {}
    unique_genres = set()
    for idx, row in movies.iterrows():
        g_list = row['genres'].split('|')
        movie_id_to_genres[row['movieId']] = g_list
        unique_genres.update(g_list)
        
    all_genres = sorted(list(unique_genres))
    genre_to_idx = {g: i+1 for i, g in enumerate(all_genres)}
    
    # 4. Create Metadata Tensors (Index -> Feature)
    num_movies = len(all_movies) + 1
    genre_lookup = np.zeros(num_movies, dtype=int)
    year_lookup = np.zeros(num_movies, dtype=int)
    movie_id_to_year = dict(zip(movies['movieId'], movies['year_bucket']))

    for mid, idx in movie_to_idx.items():
        gs = movie_id_to_genres.get(mid, [])
        genre_lookup[idx] = genre_to_idx[gs[0]] if gs else 0
        year_lookup[idx] = movie_id_to_year.get(mid, 0)
        
    mappings = {
        'movie_to_idx': movie_to_idx,
        'idx_to_movie': idx_to_movie,
        'genre_lookup': torch.tensor(genre_lookup, dtype=torch.long),
        'year_lookup': torch.tensor(year_lookup, dtype=torch.long),
        'num_movies': num_movies,
        'num_genres': len(all_genres) + 1,
        'num_years': 7
    }
    return ratings, mappings

def build_seq_tensors(ratings, mappings, max_len=200):
    print("Building sequence tensors...")
    # Sort by timestamp to ensure sequence validity
    ratings_sorted = ratings.sort_values(by=['userId', 'timestamp'])
    timeline_dict = ratings_sorted.groupby('userId')['movieId'].apply(list).to_dict()
    
    movie_to_idx = mappings['movie_to_idx']
    num_users = len(timeline_dict)
    
    input_ids = np.zeros((num_users, max_len), dtype=np.int32)
    target_ids = np.zeros((num_users, max_len), dtype=np.int32)
    user_list = list(timeline_dict.keys())

    valid_idx = 0
    for user in user_list:
        seq = timeline_dict[user]
        seq_int = [movie_to_idx.get(m, 0) for m in seq]
        
        if len(seq_int) < 2: continue
            
        full_input = seq_int[:-1]
        full_target = seq_int[1:]
        
        if len(full_input) > max_len:
            full_input = full_input[-max_len:]
            full_target = full_target[-max_len:]
            
        length = len(full_input)
        input_ids[valid_idx, -length:] = full_input
        target_ids[valid_idx, -length:] = full_target
        valid_idx += 1
        
    # Trim to actual valid users
    input_ids = input_ids[:valid_idx]
    target_ids = target_ids[:valid_idx]
    
    # Convert to Tensors
    inputs_tensor = torch.tensor(input_ids, dtype=torch.long)
    targets_tensor = torch.tensor(target_ids, dtype=torch.long)
    
    # Vectorized Lookup for Genres and Years (Super Fast)
    print("Mapping metadata...")
    genres_tensor = mappings['genre_lookup'][inputs_tensor]
    years_tensor = mappings['year_lookup'][inputs_tensor]
    
    return inputs_tensor, genres_tensor, years_tensor, targets_tensor
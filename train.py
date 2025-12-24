import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src import load_and_process_data, SASRecDatasetOptimized, SASRecModel, HybridRecommender
import time

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 50
BATCH_SIZE = 2048
EPOCHS = 3
LR = 0.001

def main():
    # 1. Load Data
    data = load_and_process_data('data/ratings.csv', 'data/movies.csv')
    
    genre_tensor = data['genre_tensor'].to(DEVICE)
    year_tensor = data['year_tensor'].to(DEVICE)

    # 2. Setup Dataset
    train_dataset = SASRecDatasetOptimized(
        data['train_timeline'], 
        data['genre_tensor'], 
        data['year_tensor'], 
        max_len=MAX_LEN
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    # 3. Initialize Model
    model = SASRecModel(
        num_movies=data['num_movies'], 
        num_genres=data['num_genres'], 
        num_years=13, # 13 buckets defined in utils
        embed_dim=256, 
        max_len=MAX_LEN
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')

    # 4. Training Loop
    print(f"Starting training on {DEVICE}...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            m, g, y, t = [b.to(DEVICE) for b in batch]
            
            if t.sum() == 0: continue
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                logits = model(m, g, y)
                loss = criterion(logits, t)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss {loss.item():.4f}")

    # 5. Save Model
    torch.save(model.state_dict(), "sasrec_model.pth")
    print("Model saved to sasrec_model.pth")

    # 6. Example Inference (Sanity Check)
    print("\n--- Running Inference Check ---")
    recommender = HybridRecommender(
        model, 
        data['ar2_transitions'], 
        data['global_pop'], 
        data['movie_to_idx'], 
        data['idx_to_movie'], 
        DEVICE
    )
    
    # Test Cold Start (AR2)
    short_history = ['1', '2'] # Assuming these IDs exist
    recs = recommender.recommend(short_history, genre_tensor, year_tensor)
    print(f"Short History {short_history} -> Recs: {recs}")

if __name__ == "__main__":
    main()
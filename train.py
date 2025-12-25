import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src import SASRecModel, SASRecDatasetSeq
from src import load_data_and_mappings, build_seq_tensors
import time

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 20
MAX_LEN = 200
LR = 0.001
ACCUMULATION_STEPS = 4

def main():
    # 1. Prepare Data
    ratings, mappings = load_data_and_mappings('data/ratings.csv', 'data/movies.csv')
    
    # 2. Build Sequence Tensors
    train_in, train_gen, train_yr, train_tgt = build_seq_tensors(ratings, mappings, max_len=MAX_LEN)
    
    # 3. Create Minimal Dataset
    train_ds = SASRecDatasetSeq(train_in, train_gen, train_yr, train_tgt)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=0)
    
    # 4. Model
    model = SASRecModel(
        num_movies=mappings['num_movies'],
        num_genres=mappings['num_genres'],
        num_years=mappings['num_years'],
        embed_dim=256,
        max_len=MAX_LEN
    ).to(DEVICE)
    print(mappings['num_movies'], mappings['num_genres'], mappings['num_years'])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    scaler = torch.amp.GradScaler('cuda')
    
    print(f"Starting Training on {len(train_ds)} sequences...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        t0 = time.time()
        
        for i, (m, g, y, t) in enumerate(train_loader):
            m, g, y, t = m.to(DEVICE), g.to(DEVICE), y.to(DEVICE), t.to(DEVICE)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                logits = model(m, g, y)
                loss = criterion(logits.view(-1, mappings['num_movies']), t.view(-1))
                loss = loss / ACCUMULATION_STEPS
            
            scaler.scale(loss).backward()
            total_loss += loss.item()
            if (i + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
            
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Time: {time.time()-t0:.1f}s")
        
    torch.save(model.state_dict(), "sasrec_model.pth")
    print("Model Saved.")

if __name__ == "__main__":
    main()
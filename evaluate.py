import torch
from torch.utils.data import DataLoader
from src import SASRecModel, SASRecDatasetSeq
from src.utils import load_data_and_mappings, build_seq_tensors

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K = 10
MAX_LEN = 200

def evaluate():
    ratings, mappings = load_data_and_mappings('data/ratings.csv', 'data/movies.csv')
    
    # Pre-compute tensors
    test_in, test_gen, test_yr, test_tgt = build_seq_tensors(ratings, mappings, max_len=MAX_LEN)
    
    test_ds = SASRecDatasetSeq(test_in, test_gen, test_yr, test_tgt)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
    
    model = SASRecModel(
        num_movies=mappings['num_movies'],
        num_genres=mappings['num_genres'],
        num_years=mappings['num_years'],
        embed_dim=256,
        max_len=MAX_LEN
    ).to(DEVICE)
    model.load_state_dict(torch.load("sasrec_model.pth"))
    model.eval()
    
    hits = 0
    total = 0
    idx_to_movie = mappings['idx_to_movie']
    
    print("Evaluating...")
    with torch.no_grad():
        for m, g, y, t in test_loader:
            m, g, y = m.to(DEVICE), g.to(DEVICE), y.to(DEVICE)
            logits = model(m, g, y)
            
            # Predict only the last step
            last_logits = logits[:, -1, :] 
            _, top_inds = torch.topk(last_logits, K)
            top_inds = top_inds.cpu().numpy()
            targets = t[:, -1].numpy()
            
            for i, target_idx in enumerate(targets):
                if target_idx == 0: continue
                total += 1
                if target_idx in top_inds[i]:
                    hits += 1

    print(f"Hit Rate @ {K}: {hits/total:.4f}")

if __name__ == "__main__":
    evaluate()
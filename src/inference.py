import torch

class HybridRecommender:
    def __init__(self, sasrec_model, ar2_transitions, global_pop, movie_to_idx, idx_to_movie, device):
        self.model = sasrec_model
        self.ar2 = ar2_transitions
        self.global_pop = global_pop
        self.movie_to_idx = movie_to_idx
        self.idx_to_movie = idx_to_movie
        self.device = device
        self.model.eval()

    def _predict_global(self, k):
        return self.global_pop[:k]

    def _predict_ar2(self, history, k):
        """Uses Markov Chains if history < 3 items"""
        # If only 1 item, use first-order (approx by using tuple of 1)
        if len(history) < 2:
            return self._predict_global(k) # Fallback to global if 1 item is too sparse
        
        last_two = tuple(history[-2:])
        
        # Get Markov predictions
        recs = [m for m, _ in self.ar2.get(last_two, [])]
        
        # Fill with global popularity if we don't have enough markov predictions
        if len(recs) < k:
            seen = set(recs) | set(history)
            for m in self.global_pop:
                if m not in seen:
                    recs.append(m)
                    if len(recs) == k: break
        return recs[:k]

    def _predict_sasrec(self, history, genre_tensor, year_tensor, k):
        """Uses Transformer if history >= 3 items"""
        seq_idxs = [self.movie_to_idx.get(m, 0) for m in history]
        seq_tensor = torch.tensor(seq_idxs, device=self.device).unsqueeze(0)
        
        if seq_tensor.size(1) > self.model.max_len:
            seq_tensor = seq_tensor[:, -self.model.max_len:]
            
        genre_seq = genre_tensor[seq_tensor]
        year_seq = year_tensor[seq_tensor]

        with torch.no_grad():
            logits = self.model(seq_tensor, genre_seq, year_seq)
            scores = logits[0, :] # Output of linear layer
            
        # Get top K
        top_k_vals, top_k_indices = torch.topk(scores, k + len(history))
        top_indices = top_k_indices.cpu().numpy()
        
        recs = []
        history_set = set(history)
        for idx in top_indices:
            movie_id = self.idx_to_movie.get(idx, None)
            if movie_id and movie_id not in history_set:
                recs.append(movie_id)
                if len(recs) == k:
                    break
        return recs

    def recommend(self, history, genre_tensor, year_tensor, k=10):
        # STRATEGY: Cascade Fallback
        
        # 1. Cold Start (New User) -> Global Popularity
        if not history:
            return self._predict_global(k)
        
        # 2. Warm Start (Short History) -> AR(2) Markov
        if len(history) < 3:
            return self._predict_ar2(history, k)
        
        # 3. Established User -> SASRec
        return self._predict_sasrec(history, genre_tensor, year_tensor, k)
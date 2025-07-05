import faiss
import torch
import numpy as np

class FaissRetrievalDatabase:
    def __init__(self, dim):
        self.dim = dim
        self.kf_counter = 0
        self.kf_ids = []

        self.index = faiss.IndexFlatL2(dim) 
        self.ids = []  

    def update(self, frame, k, max_thresh=0.0, add_after_query=True):
        token = frame.token
        if token.ndim == 1:
            token = token.unsqueeze(0) 
        vector = token.detach().cpu().numpy().astype(np.float32)

        topk_image_inds = []
        id = self.kf_counter
        if self.kf_counter > 0:
            distances, result_ids = self.query(vector, k)
            scores = torch.from_numpy(distances[0])             
            topk_image_inds = torch.tensor(result_ids[0])

            print(scores)
            valid = scores < max_thresh
            topk_image_inds = topk_image_inds[valid].tolist()

        if add_after_query:
            self.add_to_database(token, id)

        return topk_image_inds
        
    def add_to_database(self, vector, id):
        self.index.add(vector)
        self.ids.append(id)
        self.kf_counter += 1

    def query(self, vector, topk=5):
        distances, indices = self.index.search(vector, topk) 
        result_ids = []
        for inds in indices:
            result_ids.append([self.ids[i] if i < len(self.ids) else -1 for i in inds])
        return distances, result_ids

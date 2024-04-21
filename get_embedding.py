from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from sentence_transformers.quantization import quantize_embeddings
import torch

# 1. Specify preffered dimensions
# dimensions = 512

# 2. load model
embedding_model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
# def get_embeddings(data):
    

# # For retrieval you need to pass this prompt.
# query = 'Represent this sentence for searching relevant passages: A man is eating a piece of bread'

# docs = [
#     query,
#     "A man is eating food.",
#     "A man is eating pasta.",
#     "The girl is carrying a baby.",
#     "A man is riding a horse.",
# ]

# # 2. Encode
# print ("getting")
# embeddings = model.encode(docs)
# print("benchmarking")
# import time
# start = time.time()
# for i in range(10):
#     with torch.no_grad():
#         embeddings = model.encode(docs)
# print(time.time() - start)

# print(embeddings.shape)

# similarities = cos_sim(embeddings[0], embeddings[1:])
# print('similarities:', similarities)

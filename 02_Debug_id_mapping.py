import pickle

with open("id_mapping.pkl", "rb") as f:
    id_mapping = pickle.load(f)

print("Homestay ID mapping (vector index → DB ID):", id_mapping)

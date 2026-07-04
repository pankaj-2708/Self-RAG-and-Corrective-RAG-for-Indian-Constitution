import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

if not os.environ.get('OLLAMA_API_KEY'):
    raise ValueError("No ollama api key")
else:
    OLLAMA_API_KEY = os.environ['OLLAMA_API_KEY']

# Resolve paths relative to the file location to make it run from anywhere
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../data"))
vector_store_path = os.path.join(DATA_DIR, "constitution_and_ipc.faiss")

generator_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = FAISS.load_local(vector_store_path, embeddings=generator_embeddings, allow_dangerous_deserialization=True)
all_docs = vector_store.docstore._dict
faiss_index = vector_store.index

t_documents = []
t_metadata = []
t_embedding = []
for idx, doc_id in vector_store.index_to_docstore_id.items():
    doc = vector_store.docstore._dict[doc_id]
    embedding = faiss_index.reconstruct(idx)
    t_documents.append(doc.page_content)
    t_metadata.append(doc.metadata)
    t_embedding.append(embedding)

df = pd.DataFrame({"document": t_documents, "metadata": t_metadata, "embedding": t_embedding})

embeddings = []
for i in range(df.shape[0]):
    embeddings.append(df['embedding'].values[i])
embeddings = np.array(embeddings)

inertias = []
silhouette = []
best_k = -1
best_si = -1

for k in tqdm(range(4, 25)):
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(embeddings)
    y_pred = model.predict(embeddings)
    c_si = silhouette_score(embeddings, y_pred)
    if c_si > best_si:
        best_si = c_si
        best_k = k
    silhouette.append(c_si)
    inertias.append(model.inertia_)

print(f"Best K: {best_k}, Best Silhouette Score: {best_si}")

# Plot inertias
plt.plot(range(4, 25), inertias, marker='o')
plt.title('Inertia vs K')
plt.xlabel('Number of clusters k')
plt.ylabel('Inertia')
plt.savefig(os.path.join(DATA_DIR, "optimal_k_inertia.png"))
print("Saved optimal_k_inertia.png to data directory")

# Cluster with best_k
k = best_k
model = KMeans(n_clusters=k, random_state=42)
model.fit(embeddings)
y_pred = model.predict(embeddings)

df['cluster'] = y_pred

for c in sorted(df["cluster"].unique()):
    print(f"--- Cluster {c} ---")
    print(df[df.cluster == c]["document"].sample(min(3, len(df[df.cluster == c]))).tolist())

sample_size = 100
sampled_df, _ = train_test_split(
    df, train_size=sample_size, stratify=df["cluster"], random_state=42
)

df.to_csv(os.path.join(DATA_DIR, "df.csv"), index=False)
sampled_df.to_csv(os.path.join(DATA_DIR, "sampled_df.csv"), index=False)
print("Saved df.csv and sampled_df.csv to data directory")

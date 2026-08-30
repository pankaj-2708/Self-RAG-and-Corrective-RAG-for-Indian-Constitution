import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

import yaml

load_dotenv()

# Resolve paths relative to the file location to make it run from anywhere
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../data"))
vector_store_path = os.path.join(DATA_DIR, "constitution_and_ipc.chroma")

CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.yaml")
with open(CONFIG_PATH, "r") as f:
    eval_config = yaml.safe_load(f)

c_cfg = eval_config["clustering"]

vector_store = Chroma(
    collection_name="constitution_and_ipc",
    persist_directory=vector_store_path,
)

data = vector_store.get(include=["documents", "metadatas", "embeddings"])
t_documents = data["documents"]
t_metadata = data["metadatas"]
t_embedding = data["embeddings"]

df = pd.DataFrame(
    {"document": t_documents, "metadata": t_metadata, "embedding": list(t_embedding)}
)

embeddings = []
for i in range(df.shape[0]):
    embeddings.append(df["embedding"].values[i])
embeddings = np.array(embeddings)

inertias = []
silhouette = []
best_k = -1
best_si = -1

for k in tqdm(range(c_cfg["k_min"], c_cfg["k_max"])):
    model = KMeans(n_clusters=k, random_state=c_cfg["random_state"])
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
plt.plot(range(c_cfg["k_min"], c_cfg["k_max"]), inertias, marker="o")
plt.title("Inertia vs K")
plt.xlabel("Number of clusters k")
plt.ylabel("Inertia")
plt.savefig(os.path.join(DATA_DIR, "optimal_k_inertia.png"))
print("Saved optimal_k_inertia.png to data directory")

# Cluster with best_k
k = best_k
model = KMeans(n_clusters=k, random_state=c_cfg["random_state"])
model.fit(embeddings)
y_pred = model.predict(embeddings)

df["cluster"] = y_pred

for c in sorted(df["cluster"].unique()):
    print(f"--- Cluster {c} ---")
    print(
        df[df.cluster == c]["document"]
        .sample(min(3, len(df[df.cluster == c])))
        .tolist()
    )

sample_size = c_cfg["sample_size"]
sampled_df, _ = train_test_split(
    df,
    train_size=sample_size,
    stratify=df["cluster"],
    random_state=c_cfg["random_state"],
)

df.to_csv(os.path.join(DATA_DIR, "df.csv"), index=False)
sampled_df.to_csv(os.path.join(DATA_DIR, "sampled_df.csv"), index=False)
print("Saved df.csv and sampled_df.csv to data directory")

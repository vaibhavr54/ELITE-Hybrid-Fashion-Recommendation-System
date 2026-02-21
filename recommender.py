import numpy as np
import pandas as pd
import faiss
import os

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from scipy.sparse import load_npz


# ==========================================================
# 1️⃣ LOAD RAW DATA + EMBEDDINGS
# ==========================================================

data = pd.read_pickle("pickels/16k_apperal_data_preprocessed")

text_embeddings = np.load("embeddings/title_embeddings.npy")
image_embeddings = np.load("embeddings/image_embeddings.npy")
image_ids = np.load("embeddings/image_ids.npy")
extra_features = load_npz("embeddings/extra_features.npz")


# ==========================================================
# 2️⃣ ALIGN EVERYTHING (MATCH NOTEBOOK LOGIC)
# ==========================================================

# Remove .jpg from image ids
available_asins = [asin.replace(".jpg", "") for asin in image_ids]

# Map asin → original dataframe index
asin_to_index = {asin: idx for idx, asin in enumerate(data["asin"])}

valid_indices = [
    asin_to_index[asin]
    for asin in available_asins
    if asin in asin_to_index
]

# Create aligned dataset
aligned_data = data.iloc[valid_indices].reset_index(drop=True)

aligned_text_embeddings = text_embeddings[valid_indices]
aligned_meta_features = extra_features[valid_indices]
aligned_image_embeddings = image_embeddings

# Normalize all embeddings
aligned_text_embeddings = normalize(aligned_text_embeddings)
aligned_image_embeddings = normalize(aligned_image_embeddings)
aligned_meta_features = normalize(aligned_meta_features)


# ==========================================================
# 3️⃣ BUILD FAISS INDEX (ON ALIGNED TEXT)
# ==========================================================

dimension = aligned_text_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(aligned_text_embeddings.astype("float32"))


# ==========================================================
# 4️⃣ HELPER FUNCTIONS
# ==========================================================

def normalize_scores(scores):
    min_val = np.min(scores)
    max_val = np.max(scores)
    if max_val - min_val == 0:
        return scores
    return (scores - min_val) / (max_val - min_val)


def faiss_retrieve(query_index, candidate_k=300):
    query_vec = aligned_text_embeddings[query_index].reshape(1, -1).astype("float32")
    scores, indices = index.search(query_vec, candidate_k)
    return indices[0], scores[0]


# ==========================================================
# 5️⃣ MAIN HYBRID RECOMMENDER
# ==========================================================

def hybrid_recommender_faiss(
    query_index,
    top_k=20,
    w_text=0.4,
    w_img=0.3,
    w_meta=0.3,
    lambda_param=0.7,
    category_boost=0.2,
    price_boost=0.15,
    price_tolerance=0.3,
    candidate_k=300
):

    # --------------------------
    # Stage 1: FAISS Retrieval
    # --------------------------
    candidate_indices, text_scores = faiss_retrieve(query_index, candidate_k)

    text_sim = normalize_scores(text_scores.copy())

    img_sim = cosine_similarity(
        aligned_image_embeddings[query_index].reshape(1, -1),
        aligned_image_embeddings[candidate_indices]
    )[0]

    meta_sim = cosine_similarity(
        aligned_meta_features[query_index],
        aligned_meta_features[candidate_indices]
    )[0]

    img_sim = normalize_scores(img_sim)
    meta_sim = normalize_scores(meta_sim)

    # --------------------------
    # Stage 2: Hybrid Fusion
    # --------------------------
    relevance = (
        w_text * text_sim +
        w_img * img_sim +
        w_meta * meta_sim
    )

    # Remove self
    for i, idx in enumerate(candidate_indices):
        if idx == query_index:
            relevance[i] = -1

    # --------------------------
    # Stage 3: Soft Business Boosting
    # --------------------------

    # Category boost
    query_category = aligned_data.iloc[query_index]["product_type_name"]

    category_mask = (
        aligned_data.iloc[candidate_indices]["product_type_name"]
        == query_category
    ).values.astype(int)

    # Price cleaning if not already present
    if "price_clean" not in aligned_data.columns:

        aligned_data["price_clean"] = (
            aligned_data["formatted_price"]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
        )

        aligned_data["price_clean"] = pd.to_numeric(
            aligned_data["price_clean"],
            errors="coerce"
        )

    query_price = aligned_data.iloc[query_index]["price_clean"]

    if np.isnan(query_price):
        price_mask = np.zeros(len(candidate_indices))
    else:
        lower = query_price * (1 - price_tolerance)
        upper = query_price * (1 + price_tolerance)

        price_mask = (
            (aligned_data.iloc[candidate_indices]["price_clean"] >= lower) &
            (aligned_data.iloc[candidate_indices]["price_clean"] <= upper)
        ).values.astype(int)

    relevance = relevance * (
        1 +
        category_boost * category_mask +
        price_boost * price_mask
    )

    # --------------------------
    # Stage 4: MMR Re-ranking
    # --------------------------

    selected = []

    for _ in range(top_k):

        if not selected:
            idx = np.argmax(relevance)
            if relevance[idx] <= 0:
                break
            selected.append(idx)
            continue

        mmr_scores = []

        for i in range(len(candidate_indices)):

            if i in selected or relevance[i] <= 0:
                mmr_scores.append(-1)
                continue

            diversity_penalty = max(
                cosine_similarity(
                    aligned_image_embeddings[candidate_indices[i]].reshape(1, -1),
                    aligned_image_embeddings[
                        candidate_indices[selected]
                    ]
                )[0]
            )

            mmr_score = (
                lambda_param * relevance[i]
                - (1 - lambda_param) * diversity_penalty
            )

            mmr_scores.append(mmr_score)

        idx = np.argmax(mmr_scores)

        if mmr_scores[idx] <= 0:
            break

        selected.append(idx)

    # Return aligned indices
    return [candidate_indices[i] for i in selected]
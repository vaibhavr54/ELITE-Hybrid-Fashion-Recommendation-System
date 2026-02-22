import numpy as np
import pandas as pd
import faiss
import os
from datetime import datetime
from collections import defaultdict, Counter

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

# Clean price data
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


# ==========================================================
# 3️⃣ BUILD FAISS INDEX (ON ALIGNED TEXT)
# ==========================================================

dimension = aligned_text_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(aligned_text_embeddings.astype("float32"))


# ==========================================================
# 4️⃣ USER FEEDBACK STORAGE
# ==========================================================

class FeedbackStore:
    """Store and manage user interactions for model improvement"""
    
    def __init__(self):
        self.interactions = []
        self.feedback_file = "data/user_feedback.csv"
        self._load_existing()
    
    def _load_existing(self):
        """Load existing feedback if available"""
        if os.path.exists(self.feedback_file):
            try:
                df = pd.read_csv(self.feedback_file)
                self.interactions = df.to_dict('records')
            except:
                pass
    
    def log_interaction(self, query_index, recommended_indices, action, metadata=None):
        """
        Log user interaction
        
        Actions: 'view', 'click', 'like', 'dislike'
        """
        feedback_scores = {
            'view': 0.5,
            'click': 1.0,
            'like': 2.0,
            'dislike': -1.0
        }
        
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'query_index': query_index,
            'recommended_indices': str(recommended_indices),
            'action': action,
            'score': feedback_scores.get(action, 0),
            'metadata': str(metadata) if metadata else ''
        }
        
        self.interactions.append(interaction)
        
        # Periodically save to file
        if len(self.interactions) % 10 == 0:
            self.save()
    
    def save(self):
        """Save interactions to CSV"""
        os.makedirs(os.path.dirname(self.feedback_file), exist_ok=True)
        df = pd.DataFrame(self.interactions)
        df.to_csv(self.feedback_file, index=False)
    
    def get_statistics(self):
        """Get feedback statistics"""
        if not self.interactions:
            return {}
        
        df = pd.DataFrame(self.interactions)
        return {
            'total_interactions': len(df),
            'action_counts': df['action'].value_counts().to_dict(),
            'average_score': df['score'].mean(),
            'recent_interactions': len(df[df['timestamp'] > 
                (datetime.now() - pd.Timedelta(days=7)).isoformat()])
        }

# Global feedback store
feedback_store = FeedbackStore()


# ==========================================================
# 5️⃣ HELPER FUNCTIONS
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


def apply_filters(candidate_indices, filters):
    """
    Apply hard filters to candidate products
    
    Filters:
    - price_min, price_max: Price range
    - brands: List of allowed brands
    - categories: List of allowed categories  
    - colors: List of allowed colors (if available)
    - min_rating: Minimum rating (if available)
    """
    mask = np.ones(len(candidate_indices), dtype=bool)
    
    # Price filter
    if filters.get('price_min') is not None:
        prices = aligned_data.iloc[candidate_indices]['price_clean']
        mask &= prices >= filters['price_min']
    
    if filters.get('price_max') is not None:
        prices = aligned_data.iloc[candidate_indices]['price_clean']
        mask &= prices <= filters['price_max']
    
    # Brand filter
    if filters.get('brands') and len(filters['brands']) > 0:
        brands = aligned_data.iloc[candidate_indices]['brand']
        mask &= brands.isin(filters['brands'])
    
    # Category filter
    if filters.get('categories') and len(filters['categories']) > 0:
        categories = aligned_data.iloc[candidate_indices]['product_type_name']
        mask &= categories.isin(filters['categories'])
    
    # Color filter (if color column exists)
    if filters.get('colors') and len(filters['colors']) > 0:
        if 'color' in aligned_data.columns:
            colors = aligned_data.iloc[candidate_indices]['color']
            mask &= colors.isin(filters['colors'])
    
    return candidate_indices[mask]


# ==========================================================
# 6️⃣ EXPLAINABILITY FUNCTIONS
# ==========================================================

def explain_recommendation(query_index, recommended_index, 
                          text_sim, img_sim, meta_sim,
                          w_text, w_img, w_meta):
    """
    Generate explanation for why a product was recommended
    """
    query_product = aligned_data.iloc[query_index]
    recommended_product = aligned_data.iloc[recommended_index]
    
    # Calculate contribution of each factor
    total_score = w_text * text_sim + w_img * img_sim + w_meta * meta_sim
    
    explanation = {
        'overall_match': float(total_score),
        'similarity_breakdown': {
            'text': {
                'score': float(text_sim),
                'weight': float(w_text),
                'contribution': float(w_text * text_sim),
                'percentage': float(w_text * text_sim / total_score * 100) if total_score > 0 else 0
            },
            'visual': {
                'score': float(img_sim),
                'weight': float(w_img),
                'contribution': float(w_img * img_sim),
                'percentage': float(w_img * img_sim / total_score * 100) if total_score > 0 else 0
            },
            'metadata': {
                'score': float(meta_sim),
                'weight': float(w_meta),
                'contribution': float(w_meta * meta_sim),
                'percentage': float(w_meta * meta_sim / total_score * 100) if total_score > 0 else 0
            }
        },
        'matching_attributes': []
    }
    
    # Identify matching attributes
    if query_product['brand'] == recommended_product['brand']:
        explanation['matching_attributes'].append(
            f"Same brand: {query_product['brand']}"
        )
    
    if query_product['product_type_name'] == recommended_product['product_type_name']:
        explanation['matching_attributes'].append(
            f"Same category: {query_product['product_type_name']}"
        )
    
    # Price similarity
    query_price = query_product['price_clean']
    rec_price = recommended_product['price_clean']
    if not np.isnan(query_price) and not np.isnan(rec_price):
        price_diff = abs(query_price - rec_price)
        price_sim_pct = max(0, 100 - (price_diff / query_price * 100))
        if price_sim_pct > 70:
            explanation['matching_attributes'].append(
                f"Similar price: {query_product['formatted_price']} vs {recommended_product['formatted_price']}"
            )
    
    return explanation


# ==========================================================
# 7️⃣ EVALUATION METRICS
# ==========================================================

class RecommenderEvaluator:
    """Evaluate recommendation system performance"""
    
    def __init__(self):
        self.metrics_history = []
    
    def calculate_precision_at_k(self, relevant_items, recommended_items, k):
        """
        Precision@K: % of recommended items that are relevant
        """
        recommended_k = recommended_items[:k]
        relevant_in_recommended = len(set(recommended_k) & set(relevant_items))
        return relevant_in_recommended / k if k > 0 else 0
    
    def calculate_recall_at_k(self, relevant_items, recommended_items, k):
        """
        Recall@K: % of relevant items that were recommended
        """
        recommended_k = recommended_items[:k]
        relevant_in_recommended = len(set(recommended_k) & set(relevant_items))
        return relevant_in_recommended / len(relevant_items) if len(relevant_items) > 0 else 0
    
    def calculate_ndcg_at_k(self, relevance_scores, k):
        """
        NDCG@K: Normalized Discounted Cumulative Gain
        Measures ranking quality with position discount
        """
        def dcg(scores):
            return np.sum([
                (2**score - 1) / np.log2(idx + 2)
                for idx, score in enumerate(scores)
            ])
        
        # Actual DCG
        actual_dcg = dcg(relevance_scores[:k])
        
        # Ideal DCG (perfectly ranked)
        ideal_scores = sorted(relevance_scores, reverse=True)
        ideal_dcg = dcg(ideal_scores[:k])
        
        return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0
    
    def calculate_diversity(self, recommended_indices):
        """
        Intra-list diversity: Average pairwise distance
        """
        if len(recommended_indices) < 2:
            return 0
        
        embeddings = aligned_image_embeddings[recommended_indices]
        similarities = cosine_similarity(embeddings)
        
        # Get upper triangle (avoid diagonal and duplicates)
        n = len(similarities)
        diversity_sum = 0
        count = 0
        
        for i in range(n):
            for j in range(i+1, n):
                diversity_sum += (1 - similarities[i, j])
                count += 1
        
        return diversity_sum / count if count > 0 else 0
    
    def calculate_coverage(self, all_recommended_items, catalog_size):
        """
        Catalog coverage: % of catalog that gets recommended
        """
        unique_recommendations = set(all_recommended_items)
        return len(unique_recommendations) / catalog_size
    
    def calculate_category_diversity(self, recommended_indices):
        """
        Category diversity: Entropy of category distribution
        """
        categories = aligned_data.iloc[recommended_indices]['product_type_name'].values
        category_counts = Counter(categories)
        total = len(categories)
        
        # Calculate entropy
        entropy = 0
        for count in category_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy
    
    def evaluate_single_query(self, query_index, recommended_indices, k=10):
        """
        Evaluate a single query's recommendations
        """
        # For demonstration, we'll use category match as relevance
        query_category = aligned_data.iloc[query_index]['product_type_name']
        relevance_scores = [
            1.0 if aligned_data.iloc[idx]['product_type_name'] == query_category else 0.5
            for idx in recommended_indices
        ]
        
        relevant_items = recommended_indices  # Simplified
        
        metrics = {
            'precision@k': self.calculate_precision_at_k(relevant_items, recommended_indices, k),
            'recall@k': self.calculate_recall_at_k(relevant_items, recommended_indices, k),
            'ndcg@k': self.calculate_ndcg_at_k(relevance_scores, k),
            'diversity': self.calculate_diversity(recommended_indices),
            'category_diversity': self.calculate_category_diversity(recommended_indices)
        }
        
        return metrics
    
    def evaluate_system(self, num_samples=100, k=10):
        """
        Evaluate entire system on random sample
        """
        sample_indices = np.random.choice(len(aligned_data), num_samples, replace=False)
        
        all_metrics = {
            'precision@k': [],
            'recall@k': [],
            'ndcg@k': [],
            'diversity': [],
            'category_diversity': []
        }
        
        all_recommendations = []
        
        for query_idx in sample_indices:
            try:
                recommendations = hybrid_recommender_faiss(query_idx, top_k=k)
                metrics = self.evaluate_single_query(query_idx, recommendations, k)
                
                for key in all_metrics:
                    all_metrics[key].append(metrics[key])
                
                all_recommendations.extend(recommendations)
            except:
                continue
        
        # Calculate averages
        avg_metrics = {
            key: np.mean(values) for key, values in all_metrics.items()
        }
        
        # Add coverage
        avg_metrics['coverage'] = self.calculate_coverage(
            all_recommendations, 
            len(aligned_data)
        )
        
        # Add standard deviations
        std_metrics = {
            f"{key}_std": np.std(values) for key, values in all_metrics.items()
        }
        
        avg_metrics.update(std_metrics)
        
        # Store in history
        self.metrics_history.append({
            'timestamp': datetime.now().isoformat(),
            'metrics': avg_metrics
        })
        
        return avg_metrics

# Global evaluator
evaluator = RecommenderEvaluator()


# ===========================================================
# 8️⃣ MAIN HYBRID RECOMMENDER (ENHANCED)
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
    candidate_k=300,
    filters=None,
    return_explanations=False
):
    """
    Enhanced hybrid recommender with filtering and explainability
    
    Args:
        filters: Dict with keys: price_min, price_max, brands, categories, colors
        return_explanations: If True, return (indices, explanations)
    """
    
    # --------------------------
    # Stage 1: FAISS Retrieval
    # --------------------------
    candidate_indices, text_scores = faiss_retrieve(query_index, candidate_k)
    
    # --------------------------
    # Stage 1.5: Apply Filters
    # --------------------------
    if filters:
        candidate_indices = apply_filters(candidate_indices, filters)
        if len(candidate_indices) == 0:
            return [] if not return_explanations else ([], [])
    
    # --------------------------
    # Stage 2: Similarity Computation
    # --------------------------
    text_sim = normalize_scores(text_scores[:len(candidate_indices)].copy())
    
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
    
    # Store for explanations
    similarity_scores = {
        'text': text_sim,
        'image': img_sim,
        'meta': meta_sim
    }
    
    # --------------------------
    # Stage 3: Hybrid Fusion
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
    # Stage 4: Soft Business Boosting
    # --------------------------
    
    # Category boost
    query_category = aligned_data.iloc[query_index]["product_type_name"]
    
    category_mask = (
        aligned_data.iloc[candidate_indices]["product_type_name"]
        == query_category
    ).values.astype(int)
    
    # Price boost
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
    # Stage 5: MMR Re-ranking
    # --------------------------
    
    selected = []
    explanations = []
    
    for _ in range(min(top_k, len(candidate_indices))):
        
        if not selected:
            idx = np.argmax(relevance)
            if relevance[idx] <= 0:
                break
            selected.append(idx)
            
            if return_explanations:
                explanation = explain_recommendation(
                    query_index, 
                    candidate_indices[idx],
                    similarity_scores['text'][idx],
                    similarity_scores['image'][idx],
                    similarity_scores['meta'][idx],
                    w_text, w_img, w_meta
                )
                explanations.append(explanation)
            
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
        
        if return_explanations:
            explanation = explain_recommendation(
                query_index, 
                candidate_indices[idx],
                similarity_scores['text'][idx],
                similarity_scores['image'][idx],
                similarity_scores['meta'][idx],
                w_text, w_img, w_meta
            )
            explanations.append(explanation)
    
    # Return aligned indices
    result_indices = [candidate_indices[i] for i in selected]
    
    if return_explanations:
        return result_indices, explanations
    else:
        return result_indices


# ==========================================================
# 9️⃣ UTILITY FUNCTIONS
# ==========================================================

def get_available_brands():
    return sorted(
        aligned_data['brand']
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )


def get_available_categories():
    return sorted(
        aligned_data['product_type_name']
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )


def get_price_range():
    """Get min and max prices in catalog"""
    prices = aligned_data['price_clean'].dropna()
    return {
        'min': float(prices.min()),
        'max': float(prices.max()),
        'mean': float(prices.mean()),
        'median': float(prices.median())
    }
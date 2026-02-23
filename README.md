# ELITE — AI-Powered Fashion Recommendation System

A production-ready, multimodal fashion recommendation engine built with Flask, FAISS, and neural embeddings. LUXE combines text, visual, and metadata signals through a hybrid retrieval-ranking pipeline to deliver explainable, diverse, and contextually relevant product recommendations.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Recommendation Pipeline](#recommendation-pipeline)
- [Evaluation Metrics](#evaluation-metrics)
- [API Reference](#api-reference)
- [Configuration](#configuration)

---

## Overview

ELITE is an end-to-end fashion product discovery platform that addresses the core challenge of multimodal similarity search across a catalog of 16,000+ apparel items. The system fuses three independent embedding modalities — title text (transformer-based), product images (CNN-based), and structured metadata — into a single relevance score, then applies Maximal Marginal Relevance (MMR) re-ranking to balance relevance with result diversity.

The project was designed with recruiter-readiness in mind: it includes a live metrics dashboard, a user feedback loop, explainability outputs per recommendation, and a RESTful API layer.

---

## Key Features

- **Multimodal Hybrid Retrieval** — Weighted fusion of text, image, and metadata embeddings with configurable per-query weights
- **FAISS-Accelerated Search** — Inner product index enabling sub-second approximate nearest neighbor retrieval over 16K+ products
- **MMR Re-ranking** — Maximal Marginal Relevance algorithm (lambda-parameterized) to control the relevance-diversity tradeoff
- **Hard Filtering** — Price range, brand, and category filters applied post-retrieval, pre-ranking
- **Soft Business Boosting** — Category and price proximity signals boost relevance without overriding semantic similarity
- **Explainability Layer** — Per-recommendation breakdowns showing text/visual/metadata contribution percentages and matching attributes
- **Performance Dashboard** — Live metrics page covering Precision@K, Recall@K, NDCG@K, Diversity, and Catalog Coverage
- **RESTful API** — JSON endpoints for evaluation, feedback collection, and filter retrieval

---

## System Architecture
<img width="8166" height="8192" alt="architecture_diagram" src="https://github.com/user-attachments/assets/38664c4b-709d-47a4-8b06-d2d10f250263" />

---

## Tech Stack

| Layer | Technology |
|---|---|
| Web Framework | Flask |
| Vector Search | FAISS (IndexFlatIP) |
| Embeddings | NumPy (.npy), SciPy Sparse (.npz) |
| Data Processing | Pandas, NumPy |
| Similarity | scikit-learn (cosine similarity, normalize) |
| Frontend | Jinja2 Templates, Vanilla JS, CSS3 |
| Fonts | Google Fonts (Cormorant Garamond, Montserrat) |
| Data Storage | Pickle, CSV, JSON |

---

## Project Structure

```
luxe/
├── app.py                    # Flask application, routes, session management
├── recommender.py            # Core recommendation engine + evaluator + feedback
├── generate_metrics.py       # Offline evaluation script → metrics_data.json
│
├── templates/
│   ├── index.html            # Search and filter interface
│   ├── results.html          # Recommendation results with explanations
│   └── metrics.html          # Performance dashboard
│
├── static/
│   └── css/
│       └── style.css         # Application stylesheet
│
├── embeddings/
│   ├── title_embeddings.npy  # Text embeddings (transformer)
│   ├── image_embeddings.npy  # Visual embeddings (CNN)
│   ├── image_ids.npy         # ASIN-to-embedding mapping
│   └── extra_features.npz    # Sparse metadata features
│
├── pickels/
│   └── 16k_apperal_data_preprocessed  # Preprocessed product DataFrame
│
│
└── metrics_data.json         # Cached evaluation results
```

---

## Installation

**Prerequisites:** Python 3.8+, pip

```bash
# Clone the repository
git clone https://github.com/vaibhavr54/ELITE-Hybrid-Fashion-Recommendation-System.git
cd ELITE-Hybrid-Fashion-Recommendation-System

# Create a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# Install dependencies
pip install flask faiss-cpu numpy pandas scikit-learn scipy

# Set environment variables
export SECRET_KEY="your-secret-key"
export PORT=5000                # Optional, defaults to 5000
```

**Note:** The `embeddings/` and `pickels/` directories contain large binary files. These must be generated or obtained separately. See the notebooks (if included) for embedding generation instructions.

---

## Usage

**Run the development server:**

```bash
python app.py
```

The application will be available at `http://localhost:5000`.

**Generate offline evaluation metrics** (run once before visiting `/metrics`):

```bash
python generate_metrics.py
```

This evaluates the system over 50 random samples and writes results to `metrics_data.json`.

---

## Recommendation Pipeline

The core pipeline in `recommender.py` operates in five sequential stages:

**Stage 1 — FAISS Retrieval**
A query product's normalized text embedding is used to retrieve the top 300 nearest neighbor candidates from the FAISS flat inner product index. This acts as an efficient first-pass filter over the full catalog.

**Stage 1.5 — Hard Filtering**
Candidate indices are filtered against user-specified constraints: price range (min/max), allowed brands, and allowed categories. Any candidates outside these bounds are removed before scoring.

**Stage 2 — Multimodal Similarity Computation**
For each remaining candidate, three similarity scores are computed independently: cosine similarity against text embeddings, image embeddings, and sparse metadata features. Each score is min-max normalized to [0, 1].

**Stage 3 — Weighted Hybrid Fusion**
The three normalized scores are fused into a single relevance score using configurable weights `(w_text, w_img, w_meta)`. Weights are normalized to sum to 1 before application.

```
relevance = w_text * text_sim + w_img * img_sim + w_meta * meta_sim
```

**Stage 4 — Soft Business Boosting**
Relevance scores are multiplied by a boost factor that rewards items sharing the query product's category (+20% by default) and falling within ±30% of the query product's price (+15% by default). This keeps business logic as a soft signal, not a hard constraint.

**Stage 5 — MMR Re-ranking**
Maximal Marginal Relevance iteratively selects the next item that maximizes the tradeoff between relevance and dissimilarity to already-selected items. The `lambda` parameter (0–1) controls this tradeoff: higher values favor relevance, lower values favor diversity.

```
MMR(i) = lambda * relevance(i) - (1 - lambda) * max_similarity(i, selected)
```

---

## Evaluation Metrics

The `RecommenderEvaluator` class computes the following metrics using sampled query-relevant pairs derived from shared category membership as a relevance proxy:

| Metric | Description |
|---|---|
| Precision@K | Fraction of top-K results that are relevant |
| Recall@K | Fraction of all relevant items captured in top-K |
| NDCG@K | Normalized Discounted Cumulative Gain — ranking quality with position discount |
| Diversity | Average pairwise image embedding dissimilarity among results |
| Category Diversity | Shannon entropy of category distribution in results |
| Catalog Coverage | Fraction of catalog items that appear in at least one recommendation set |

Metrics are accessible via the `/metrics` dashboard or re-triggered via the `/api/evaluate` endpoint.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Main search interface |
| POST | `/recommend` | Submit a recommendation query with parameters and filters |
| GET | `/results` | View recommendation results (session-based) |
| GET | `/metrics` | Performance metrics dashboard |
| POST | `/api/evaluate` | Trigger a new evaluation run (JSON: `num_samples`, `k`) |
| GET | `/api/filters` | Retrieve available filter options (brands, categories, price range) |


---

## Configuration

The following parameters can be tuned at query time via the search interface:

| Parameter | Default | Description |
|---|---|---|
| `w_text` | 0.4 | Weight for text embedding similarity |
| `w_img` | 0.3 | Weight for image embedding similarity |
| `w_meta` | 0.3 | Weight for metadata feature similarity |
| `lambda` | 0.7 | MMR relevance-diversity tradeoff (1.0 = pure relevance) |
| `top_k` | 8 | Number of recommendations to return (max 50) |
| `candidate_k` | 300 | FAISS retrieval pool size before re-ranking |
| `category_boost` | 0.2 | Multiplicative boost for same-category items |
| `price_boost` | 0.15 | Multiplicative boost for price-similar items |
| `price_tolerance` | 0.3 | Allowed ±30% price deviation for price boost |

---

## License

This project is released under the MIT License.

---

#### Build with ❤️ by Vaibhav Rakshe!

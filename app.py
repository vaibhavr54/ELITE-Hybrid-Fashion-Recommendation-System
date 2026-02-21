from flask import Flask, render_template, request, jsonify
import pandas as pd
from recommender import hybrid_recommender_faiss, aligned_data

app = Flask(__name__)

# Load dataframe once
data = pd.read_pickle("pickels/16k_apperal_data_preprocessed")

@app.route("/")
def home():
    return render_template(
        "index.html",
        products=aligned_data.reset_index().to_dict(orient="records")
    )

@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        query_index = int(request.form["query_index"])
        top_k = int(request.form["top_k"])
        w_text = float(request.form["w_text"])
        w_img = float(request.form["w_img"])
        w_meta = float(request.form["w_meta"])
        lambda_param = float(request.form["lambda"])

        # Validate inputs
        if top_k < 1 or top_k > 50:
            top_k = 10
        
        # Normalize weights if they don't sum to 1
        total_weight = w_text + w_img + w_meta
        if total_weight > 0:
            w_text /= total_weight
            w_img /= total_weight
            w_meta /= total_weight

        results = hybrid_recommender_faiss(query_index, top_k, w_text, w_img, w_meta, lambda_param)

        recommended_products = aligned_data.iloc[results].to_dict(orient="records")

        return render_template(
            "results.html",
            query=aligned_data.iloc[query_index],
            recommendations=recommended_products,
            params={
                'w_text': w_text,
                'w_img': w_img,
                'w_meta': w_meta,
                'lambda': lambda_param,
                'top_k': top_k
            }
        )
    except Exception as e:
        return render_template(
            "error.html",
            error=str(e)
        )

if __name__ == "__main__":
    app.run(debug=True)
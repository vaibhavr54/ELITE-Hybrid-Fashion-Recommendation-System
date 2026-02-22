from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import json
import pandas as pd
import json
import os

from recommender import (
    hybrid_recommender_faiss, 
    aligned_data, 
    evaluator, 
    feedback_store,
    get_available_brands,
    get_available_categories,
    get_price_range
)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev_key")

# Load dataframe once
# data = pd.read_pickle("pickels/16k_apperal_data_preprocessed")

@app.route("/")
def home():
    """Main search interface"""
    return render_template(
        "index.html",
        products=aligned_data.reset_index().to_dict(orient="records"),
        brands=get_available_brands(),
        categories=get_available_categories(),
        price_range=get_price_range()
    )

@app.route("/recommend", methods=["POST"])
def recommend():
    """Generate recommendations with filtering and explanations"""
    try:
        query_index = int(request.form["query_index"])
        top_k = int(request.form["top_k"])
        w_text = float(request.form["w_text"])
        w_img = float(request.form["w_img"])
        w_meta = float(request.form["w_meta"])
        lambda_param = float(request.form["lambda"])

        # Validate inputs
        if top_k < 1 or top_k > 50:
            top_k = 8
        
        # Normalize weights
        total_weight = w_text + w_img + w_meta
        if total_weight > 0:
            w_text /= total_weight
            w_img /= total_weight
            w_meta /= total_weight
        
        # Filters
        filters = {}
        
        if request.form.get("price_min"):
            filters['price_min'] = float(request.form["price_min"])
        
        if request.form.get("price_max"):
            filters['price_max'] = float(request.form["price_max"])
        
        brands = request.form.getlist("brands")
        if brands:
            filters['brands'] = brands
        
        categories = request.form.getlist("categories")
        if categories:
            filters['categories'] = categories

        # Run recommender
        results, explanations = hybrid_recommender_faiss(
            query_index,
            top_k,
            w_text,
            w_img,
            w_meta,
            lambda_param,
            filters=filters if filters else None,
            return_explanations=True
        )

        # Log feedback
        feedback_store.log_interaction(
            query_index,
            results,
            'view',
            metadata={'filters': filters}
        )

                # Store in session
                # Convert results to pure Python ints
        clean_results = [int(x) for x in results]

        # Convert explanations safely
        clean_explanations = json.loads(json.dumps(explanations))

        # Convert params safely
        clean_params = json.loads(json.dumps({
            'w_text': float(w_text),
            'w_img': float(w_img),
            'w_meta': float(w_meta),
            'lambda': float(lambda_param),
            'top_k': int(top_k),
            'filters': filters
        }))

        session['query_index'] = int(query_index)
        session['results'] = clean_results
        session['explanations'] = clean_explanations
        session['params'] = clean_params

        # Redirect to GET route
        return redirect(url_for("show_results"))

    except Exception as e:
        print(f"Error in recommend: {e}")
        return render_template("error.html", error=str(e))
@app.route("/results")
def show_results():
    query_index = session.get('query_index')
    results = session.get('results')
    explanations = session.get('explanations')
    params = session.get('params')

    if query_index is None or results is None:
        return redirect(url_for("home"))

    recommended_products = aligned_data.iloc[results].to_dict(orient="records")

    return render_template(
        "results.html",
        query=aligned_data.iloc[query_index],
        recommendations=recommended_products,
        explanations=explanations,
        params=params
    )

@app.route("/metrics")
def metrics_dashboard():
    """Metrics dashboard showing model performance"""
    try:
        # Run evaluation if not recent
        if not evaluator.metrics_history or \
           len(evaluator.metrics_history) == 0:
            print("Running evaluation...")
            metrics = evaluator.evaluate_system(num_samples=50, k=10)
        else:
            metrics = evaluator.metrics_history[-1]['metrics']
        
        # Get feedback statistics
        feedback_stats = feedback_store.get_statistics()
        
        # Get catalog statistics
        catalog_stats = {
            'total_products': len(aligned_data),
            'total_brands': aligned_data['brand'].nunique(),
            'total_categories': aligned_data['product_type_name'].nunique(),
            'price_range': get_price_range()
        }
        
        return render_template(
            "metrics.html",
            metrics=metrics,
            feedback_stats=feedback_stats,
            catalog_stats=catalog_stats,
            metrics_history=evaluator.metrics_history
        )
    except Exception as e:
        print(f"Error in metrics: {e}")
        return render_template(
            "error.html",
            error=str(e)
        )

@app.route("/api/evaluate", methods=["POST"])
def api_evaluate():
    """API endpoint to run evaluation"""
    try:
        num_samples = int(request.json.get('num_samples', 100))
        k = int(request.json.get('k', 10))
        
        metrics = evaluator.evaluate_system(num_samples=num_samples, k=k)
        
        return jsonify({
            'success': True,
            'metrics': metrics
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route("/api/feedback", methods=["POST"])
def api_feedback():
    """API endpoint to collect user feedback"""
    try:
        data = request.json
        query_index = int(data['query_index'])
        recommended_index = int(data['recommended_index'])
        action = data['action']  # 'like', 'dislike', 'click'
        
        # Log feedback
        feedback_store.log_interaction(
            query_index,
            [recommended_index],
            action
        )
        
        return jsonify({
            'success': True,
            'message': 'Feedback recorded'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route("/api/filters")
def api_filters():
    """API endpoint to get available filter options"""
    try:
        return jsonify({
            'brands': get_available_brands(),
            'categories': get_available_categories(),
            'price_range': get_price_range()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
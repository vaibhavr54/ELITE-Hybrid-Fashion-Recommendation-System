import json
from recommender import RecommenderEvaluator

print("Initializing evaluator...")

evaluator = RecommenderEvaluator()

print("Running evaluation (this may take some time)...")

metrics = evaluator.evaluate_system(num_samples=50, k=10)

with open("metrics_data.json", "w") as f:
    json.dump(metrics, f)

print("Metrics saved successfully to metrics_data.json")
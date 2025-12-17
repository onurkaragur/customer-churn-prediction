import argparse
import json
import sys
from typing import Any, Dict

import pandas as pd

from app import model_loader


def _predict_from_json(model_name: str, payload: Dict[str, Any]):
	models = model_loader.load_models()
	preprocessor = model_loader.load_preprocessor()

	model = model_loader.get_model(models, model_name) if model_name else None
	if model is None:
		# fallback to random forest if available
		model = models.get("rf") or next(iter(models.values()), None)

	if model is None:
		raise RuntimeError("No model available. Please train and save a model first.")

	# payload can be single dict or list of dicts
	if isinstance(payload, dict):
		df = pd.DataFrame([payload])
	else:
		df = pd.DataFrame(payload)

	preds, probs = model_loader.predict(model, preprocessor, df)

	out = {"predictions": preds.tolist() if hasattr(preds, "tolist") else list(preds)}
	if probs is not None:
		out["probabilities"] = probs.tolist()
	return out


def serve(host: str = "127.0.0.1", port: int = 5000, model_name: str = ""):
	try:
		from flask import Flask, request, jsonify
	except Exception:
		print("Flask is not installed. Install Flask to run the API server.")
		sys.exit(1)

	app = Flask(__name__)

	models = model_loader.load_models()
	preprocessor = model_loader.load_preprocessor()

	@app.route("/health")
	def health():
		return jsonify({"status": "ok", "models_loaded": list(models.keys())})

	@app.route("/predict", methods=["POST"])
	def predict_route():
		payload = request.get_json()
		try:
			result = _predict_from_json(model_name, payload)
			return jsonify(result)
		except Exception as exc:
			return jsonify({"error": str(exc)}), 400

	print(f"Starting server on {host}:{port} (models: {list(models.keys())})")
	app.run(host=host, port=port)


def main():
	parser = argparse.ArgumentParser(description="App entrypoint for serving and simple predictions")
	parser.add_argument("--serve", action="store_true", help="Start Flask API server")
	parser.add_argument("--host", default="127.0.0.1")
	parser.add_argument("--port", type=int, default=5000)
	parser.add_argument("--model", default="", help="Model to use (logreg|rf|xgb)")
	parser.add_argument("--predict-json", help="Path to a JSON file with input data (object or list)")

	args = parser.parse_args()

	if args.serve:
		serve(args.host, args.port, args.model)
		return

	if args.predict_json:
		with open(args.predict_json, "r", encoding="utf-8") as f:
			payload = json.load(f)
		out = _predict_from_json(args.model, payload)
		print(json.dumps(out, indent=2))
		return

	parser.print_help()


if __name__ == "__main__":
	main()

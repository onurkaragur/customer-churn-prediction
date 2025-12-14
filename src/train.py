import os
import pickle
from pprint import pformat

from config import DATA_PROCESSED, MODEL_DIR, MODEL_LOGREG, MODEL_RF, MODEL_XGB

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

try:
	from xgboost import XGBClassifier
	_HAS_XGB = True
except Exception:
	_HAS_XGB = False


def _load_processed(path=DATA_PROCESSED):
	# If config points to an unexpected BASE_DIR, try a few sensible fallbacks
	if os.path.exists(path):
		with open(path, "rb") as f:
			return pickle.load(f)

	# try common project-relative locations from current working directory and parents
	filename = os.path.basename(path)
	candidates = []
	cwd = os.getcwd()
	for base in (cwd, os.path.dirname(cwd), os.path.dirname(os.path.dirname(cwd))):
		candidates.append(os.path.join(base, "data", "processed", filename))

	for candidate in candidates:
		if os.path.exists(candidate):
			with open(candidate, "rb") as f:
				return pickle.load(f)

	# last resort: raise the original error with helpful message
	raise FileNotFoundError(f"Processed data not found. Tried: {path} and {candidates}")


def _extract_data(data):
	# Notebook uses keys like X_train_resampled, X_test_processed, y_train_resampled, y_test
	# fallback to more basic keys produced by `preprocess.py`
	if isinstance(data, dict):
		if "X_train_resampled" in data and "y_train_resampled" in data:
			X_train = data["X_train_resampled"]
			y_train = data["y_train_resampled"]
		elif "X_train" in data and "y_train" in data:
			X_train = data["X_train"]
			y_train = data["y_train"]
		else:
			raise KeyError("Could not find training arrays in processed data pickle")

		if "X_test_processed" in data and "y_test" in data:
			X_test = data["X_test_processed"]
			y_test = data["y_test"]
		elif "X_test" in data and "y_test" in data:
			X_test = data["X_test"]
			y_test = data["y_test"]
		else:
			raise KeyError("Could not find test arrays in processed data pickle")

		return X_train, X_test, y_train, y_test
	else:
		raise TypeError("Processed data must be a dict-like object saved by preprocessing")


def _evaluate(y_true, y_pred, y_prob=None):
	stats = {
		"accuracy": accuracy_score(y_true, y_pred),
		"precision": precision_score(y_true, y_pred),
		"recall": recall_score(y_true, y_pred),
		"f1": f1_score(y_true, y_pred),
	}
	if y_prob is not None:
		try:
			stats["roc_auc"] = roc_auc_score(y_true, y_prob)
		except Exception:
			stats["roc_auc"] = None
	return stats


def train_and_save():
	os.makedirs(MODEL_DIR, exist_ok=True)

	print(f"Loading processed data from: {DATA_PROCESSED}")
	data = _load_processed()
	print("Processed data keys:\n", pformat(list(data.keys()) if isinstance(data, dict) else []))

	X_train, X_test, y_train, y_test = _extract_data(data)

	models = {}

	print("Training Logistic Regression...")
	logreg = LogisticRegression(max_iter=1000)
	logreg.fit(X_train, y_train)
	models[MODEL_LOGREG] = logreg

	print("Training Random Forest...")
	rf = RandomForestClassifier(n_estimators=200, random_state=42)
	rf.fit(X_train, y_train)
	models[MODEL_RF] = rf

	if _HAS_XGB:
		print("Training XGBoost...")
		xgb = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42, use_label_encoder=False, eval_metric="logloss")
		xgb.fit(X_train, y_train)
		models[MODEL_XGB] = xgb
	else:
		print("XGBoost not available; skipping XGB model.")

	print("Evaluating and saving models...")
	for path, model in models.items():
		y_pred = model.predict(X_test)
		y_prob = None
		try:
			y_prob = model.predict_proba(X_test)[:, 1]
		except Exception:
			pass

		stats = _evaluate(y_test, y_pred, y_prob)
		print(f"Model: {os.path.basename(path)} - Metrics: {stats}")

		with open(path, "wb") as f:
			pickle.dump(model, f)

	print("All done. Models saved to:")
	for p in models:
		print(" -", p)


if __name__ == "__main__":
	train_and_save()


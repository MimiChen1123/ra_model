import argparse
import os

import pandas as pd


def normalize_model_path(model_path):
    return model_path[:-4] if model_path.endswith(".pkl") else model_path


def load_data(input_path):
    data = pd.read_json(input_path)
    data = data.dropna().reset_index(drop=True)

    if "score" in data.columns:
        data["score"] = data["score"].replace({0: 2, 0.5: 2, 1: 2, 1.5: 2})

    return data


def add_score_prediction(task_type, prediction):
    if task_type == "classification":
        prediction["predicted_score"] = prediction["prediction_label"].astype(float)
    else:
        prediction["predicted_score"] = (
            prediction["prediction_label"]
            .clip(2, 5)
            .apply(lambda x: round(x * 2) / 2)
        )

    return prediction


def save_prediction(prediction, output_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if output_path.endswith(".jsonl"):
        prediction.to_json(output_path, orient="records", lines=True, force_ascii=False)
    elif output_path.endswith(".csv"):
        prediction.to_csv(output_path, index=False)
    else:
        prediction.to_json(output_path, orient="records", indent=2, force_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Run inference with a trained PyCaret essay scoring model.")
    parser.add_argument("--input_data_path", required=True, help="Path to input JSON data.")
    parser.add_argument("--model_path", required=True, help="Path to trained PyCaret model, with or without .pkl.")
    parser.add_argument("--output_path", required=True, help="Path to save predictions: .json, .jsonl, or .csv.")
    parser.add_argument("--type", required=True, choices=["classification", "regression"], help="Model task type.")
    args = parser.parse_args()

    data = load_data(args.input_data_path)

    if args.type == "classification":
        from pycaret.classification import ClassificationExperiment

        exp = ClassificationExperiment()
    else:
        from pycaret.regression import RegressionExperiment

        exp = RegressionExperiment()

    model = exp.load_model(normalize_model_path(args.model_path))
    prediction = exp.predict_model(model, data=data)
    prediction = add_score_prediction(args.type, prediction)
    save_prediction(prediction, args.output_path)

    print(f"Prediction output saved to {args.output_path}")


if __name__ == "__main__":
    main()

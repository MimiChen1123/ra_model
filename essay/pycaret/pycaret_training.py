import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import mean_squared_error

def evaluation(task, prediction):

    PASS_THRESHOLD = 4.0
    
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    error_ctr = {error/2: 0 for error in range(-6, 7)}
    
    if task == "classification":
        pred_labels = prediction["prediction_label"].apply(lambda x: float(x))   # prediction is represented as string type
        gt_labels = prediction["score"].apply(lambda x: float(x))     # score is represented as string type     
    else:
        prediction["prediction_label"] = prediction["prediction_label"].clip(2, 5)    # clip the prediction to between 0 ~ 5
        pred_labels = prediction["prediction_label"].apply(lambda x: round(x * 2) / 2)  # to the nearest score point
        # pred_labels = prediction["prediction_label"].apply(lambda x: np.floor(x * 2) / 2)  # to the nearest low score point
        gt_labels = prediction["score"]
        
    mse = mean_squared_error(gt_labels, pred_labels)

    for gt, pred in zip(gt_labels, pred_labels):
        pass_or_not = True if gt >= PASS_THRESHOLD else False
        
        if pass_or_not:
            if pred >= PASS_THRESHOLD:
                true_positive += 1
            else:
                false_negative += 1
        else:
            if pred >= PASS_THRESHOLD:
                false_positive += 1
            else:
                true_negative += 1

        error_ctr[pred - gt] += 1      
    
    return true_positive, true_negative, false_positive, false_negative, error_ctr, mse

def _plot_confusion_matrix_ax(ax, tp, tn, fp, fn):
    """
    Plots a confusion matrix on a given matplotlib Axes object.
    """
    PASS_THRESHOLD = 4.0
    # Matrix with TP in top-left
    matrix = np.array([[tp, fn],
                       [fp, tn]])
    
    total = matrix.sum()
    
    # Prepare annotations with counts and percentages
    if total > 0:
        annot = np.array([
            [f"{tp}\n({tp/total:.2%})", f"{fn}\n({fn/total:.2%})"],
            [f"{fp}\n({fp/total:.2%})", f"{tn}\n({tn/total:.2%})"]
        ])
        fmt = ''
    else:
        annot = True # Just show counts if total is zero
        fmt = 'd'

    # Rearrange labels to match the new matrix layout (Pass, Fail)
    labels = [f"Pass (>= {PASS_THRESHOLD})", f"Fail (< {PASS_THRESHOLD})"]
    
    sns.heatmap(matrix, annot=annot, fmt=fmt, cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                annot_kws={"size": 16}, ax=ax)
    
    ax.set_ylabel('Actual Condition')
    ax.set_xlabel('Predicted Condition')
    ax.xaxis.set_label_position('top') 
    ax.xaxis.tick_top()
    ax.set_title('Pass/Fail Confusion Matrix', fontsize=14, pad=20)

def _plot_error_distribution_ax(ax, error_ctr, mse):
    """
    Plots a bar chart for error distribution on a given matplotlib Axes object.
    """
    if not error_ctr:
        ax.text(0.5, 0.5, "No error data to plot.", ha='center', va='center')
        return

    # Filter out zero-count errors and sort by error value
    sorted_errors = sorted([(k, v) for k, v in error_ctr.items() if v > 0])
    
    if not sorted_errors:
        ax.text(0.5, 0.5, "No errors to plot.", ha='center', va='center')
    else:
        labels = [str(k) for k, v in sorted_errors]
        counts = [v for k, v in sorted_errors]

        bars = ax.bar(labels, counts, color=sns.color_palette('viridis', len(labels)))
        
        ax.set_ylabel('Count')
        ax.set_xlabel('Prediction Error (pred - gt)')
        ax.set_title('Error Distribution', fontsize=14, pad=20)
        ax.tick_params(axis='x', rotation=45)
        
        for label in ax.get_xticklabels():
            label.set_ha("right")

        max_count = max(counts)
        for bar in bars:
            yval = bar.get_height()
            if yval > 0:
                ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02 * max_count, f'{yval}', ha='center', va='bottom')

    # Display MSE on the plot
    ax.text(0.95, 0.95, f'MSE: {mse:.4f}', transform=ax.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

def plot_evaluation(tp, tn, fp, fn, error_ctr, mse, task, level, model_name="model"):
    """
    Creates and saves a single image containing both the confusion matrix and the error distribution bar chart.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Plot confusion matrix on the left subplot
    _plot_confusion_matrix_ax(ax1, tp, tn, fp, fn)

    # Plot error distribution on the right subplot
    _plot_error_distribution_ax(ax2, error_ctr, mse)

    # Add a title for the entire figure and adjust layout
    fig.suptitle(f'Evaluation Summary for {model_name} ({task})', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Ensure output directory exists
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f"{level}_{task}_{model_name}.png")
    
    plt.savefig(output_filename)
    print(f"Combined evaluation plot saved to {output_filename}")
    plt.close()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", type=str, help="Path to the input JSON file")
    parser.add_argument("--test_data_path", type=str, help="Path to the input JSON file")
    parser.add_argument("--model_output_path", type=str, help="Path to the output JSON file")
    parser.add_argument("--level", type=str, choices=["I", "HI"], help="Level of the task")
    parser.add_argument("--type", type=str, choices=["classification", "regression"], help="Type of the task")
    args = parser.parse_args()
    
    train_data = pd.read_json(args.train_data_path)   # load train data
    test_data = pd.read_json(args.test_data_path)   # load test data
    train_data = train_data.dropna().reset_index(drop=True)
    test_data = test_data.dropna().reset_index(drop=True)
    
    # Preprocess for low frequent score
    train_data["score"] = train_data["score"].replace({0: 2, 0.5: 2, 1: 2, 1.5: 2})
    test_data["score"] = test_data["score"].replace({0: 2, 0.5: 2, 1: 2, 1.5: 2})

    # setup
    if args.type=="classification":
        print(f"Task Type: Classification")
        from pycaret.classification import *
        exp = ClassificationExperiment()
        metric = "Accuracy"
        
        # Convert the float target score into string for multiclasses training
        target_value_counts = train_data["score"].value_counts()
        train_data["score"] = train_data["score"].replace({key: str(key) for key in target_value_counts.index})
        test_data["score"] = test_data["score"].replace({key: str(key) for key in target_value_counts.index})
    else:
        print(f"Task Type: Regression")
        from pycaret.regression import *
        exp = RegressionExperiment()
        metric = "MSE"

    s = exp.setup(
        # General setting
        data=train_data,
        target="score",
        session_id=42,
        train_size=0.8,
        # Preprocessing
        normalize=True,
        normalize_method="minmax",
        ignore_features=["document_id", "source_id", "subject", "content"],
        ordinal_features={"level":["I", "HI"], "cefr_prediction":["A1", "A2", "B1", "B2", "C1", "C2"]},
        numeric_features=["word_count", "RELEVANCE", "COHERENCE", "ORGANIZATION"],
        # Missing values
        imputation_type="simple",
        numeric_imputation="mean",
        categorical_imputation="mode",
        # Advanced
        feature_selection=False,
        remove_multicollinearity=False,
        pca=False,
        verbose=True,
        # Logging
        log_experiment=["wandb"],
        experiment_name=f"{args.type}_{args.level}",
        log_plots=False,
        log_profile=False,
        log_data=False,
    )
    
    # comprare models
    best = s.compare_models(n_select=5, sort=metric)
    comparison_grid = s.pull()
    
    # Evaluation
    model_output_dir = os.path.join(args.model_output_path, args.type, f"level_{args.level}")
    os.makedirs(model_output_dir, exist_ok=True)
    
    for idx, model in enumerate(best):
        prediction = s.predict_model(model, data=test_data)

        model_name = comparison_grid.index[idx]
        print(f"Model {idx} ({model_name}):\n{model}")
        
        tp, tn, fp, fn, error_ctr, mse = evaluation(args.type, prediction)
        plot_evaluation(tp, tn, fp, fn, error_ctr, mse, args.type, args.level, model_name=model_name)
        
        # Save Model
        s.save_model(model, os.path.join(model_output_dir, f"{model_name}"))
#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import os
import logging
import warnings
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from obf_class import OBFDataset
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import  RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier,  VotingClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay, log_loss, accuracy_score
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
import xgboost as xgb
import joblib
import time
from scipy.stats import uniform, randint
import cuml  # For GPU-accelerated sklearn algorithms
from cuml.linear_model import LogisticRegression as cuLogisticRegression
from cuml.svm import SVC

import cudf
import torch
results_dir = './results'
models_dir = './models'
plots_dir = './plots'

# ---- GPU-accelerated classifiers ----

def run_xgboost(X_train, y_train, X_test, y_test, n_iter=20, xgb_model=None):
    """Train and evaluate XGBoost using GPU acceleration"""
    logging.info("Training XGBoost classifier (GPU-accelerated)...")
    start_time = time.time()
    # Encode string labels into integers
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    # Parameter distribution for randomized search
    param_dist = {
        'n_estimators': randint(50, 300),
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.3),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'gamma': uniform(0, 0.5),
        'min_child_weight': randint(1, 10)
    }
    # Base XGBoost model with GPU acceleration
    xgb_model = xgb.XGBClassifier(
        tree_method='hist',
        device= 'cuda',# Use GPU
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42,
        early_stopping_rounds = 10,
        xgb_model = xgb_model
        
    )
    
    # Randomized search is much faster than grid search for large parameter spaces
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    random_search = RandomizedSearchCV(
        xgb_model, 
        param_distributions=param_dist,
        n_iter=n_iter,  # Number of parameter settings sampled
        cv=cv,
        scoring='accuracy',
        n_jobs=1,  # Use single process with GPU
        verbose=1,
        random_state=42,
        error_score='raise'  # Raise errors instead of masking them
    )
    
    # Fit the model 
    eval_set = [(X_train, y_train_encoded), (X_test, y_test_encoded)]
    random_search.fit(X_train, y_train_encoded, eval_set=eval_set)
        
    # Get best model
    best_model = random_search.best_estimator_

    # Extract training and evaluation loss history
    evals_result = best_model.evals_result()
    train_loss = evals_result['validation_0']['mlogloss']
    eval_loss = evals_result['validation_1']['mlogloss']

    # Plot the learning curve
    plot_loss_function(train_loss,eval_loss,'XGBoost model')

    
    # Evaluate the best model
    y_pred_encoded = best_model.predict(X_test)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)  # Convert predictions back to original labels
    y_proba = best_model.predict_proba(X_test)
    
    # Print results
    training_time = time.time() - start_time
    logging.info(f"XGBoost training completed in {training_time:.2f} seconds")
    logging.info(f"Best parameters: {random_search.best_params_}")
    logging.info(f"Best cross-validation score: {random_search.best_score_:.4f}")
    
    # Save model
    joblib.dump(best_model, os.path.join(models_dir, 'xgboost_best_model.pkl'))
    
    # Create evaluation visualizations
    create_evaluation_plots(y_test, y_pred, y_proba, best_model.classes_, "XGBoost")
    return {
        'model': best_model,
        'best_params': random_search.best_params_,
        'best_score': random_search.best_score_,
        'training_time': training_time,
        'eval_loss' : eval_loss,
        'train_loss' : train_loss
    }

def run_gpu_logistic_regression(X_train, y_train, X_test, y_test):
    """Train and evaluate Logistic Regression using RAPIDS cuML"""
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        logging.warning("CUDA not available. Skipping GPU Logistic Regression.")
        return None
    
    logging.info("Training Logistic Regression classifier (GPU-accelerated)...")
    start_time = time.time()
    
    try:
        # Encode y_train and y_test using LabelEncoder
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)

        
        # Convert data to cuDF for GPU processing
        X_train_gpu = cudf.DataFrame.from_pandas(X_train)
        y_train_gpu = cudf.Series(y_train_encoded.astype('int32'))
        X_test_gpu = cudf.DataFrame.from_pandas(X_test)
        
        # Parameter distribution for randomized search
        param_dist = {
            'C': np.logspace(-3, 3, 10),
            'max_iter': [1000, 2000, 3000],
            'penalty': ['l1', 'l2', 'none']
        }
        
        # Using cuML's LogisticRegression
        log_reg = cuLogisticRegression(solver='qn')

        # Lists to store training and evaluation loss
        train_loss = []
        eval_loss = []
        
        # We'll do a simplified parameter search since cuML doesn't have RandomizedSearchCV
        best_score = 0
        best_params = {}
        for C in param_dist['C']:
            for penalty in param_dist['penalty']:
                for max_iter in param_dist['max_iter']:
                    # Skip invalid combination
                    if penalty == 'none' and C != 1.0:
                        continue
                    
                    # Configure model
                    log_reg.C = C
                    log_reg.penalty = penalty
                    log_reg.max_iter = max_iter
                    
                    # Fit model
                    log_reg.fit(X_train_gpu, y_train_gpu)
                    
                    # Compute training loss
                    y_train_pred_proba = log_reg.predict_proba(X_train_gpu).to_numpy()
                    train_loss.append(log_loss(y_train_encoded, y_train_pred_proba))
                    
                    # Compute evaluation loss
                    y_test_pred_proba = log_reg.predict_proba(X_test_gpu).to_numpy()
                    eval_loss.append(log_loss(y_test_encoded, y_test_pred_proba))
                    # Evaluate
                    accuracy = log_reg.score(X_train_gpu, y_train_gpu)
                    
                    if accuracy > best_score:
                        best_score = accuracy
                        best_params = {'C': C, 'penalty': penalty, 'max_iter': max_iter}
        
        # Train with best parameters
        best_model = cuLogisticRegression(
            C=best_params['C'],
            penalty=best_params['penalty'],
            max_iter=best_params['max_iter'],
            solver='qn'
        )
        best_model.fit(X_train_gpu, y_train_gpu)
        
        # Convert predictions back to CPU for evaluation
        y_pred_gpu = best_model.predict(X_test_gpu)
        y_pred_encoded = y_pred_gpu.to_numpy()
        
        # Decode predictions back to original labels
        y_pred = label_encoder.inverse_transform(y_pred_encoded)
        
        # For probability, we need to get it from cuML and convert
        y_proba_gpu = best_model.predict_proba(X_test_gpu)
        y_proba = y_proba_gpu.to_numpy()
        
        # Print results
        training_time = time.time() - start_time
        logging.info(f"GPU Logistic Regression completed in {training_time:.2f} seconds")
        logging.info(f"Best parameters: {best_params}")
        logging.info(f"Best score: {best_score:.4f}")
        
        # Save model (need to convert to CPU model for portability)
        # For cuML models, serialization may be challenging, so you might need a different approach
        
        # Create evaluation visualizations
        class_names = list(set(y_train))
        create_evaluation_plots(y_test, y_pred, y_proba, class_names, "LogisticRegression_GPU")
        plot_loss_function(train_loss,eval_loss,'logistic regression')
        
        return {
            'model': best_model,
            'best_params': best_params,
            'best_score': best_score,
            'training_time': training_time,
            'y_pred': y_pred,  # Decoded predictions
            'y_proba': y_proba,  # Probabilities
            'train_loss': train_loss,  # Training loss history
            'eval_loss': eval_loss  # Evaluation loss history
        }
    except Exception as e:
        logging.error(f"Error during GPU Logistic Regression: {e}")
        return None

def run_random_forest(X_train, y_train, X_test, y_test):
    """Train and evaluate Random Forest (CPU-based since sklearn lacks GPU support)"""
    logging.info("Training Random Forest classifier...")
    start_time = time.time()
    
    # Parameter distribution for randomized search
    param_dist = {
        'n_estimators': randint(50, 300),
        'max_depth': randint(5, 20),
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 10),
        'max_features': ['sqrt', 'log2', None]
    }
    
    # Base Random Forest model
    rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)  # Use all CPU cores
    
    # Randomized search
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    random_search = RandomizedSearchCV(
        rf_model, 
        param_distributions=param_dist,
        n_iter=15,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,  # Use all CPU cores
        verbose=1,
        random_state=42
    )
    
    # Fit the model
    random_search.fit(X_train, y_train)
    
    # Get best model
    best_model = random_search.best_estimator_
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)
    
    # Print results
    training_time = time.time() - start_time
    logging.info(f"Random Forest training completed in {training_time:.2f} seconds")
    logging.info(f"Best parameters: {random_search.best_params_}")
    logging.info(f"Best cross-validation score: {random_search.best_score_:.4f}")
    
    # Save model
    joblib.dump(best_model, os.path.join(models_dir, 'randomforest_best_model.pkl'))
    
    # Create evaluation visualizations
    create_evaluation_plots(y_test, y_pred, y_proba, best_model.classes_, "RandomForest")
    
    return {
        'model': best_model,
        'best_params': random_search.best_params_,
        'best_score': random_search.best_score_,
        'training_time': training_time
    }




from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, VotingClassifier
# from sklearn.svm import SVC
from scipy.stats import randint, uniform

def train_ensemble_model(X_train, y_train, X_test, y_test, use_gpu: bool = True):
    """Train an ensemble model combining multiple classifiers with RandomizedSearchCV."""
    # Define individual models with their hyperparameter distributions for RandomizedSearchCV
    # Encode string labels into integers
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

        
    models = {
        'rf': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': randint(50, 200),
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': randint(2, 10),
                'min_samples_leaf': randint(1, 4)
            }
        },
        'xgb': {
            'model': xgb.XGBClassifier(tree_method='hist', device='cuda', random_state=42),
            'params': {
                'n_estimators': randint(50, 200),
                'max_depth': randint(3, 10),
                'learning_rate': uniform(0.01, 0.3),
                'subsample': uniform(0.6, 0.4),
                'colsample_bytree': uniform(0.6, 0.4)
            }
        },
        'gb': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': randint(50, 200),
                'max_depth': randint(3, 10),
                'learning_rate': uniform(0.01, 0.3),
                'subsample': uniform(0.6, 0.4)
            }
        },
        'hgb': {
            'model': HistGradientBoostingClassifier(random_state=42),
            'params': {
                'max_iter': randint(50, 200),
                'max_depth': randint(3, 10),
                'learning_rate': uniform(0.01, 0.3)
            }
        },
        'mlp': {
            'model': MLPClassifier(random_state=42),
            'params': {
                'hidden_layer_sizes': [(50,), (100,), (100, 50)],
                'alpha': uniform(0.0001, 0.01),
                'learning_rate': ['constant', 'adaptive']
            }
        },
        # 'svc': {   #takes a lot of time even with gpu
        #     'model': SVC(probability=True, random_state=42),
        #     'params': {
        #         'C': uniform(0.1, 10),
        #         'kernel': ['linear', 'rbf'],
        #         'gamma': ['scale', 'auto']
        #     },
        #     'n_iter': 5  # Limiter à 5 itérations pour SVC
        # },
        'lr': {
            'model': LogisticRegression(random_state=42),
            'params': {
                'C': uniform(0.1, 10),
                'penalty': ['l2'],
                'solver': ['lbfgs', 'liblinear']
            }
        }
    }

    # Train each model with RandomizedSearchCV
    trained_models = {}
    for name, config in models.items():
        logging.info(f"Training {name} with RandomizedSearchCV...")
        random_search = RandomizedSearchCV(
            config['model'],
            config['params'],
            n_iter=100,  # Number of parameter settings to sample
            cv=5,       # 5-fold cross-validation
            scoring='accuracy',
            random_state=42,
            n_jobs=-1   # Use all available cores
        )
        if name == 'xgb' and use_gpu:
            X_train_gpu = X_train.astype(np.float32)  # Ensure data type compatibility
            X_test_gpu = X_test.astype(np.float32)
            random_search.fit(X_train_gpu, y_train_encoded)
        else:
            random_search.fit(X_train, y_train_encoded)
        trained_models[name] = random_search.best_estimator_
        logging.info(f"Best parameters for {name}: {random_search.best_params_}")
        logging.info(f"Best cross-validation accuracy for {name}: {random_search.best_score_:.4f}")

    # Create the ensemble model
    ensemble = VotingClassifier(
        estimators=[(name, model) for name, model in trained_models.items()],
        voting='soft'  # Use soft voting for probability-based predictions
    )

    # Train the ensemble model
    ensemble.fit(X_train, y_train_encoded)

    # Evaluate the ensemble model
    y_pred_encoded = ensemble.predict(X_test)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)  # Convert predictions back to original labels
    y_proba = ensemble.predict_proba(X_test)

    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    # Create evaluation visualizations
    class_names = list(set(y_train))
    create_evaluation_plots(y_test, y_pred, y_proba, class_names, "Ensemble model")

    # Log results
    logging.info(f"Ensemble Model Accuracy: {accuracy:.4f}")
    logging.info(f"Ensemble Model ROC AUC: {roc_auc:.4f}")
    logging.info("Confusion Matrix:")
    logging.info(conf_matrix)
    logging.info("Classification Report:")
    logging.info(class_report)

    return ensemble, y_pred, y_proba

def create_evaluation_plots(y_true, y_pred, y_proba, class_names, model_name):
    """Create and save evaluation plots for a classifier"""
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'confusion_matrix_{model_name}.png'))
    plt.close()
    
    # Classification Report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).T
    # Save report to CSV
    report_df.to_csv(os.path.join(results_dir, f'classification_report_{model_name}.csv'))
    
    # ROC Curves (One-vs-Rest)
    if len(class_names) > 2:
        plt.figure(figsize=(10, 8))
        # Convert string labels to numeric for binarization
        label_binarizer = LabelBinarizer().fit(y_true)
        y_onehot = label_binarizer.transform(y_true)
        
        for i, class_name in enumerate(class_names):
            RocCurveDisplay.from_predictions(
                y_onehot[:, i],
                y_proba[:, i],
                name=f"{class_name}",
                ax=plt.gca()
            )
        
        plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
        plt.axis("square")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curves (One-vs-Rest) - {model_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'roc_curves_{model_name}.png'))
        plt.close()
    elif len(class_names) == 2:
        # Binary classification
        RocCurveDisplay.from_predictions(
            y_true == class_names[1],  # Convert to binary
            y_proba[:, 1],  # Probability of positive class
            name=f"{model_name}",
        )
        plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
        plt.axis("square")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {model_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'roc_curve_{model_name}.png'))
        plt.close()

def plot_loss_function(training_losses, validation_losses,model_name):
    # --- Plotting Training and Validation Loss ---
    plt.figure(figsize=(10, 5))
    plt.plot(training_losses, label='Training Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.savefig(f'./plots/loss_function_{model_name}.png')
    plt.show()

    
def compare_models(results):
    """Compare all models and create a summary visualization"""
    # Prepare data for comparison
    models = []
    accuracies = []
    training_times = []
    
    for name, result in results.items():
        if result is not None:
            models.append(name)
            accuracies.append(result['best_score'])
            training_times.append(result['training_time'])
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame({
        'Model': models,
        'Accuracy': accuracies,
        'Training Time (s)': training_times
    })
    
    # Save comparison to CSV
    comparison_df.to_csv(os.path.join(results_dir, 'model_comparison.csv'), index=False)
    
    # Create comparison plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Accuracy comparison
    sns.barplot(x='Model', y='Accuracy', data=comparison_df, ax=ax1)
    ax1.set_title('Model Accuracy Comparison')
    ax1.set_ylim(0, 1)  # Accuracy range
    ax1.set_ylabel('Accuracy')
    
    # Training time comparison
    sns.barplot(x='Model', y='Training Time (s)', data=comparison_df, ax=ax2)
    ax2.set_title('Model Training Time Comparison')
    ax2.set_ylabel('Training Time (seconds)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'model_comparison.png'))
    plt.close()
    
    # Print best model
    best_model = comparison_df.loc[comparison_df['Accuracy'].idxmax(), 'Model']
    logging.info(f"Best performing model: {best_model} with accuracy {comparison_df['Accuracy'].max():.4f}")
    
    return comparison_df


"""
Customer Churn Prediction using TensorFlow/Keras Neural Network
Author: Michael Obuma

This module implements a complete churn prediction system with data preprocessing,
model training, evaluation, and prediction capabilities.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any, Optional
import os
from config import ChurnPredictionConfig

# Try to import TensorFlow, fallback to sklearn if not available
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    print("TensorFlow Version:", tf.__version__)
    TF_AVAILABLE = True
except ImportError:
    print("TensorFlow not available, using sklearn MLPClassifier as fallback")
    from sklearn.neural_network import MLPClassifier
    TF_AVAILABLE = False


class ChurnDataProcessor:
    """Handles data loading, preprocessing, and splitting for churn prediction."""
    
    def __init__(self, config: ChurnPredictionConfig):
        self.config = config
        self.preprocessor = None
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load churn data from CSV file with error handling.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            DataFrame containing the loaded data
            
        Raises:
            FileNotFoundError: If the CSV file doesn't exist
            ValueError: If required columns are missing
        """
        try:
            df = pd.read_csv(file_path)
            print(f"Data loaded successfully. Shape: {df.shape}")
            
            # Validate required columns
            required_columns = ['Churn'] + self.config.categorical_features
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
                
            return df
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {file_path}")
        except Exception as e:
            raise ValueError(f"Error loading data: {str(e)}")
    
    def preprocess_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess features and target variable.
        
        Args:
            df: Raw dataframe
            
        Returns:
            Tuple of (processed_features, target_labels)
        """
        # Separate features and target
        X = df.drop('Churn', axis=1)
        y = df['Churn'].astype(int)  # Convert boolean to int
        
        # Identify numerical features automatically
        numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        print(f"Categorical features: {self.config.categorical_features}")
        print(f"Numerical features: {numerical_features}")
        
        # Create preprocessing pipeline
        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, self.config.categorical_features)
            ]
        )
        
        return X, y
    
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and testing sets.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test) processed arrays
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config.test_size, 
            random_state=self.config.random_state, 
            stratify=y
        )
        
        # Apply preprocessing
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)
        
        print(f"Training data shape: {X_train_processed.shape}")
        print(f"Testing data shape: {X_test_processed.shape}")
        print(f"Training target distribution: {np.bincount(y_train)}")
        print(f"Testing target distribution: {np.bincount(y_test)}")
        
        return X_train_processed, X_test_processed, y_train.values, y_test.values


class ChurnModelBuilder:
    """Builds and configures the neural network model."""
    
    def __init__(self, config: ChurnPredictionConfig):
        self.config = config
        
    def build_model(self, input_dim: int):
        """
        Build the neural network architecture.
        
        Args:
            input_dim: Number of input features
            
        Returns:
            Compiled model (TensorFlow or sklearn)
        """
        if TF_AVAILABLE:
            model = Sequential([
                # Input layer and first hidden layer
                Dense(self.config.hidden_layers[0], 
                      activation=self.config.activation, 
                      input_dim=input_dim),
                
                # Dropout for regularization
                Dropout(self.config.dropout_rate),
                
                # Second hidden layer
                Dense(self.config.hidden_layers[1], 
                      activation=self.config.activation),
                
                # Output layer for binary classification
                Dense(1, activation=self.config.output_activation)
            ])
            return model
        else:
            # Fallback to sklearn MLPClassifier
            model = MLPClassifier(
                hidden_layer_sizes=tuple(self.config.hidden_layers),
                activation=self.config.activation,
                max_iter=self.config.max_epochs,
                random_state=self.config.random_state,
                early_stopping=True,
                validation_fraction=0.2
            )
            return model
    
    def compile_model(self, model):
        """
        Compile the model with optimizer, loss, and metrics.
        
        Args:
            model: Uncompiled model
            
        Returns:
            Compiled model
        """
        if TF_AVAILABLE:
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
                loss='binary_crossentropy',
                metrics=['accuracy', 'precision', 'recall']
            )
        # sklearn models don't need compilation
        return model


class ChurnModelTrainer:
    """Handles model training with callbacks and monitoring."""
    
    def __init__(self, config: ChurnPredictionConfig):
        self.config = config
        
    def setup_callbacks(self) -> list:
        """Set up training callbacks."""
        if not TF_AVAILABLE:
            return []
            
        callbacks = []
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Model checkpointing
        if not os.path.exists('models'):
            os.makedirs('models')
            
        checkpoint = ModelCheckpoint(
            self.config.model_save_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        callbacks.append(checkpoint)
        
        return callbacks
    
    def train_model(self, model, X_train: np.ndarray, y_train: np.ndarray, 
                   X_test: np.ndarray, y_test: np.ndarray):
        """
        Train the model with validation monitoring.
        
        Args:
            model: Compiled model
            X_train: Training features
            y_train: Training labels
            X_test: Test features (used for validation)
            y_test: Test labels (used for validation)
            
        Returns:
            Training history or fitted model
        """
        print("\nStarting model training...")
        
        if TF_AVAILABLE:
            callbacks = self.setup_callbacks()
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=self.config.max_epochs,
                batch_size=self.config.batch_size,
                callbacks=callbacks,
                verbose=1
            )
            print("Model training completed.")
            return history
        else:
            # sklearn training
            model.fit(X_train, y_train)
            print("Model training completed.")
            return model


class ChurnModelEvaluator:
    """Evaluates model performance and generates visualizations."""
    
    def __init__(self, config: ChurnPredictionConfig):
        self.config = config
        
    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if TF_AVAILABLE:
            # TensorFlow model evaluation
            y_pred_prob = model.predict(X_test)
            y_pred = (y_pred_prob > self.config.classification_threshold).astype(int).flatten()
            
            # Calculate metrics
            test_loss, test_acc, test_precision, test_recall = model.evaluate(X_test, y_test, verbose=0)
            
            results = {
                'test_loss': test_loss,
                'test_accuracy': test_acc,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'predictions': y_pred,
                'prediction_probabilities': y_pred_prob.flatten()
            }
            
            print(f"\nTest Accuracy: {test_acc*100:.2f}%")
            print(f"Test Loss: {test_loss:.4f}")
            print(f"Test Precision: {test_precision:.4f}")
            print(f"Test Recall: {test_recall:.4f}")
            
        else:
            # sklearn model evaluation
            y_pred = model.predict(X_test)
            y_pred_prob = model.predict_proba(X_test)[:, 1]  # Get probability for class 1
            
            accuracy = accuracy_score(y_test, y_pred)
            
            results = {
                'test_accuracy': accuracy,
                'predictions': y_pred,
                'prediction_probabilities': y_pred_prob
            }
            
            print(f"\nTest Accuracy: {accuracy*100:.2f}%")
        
        return results
    
    def plot_training_history(self, history):
        """Plot training and validation metrics."""
        if TF_AVAILABLE and hasattr(history, 'history'):
            history_df = pd.DataFrame(history.history)
            
            plt.figure(figsize=(12, 4))
            
            # Plot accuracy
            plt.subplot(1, 2, 1)
            plt.plot(history_df['accuracy'], label='Train Accuracy')
            plt.plot(history_df['val_accuracy'], label='Validation Accuracy')
            plt.title('Model Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend()
            plt.grid(True)
            
            # Plot loss
            plt.subplot(1, 2, 2)
            plt.plot(history_df['loss'], label='Train Loss')
            plt.plot(history_df['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.show()
        else:
            print("Training history not available for sklearn models")
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Churn', 'Churn'],
                   yticklabels=['No Churn', 'Churn'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
    
    def generate_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """Generate detailed classification report."""
        report = classification_report(y_true, y_pred, 
                                     target_names=['No Churn', 'Churn'])
        print("\nClassification Report:")
        print(report)
        return report


class ChurnPredictionPipeline:
    """Main pipeline orchestrator for the churn prediction system."""
    
    def __init__(self, config: Optional[ChurnPredictionConfig] = None):
        self.config = config or ChurnPredictionConfig()
        self.data_processor = ChurnDataProcessor(self.config)
        self.model_builder = ChurnModelBuilder(self.config)
        self.trainer = ChurnModelTrainer(self.config)
        self.evaluator = ChurnModelEvaluator(self.config)
        
    def run_full_pipeline(self, data_path: str) -> Dict[str, Any]:
        """
        Execute the complete churn prediction pipeline.
        
        Args:
            data_path: Path to the data file
            
        Returns:
            Dictionary containing results and trained model
        """
        print("=== Starting Churn Prediction Pipeline ===\n")
        
        # 1. Load and preprocess data
        print("1. Loading and preprocessing data...")
        df = self.data_processor.load_data(data_path)
        X, y = self.data_processor.preprocess_features(df)
        X_train, X_test, y_train, y_test = self.data_processor.split_data(X, y)
        
        # 2. Build and compile model
        print("\n2. Building model...")
        input_dim = X_train.shape[1]
        model = self.model_builder.build_model(input_dim)
        model = self.model_builder.compile_model(model)
        
        print("\nModel Architecture:")
        if TF_AVAILABLE:
            model.summary()
        else:
            print(f"MLPClassifier with hidden layers: {self.config.hidden_layers}")
            print(f"Activation: {self.config.activation}")
            print(f"Max iterations: {self.config.max_epochs}")
            print(f"Input features: {input_dim}")
        
        # 3. Train model
        print("\n3. Training model...")
        history = self.trainer.train_model(model, X_train, y_train, X_test, y_test)
        
        # 4. Evaluate model
        print("\n4. Evaluating model...")
        results = self.evaluator.evaluate_model(model, X_test, y_test)
        
        # 5. Generate visualizations
        print("\n5. Generating visualizations...")
        self.evaluator.plot_training_history(history)
        self.evaluator.plot_confusion_matrix(y_test, results['predictions'])
        self.evaluator.generate_classification_report(y_test, results['predictions'])
        
        # Package results
        pipeline_results = {
            'model': model,
            'history': history,
            'evaluation_results': results,
            'data_processor': self.data_processor,
            'config': self.config
        }
        
        print("\n=== Pipeline Completed Successfully ===")
        return pipeline_results


def main():
    """Main execution function."""
    # Initialize configuration
    config = ChurnPredictionConfig()
    
    # Create and run pipeline
    pipeline = ChurnPredictionPipeline(config)
    results = pipeline.run_full_pipeline('../data/churn-bigml-80.csv')
    
    print(f"\nFinal Test Accuracy: {results['evaluation_results']['test_accuracy']*100:.2f}%")


if __name__ == "__main__":
    main()
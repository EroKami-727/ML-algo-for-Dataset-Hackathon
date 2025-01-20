# Set environment variables to suppress TensorFlow warnings
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import all required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras import Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from imblearn.over_sampling import SMOTE
import shap
import seaborn as sns
import matplotlib.pyplot as plt

def analyze_errors(data, y_test, y_pred_classes, label_encoders):
    """Analyze error patterns and create a detailed report"""
    error_report = []
    error_report.append("ERROR ANALYSIS REPORT")
    error_report.append("=" * 50 + "\n")

    try:
        # 1. Overall Error Rate
        error_rate = (y_test != y_pred_classes).mean() * 100
        error_report.append(f"Overall Error Rate: {error_rate:.2f}%\n")

        # 2. Error Distribution by Type
        error_report.append("Error Distribution by Type:")
        error_report.append("-" * 30)
        
        # Convert numeric labels back to original error types
        # Convert to int before using inverse_transform
        y_test_int = y_test.astype(int)
        y_pred_int = y_pred_classes.astype(int)
        
        try:
            true_labels = label_encoders['error_type'].inverse_transform(y_test_int)
            pred_labels = label_encoders['error_type'].inverse_transform(y_pred_int)
        except Exception as e:
            error_report.append(f"Warning: Could not convert labels back to original types: {str(e)}")
            true_labels = y_test_int
            pred_labels = y_pred_int
        
        error_indices = np.where(y_test != y_pred_classes)[0]
        
        # Create DataFrame with error patterns
        error_types = pd.DataFrame({
            'True Error': true_labels[error_indices],
            'Predicted Error': pred_labels[error_indices]
        })
        
        if len(error_types) > 0:
            error_counts = error_types.groupby(['True Error', 'Predicted Error']).size().reset_index(name='count')
            error_counts = error_counts.sort_values('count', ascending=False)
            
            for _, row in error_counts.iterrows():
                error_report.append(f"True: {row['True Error']} → Predicted: {row['Predicted Error']}: {row['count']} times")
        else:
            error_report.append("No misclassifications found.")
        
        error_report.append("\n")

        # 3. Most Problematic Error Types
        error_report.append("Most Frequently Misclassified Error Types:")
        error_report.append("-" * 40)
        if len(error_types) > 0:
            problem_types = error_types['True Error'].value_counts()
            for error_type, count in problem_types.items():
                error_report.append(f"{error_type}: {count} misclassifications")
        else:
            error_report.append("No problematic error types found.")
        
        error_report.append("\n")

        # 4. Time-based Analysis
        error_report.append("Temporal Error Patterns:")
        error_report.append("-" * 25)
        
        try:
            # Safely convert time_in_seconds to datetime
            data['hour'] = pd.to_datetime(data['time_in_seconds'].astype(float), unit='s').dt.hour
            hourly_errors = data.groupby('hour')['error_type'].count()
            peak_hour = hourly_errors.idxmax()
            error_report.append(f"Peak Error Hour: {peak_hour:02d}:00")
            error_report.append(f"Error Count at Peak: {hourly_errors[peak_hour]}")
        except Exception as e:
            error_report.append(f"Could not analyze temporal patterns: {str(e)}")
        
        error_report.append("\n")

        # 5. Source-Destination Analysis
        error_report.append("Source-Destination Error Patterns:")
        error_report.append("-" * 35)
        
        try:
            problematic_routes = data.groupby(['Source', 'Destination'])['error_type'].count().sort_values(ascending=False).head(5)
            for (source, dest), count in problematic_routes.items():
                try:
                    source_name = label_encoders['Source'].inverse_transform([int(source)])[0]
                    dest_name = label_encoders['Destination'].inverse_transform([int(dest)])[0]
                    error_report.append(f"Route {source_name} → {dest_name}: {count} errors")
                except:
                    error_report.append(f"Route {source} → {dest}: {count} errors")
        except Exception as e:
            error_report.append(f"Could not analyze source-destination patterns: {str(e)}")
        
        error_report.append("\n")

        # 6. Protocol Analysis
        error_report.append("Protocol-specific Error Patterns:")
        error_report.append("-" * 35)
        
        try:
            protocol_errors = data.groupby('Protocol')['error_type'].count().sort_values(ascending=False)
            for protocol, count in protocol_errors.items():
                try:
                    protocol_name = label_encoders['Protocol'].inverse_transform([int(protocol)])[0]
                    error_report.append(f"Protocol {protocol_name}: {count} errors")
                except:
                    error_report.append(f"Protocol {protocol}: {count} errors")
        except Exception as e:
            error_report.append(f"Could not analyze protocol patterns: {str(e)}")
        
        error_report.append("\n")

        # 7. Recommendations
        error_report.append("RECOMMENDATIONS")
        error_report.append("=" * 20)
        
        if error_rate > 20:
            error_report.append("- High overall error rate suggests need for system-wide review")
        
        if len(error_types) > 0:
            top_error_type = problem_types.index[0]
            error_report.append(f"- Focus on improving detection of '{top_error_type}' errors")
        
        try:
            if peak_hour in range(9, 18):  # Business hours
                error_report.append("- Peak errors during business hours - consider load balancing")
            else:
                error_report.append("- Peak errors during off-hours - review maintenance schedules")
        except:
            error_report.append("- Unable to determine peak error times - review time logging")

    except Exception as e:
        error_report.append(f"\nError during analysis: {str(e)}")
        error_report.append("Please check data types and values in the input data.")

    return "\n".join(error_report)
def save_plot(plt, filename, data_dir):
    """Helper function to save plots"""
    try:
        filepath = os.path.join(data_dir, filename)
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Successfully saved plot to: {filepath}")
    except Exception as e:
        print(f"Error saving plot {filename}: {str(e)}")

def main():
    # Define file paths
    data_dir = r'C:\mnt\data'
    input_file = os.path.join(data_dir, 'merged_log_node_data.csv')
    output_file = os.path.join(data_dir, 'processed_logs_with_clusters.csv')
    shap_output_file = os.path.join(data_dir, 'shap_values.npy')
    model_file = os.path.join(data_dir, 'best_model.keras')

    # Verify input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found at: {input_file}")

    print(f"Using data directory: {data_dir}")
    print(f"Input file path: {input_file}")

    try:
        # Load Data
        print("Loading and preprocessing data...")
        data = pd.read_csv(input_file)
        
        # Handle missing values
        if data.isnull().values.any():
            data = data.fillna(method='ffill')

        # Feature Engineering
        data['time_diff'] = data['time_in_seconds'].diff().fillna(0)
        data['error_count'] = data.groupby(['Source', 'Destination'])['error_type'].transform('count')
        data['source_error_rate'] = data.groupby('Source')['error_type'].transform('count') / len(data)
        data['dest_error_rate'] = data.groupby('Destination')['error_type'].transform('count') / len(data)

        # Preprocessing
        label_encoders = {}
        scaler = MinMaxScaler()

        # Encode categorical features
        categorical_columns = ['Source', 'Destination', 'Protocol', 'error_level', 'error_type']
        for col in categorical_columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            label_encoders[col] = le

        # Define features for sequence generation
        feature_columns = categorical_columns + ['Length', 'time_in_seconds', 'time_diff', 
                                              'error_count', 'source_error_rate', 'dest_error_rate']
        
        # Generate sequences
        print("Generating sequences...")
        sequence_length = 5
        sequences = []
        targets = []
        
        data_array = data[feature_columns].values
        for i in range(len(data) - sequence_length):
            sequences.append(data_array[i:i+sequence_length])
            targets.append(data_array[i+sequence_length, feature_columns.index('error_type')])

        sequences = np.array(sequences)
        targets = np.array(targets)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            sequences, targets, test_size=0.2, random_state=42, stratify=targets
        )

        # Balance training data using SMOTE
        print("Balancing dataset...")
        smote = SMOTE(random_state=42)
        X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_reshaped, y_train)
        X_train_balanced = X_train_balanced.reshape(-1, sequence_length, X_train.shape[2])

        # Normalize numerical features
        numerical_columns = ['Length', 'time_in_seconds', 'time_diff', 'error_count', 
                            'source_error_rate', 'dest_error_rate']
        numerical_indices = [feature_columns.index(col) for col in numerical_columns]
        
        for i in numerical_indices:
            scaler = MinMaxScaler()
            X_train_balanced[:, :, i] = scaler.fit_transform(X_train_balanced[:, :, i].reshape(-1, 1)).reshape(X_train_balanced.shape[0], -1)
            X_test[:, :, i] = scaler.transform(X_test[:, :, i].reshape(-1, 1)).reshape(X_test.shape[0], -1)

        # Calculate class weights
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = dict(enumerate(class_weights))

        # Build model
        print("Building and training model...")
        num_classes = len(np.unique(targets))
        model = Sequential([
            Input(shape=(sequence_length, len(feature_columns))),
            LSTM(128, return_sequences=True),
            Dropout(0.3),
            LSTM(64),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Training callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(model_file, monitor='val_loss', save_best_only=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
        ]

        # Train model
        print("Training model...")
        history = model.fit(
            X_train_balanced, y_train_balanced,
            validation_data=(X_test, y_test),
            epochs=50,
            batch_size=16,
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=1
        )

        # Evaluate model
        print("\nEvaluating model...")
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_classes))

        # Plot confusion matrix
        print("\nGenerating confusion matrix plot...")
        plt.figure(figsize=(12, 8))
        cm = confusion_matrix(y_test, y_pred_classes)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        save_plot(plt, 'confusion_matrix.png', data_dir)

        # Plot training history
        print("\nGenerating training history plot...")
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        save_plot(plt, 'training_history.png', data_dir)

        # SHAP Analysis
        print("\nCalculating SHAP values...")
        try:
            background = X_train_balanced[np.random.choice(X_train_balanced.shape[0], 100, replace=False)]
            explainer = shap.DeepExplainer(model, background)
            shap_values = explainer.shap_values(X_test[:100])
            np.save(shap_output_file, shap_values)
            print(f"Successfully saved SHAP values to: {shap_output_file}")

            # Feature Importance Analysis
            print("\nGenerating feature importance plot...")
            feature_importance = np.abs(np.array(shap_values)).mean(axis=0)
            importance_df = pd.DataFrame({
                'feature': feature_columns,
                'importance': feature_importance.mean(axis=0)
            })
            importance_df = importance_df.sort_values('importance', ascending=False)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(data=importance_df, x='importance', y='feature')
            plt.title('Feature Importance')
            plt.tight_layout()
            save_plot(plt, 'feature_importance.png', data_dir)
            
            print("\nFeature Importance:")
            print(importance_df)
        except Exception as e:
            print(f"Error in SHAP/Feature Importance analysis: {str(e)}")

        # Save processed data
        data.to_csv(output_file, index=False)
        print(f"\nProcessed data saved to: {output_file}")

        print("\nAnalysis completed!")
        print("\nFiles generated in", data_dir + ":")
        print("- processed_logs_with_clusters.csv")
        print("- shap_values.npy")
        print("- best_model.keras")
        print("- confusion_matrix.png")
        print("- training_history.png")
        print("- feature_importance.png")
         # Generate and save error analysis report
        print("\nGenerating error analysis report...")
        error_analysis = analyze_errors(data, y_test, y_pred_classes, label_encoders)
        
        # Save the error analysis report
        report_file = os.path.join(data_dir, 'error_analysis_report.txt')
        with open(report_file, 'w') as f:
            f.write(error_analysis)
        print(f"Error analysis report saved to: {report_file}")

    except Exception as e:
        print(f"An error occurred in main execution: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "_main_":
    main()
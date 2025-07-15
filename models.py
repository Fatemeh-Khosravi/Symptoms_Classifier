import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import VotingClassifier

class SymptomClassifier:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_score = 0
        
    def load_data(self):
        # Load your existing data
        data = pd.read_csv('Training.csv')
        X = data.drop('prognosis', axis=1)
        y = data['prognosis']
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    def initialize_models(self):
        """Initialize multiple ML algorithms"""
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Naive Bayes': GaussianNB(),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42)
        }
        self.models['XGBoost'] = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        self.models['LightGBM'] = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """Train all models and compare performance"""
        results = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred
            }
            
            print(f"{name}: Accuracy = {accuracy:.4f}, CV = {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return results
    
    def select_best_model(self, results):
        """Select the best performing model"""
        best_name = max(results.keys(), key=lambda k: results[k]['cv_mean'])
        self.best_model = results[best_name]['model']
        self.best_score = results[best_name]['cv_mean']
        
        print(f"\nBest Model: {best_name}")
        print(f"Cross-validation Score: {self.best_score:.4f}")
        
        return best_name
    
    def save_models(self, results, best_name):
        """Save all models and results"""
        # Save best model
        joblib.dump(self.best_model, 'best_model.pkl')
        
        # Save all models
        for name, result in results.items():
            joblib.dump(result['model'], f'models/{name.lower().replace(" ", "_")}.pkl')
        
        # Save results summary
        summary = {name: {'accuracy': result['accuracy'], 
                         'cv_mean': result['cv_mean'], 
                         'cv_std': result['cv_std']} 
                  for name, result in results.items()}
        
        pd.DataFrame(summary).T.to_csv('model_comparison_results.csv')
    
    def hyperparameter_tuning(self, X_train, y_train):
        """Perform hyperparameter tuning for best models"""
        
        # Define parameter grids
        param_grids = {
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            },
            'XGBoost': {
                'n_estimators': [100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2]
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.1, 0.01]
            }
        }
        
        tuned_models = {}
        
        for name, model in self.models.items():
            if name in param_grids:
                print(f"Tuning {name}...")
                
                grid_search = GridSearchCV(
                    model, 
                    param_grids[name], 
                    cv=5, 
                    scoring='accuracy',
                    n_jobs=-1
                )
                
                grid_search.fit(X_train, y_train)
                tuned_models[name] = grid_search.best_estimator_
                
                print(f"Best parameters: {grid_search.best_params_}")
                print(f"Best score: {grid_search.best_score_:.4f}")
        
        return tuned_models

    def create_ensemble_model(self, results, top_n=3):
        """Create ensemble from top N models"""
        # Get top N models
        top_models = sorted(results.items(), 
                           key=lambda x: x[1]['cv_mean'], 
                           reverse=True)[:top_n]
        
        estimators = [(name, result['model']) for name, result in top_models]
        
        # Create voting classifier
        ensemble = VotingClassifier(estimators=estimators, voting='soft')
        
        return ensemble

def generate_comparison_report(results, X_test, y_test):
    """Generate detailed comparison report"""
    report_data = []
    
    for name, result in results.items():
        # Classification report
        class_report = classification_report(y_test, result['predictions'], output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, result['predictions'])
        
        report_data.append({
            'Model': name,
            'Accuracy': result['accuracy'],
            'CV_Mean': result['cv_mean'],
            'CV_Std': result['cv_std'],
            'Precision': class_report['weighted avg']['precision'],
            'Recall': class_report['weighted avg']['recall'],
            'F1-Score': class_report['weighted avg']['f1-score']
        })
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(report_data)
    comparison_df = comparison_df.sort_values('CV_Mean', ascending=False)
    
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)
    print(comparison_df.to_string(index=False))
    
    # Save detailed report
    comparison_df.to_csv('detailed_model_comparison.csv', index=False)
    
    return comparison_df

def load_all_models(self):
    """Load all trained models"""
    import os
    
    models = {}
    models_dir = 'models/'
    
    if os.path.exists(models_dir):
        for file in os.listdir(models_dir):
            if file.endswith('.pkl'):
                name = file.replace('.pkl', '').replace('_', ' ').title()
                model_path = os.path.join(models_dir, file)
                models[name] = joblib.load(model_path)
    
    return models

def predict_with_all_models(self, symptoms, models):
    """Get predictions from all models"""
    predictions = {}
    
    for name, model in models.items():
        pred = model.predict([symptoms])[0]
        prob = model.predict_proba([symptoms])[0]
        predictions[name] = {
            'prediction': pred,
            'confidence': max(prob)
        }
    
    return predictions

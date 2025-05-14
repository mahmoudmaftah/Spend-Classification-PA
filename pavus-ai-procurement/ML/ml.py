"""
Procurement Data Classification System

This script trains and evaluates multiple machine learning models for classifying 
procurement data into hierarchical category levels.

Features:
- Advanced text processing and feature engineering
- Multiple classification models with performance comparison
- Dimensionality reduction and visualization
- Clustering algorithms for data exploration
- Support for incremental learning with user feedback

Usage:
    python procurement_classifier.py --data path/to/data.csv --output path/to/output
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import re
import os
import pickle
import logging
from typing import Dict, List, Tuple, Union, Optional, Any
from datetime import datetime
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Text processing
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Machine learning
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, 
    f1_score, precision_score, recall_score, roc_auc_score
)

# Clustering
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

# Dimensionality reduction
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP not available. Install with 'pip install umap-learn' for additional dimensionality reduction.")

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')


class ProcurementClassifier:
    """
    A comprehensive system for classifying procurement data using
    machine learning models with feature engineering, model comparison,
    and incremental learning capabilities.
    """
    
    def __init__(self, data_path: str, output_dir: str = 'output'):
        """
        Initialize the classifier with data path and output directory.
        
        Args:
            data_path: Path to the CSV file containing procurement data
            output_dir: Directory to save outputs (models, plots, etc.)
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_trains = {}  # Dictionary to store target variables for each category level
        self.y_tests = {}
        self.models = {}  # Dictionary to store trained models for each category level
        self.encoders = {}  # Dictionary to store label encoders
        self.vectorizers = {}  # Dictionary to store text vectorizers
        self.feature_names = []  # Names of features after engineering
        self.train_indices = None
        self.test_indices = None
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Configure logging
        self._setup_logging()
        
        self.logger.info("ProcurementClassifier initialized")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        self.logger = logging.getLogger('procurement_classifier')
        self.logger.setLevel(logging.INFO)
        
        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler(os.path.join(self.output_dir, 'classifier.log'))
        
        # Create formatters
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(formatter)
        f_handler.setFormatter(formatter)
        
        # Clear any existing handlers
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Add handlers to logger
        self.logger.addHandler(c_handler)
        self.logger.addHandler(f_handler)
    
    def load_data(self) -> pd.DataFrame:
        """
        Load and perform initial processing of the procurement data.
        
        Returns:
            Processed DataFrame
        """
        self.logger.info(f"Loading data from {self.data_path}")
        
        try:
            # Load data
            self.data = pd.read_csv(self.data_path)
            
            # Check for required columns
            required_cols = [
                'Item Number', 'Item Description', 'clean_description', 
                'extracted_keywords', 'category_l1', 'category_l2', 'category_l3',
                'attributes', 'specifications', 'primary_function'
            ]
            
            missing_cols = [col for col in required_cols if col not in self.data.columns]
            if missing_cols:
                self.logger.warning(f"Missing required columns: {missing_cols}")
            
            # Check for target columns
            target_cols = [
                'category_l1_name', 'category_l2_name', 'category_l3_name',
                'category_l4_name', 'category_l5_name'
            ]
            
            missing_targets = [col for col in target_cols if col not in self.data.columns]
            if missing_targets:
                self.logger.warning(f"Missing target columns: {missing_targets}")
            
            # Basic info about the data
            self.logger.info(f"Loaded data with {len(self.data)} rows and {len(self.data.columns)} columns")
            self.logger.info(f"Column names: {list(self.data.columns)}")
            
            # Parse JSON fields
            for col in ['attributes', 'specifications', 'extracted_keywords']:
                if col in self.data.columns:
                    self.data[col] = self.data[col].apply(self._parse_json_field)
            
            return self.data
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _parse_json_field(self, field):
        """Parse JSON fields that might be strings."""
        if pd.isna(field) or field == '':
            return []
        
        try:
            if isinstance(field, str):
                # Try to parse JSON string
                return json.loads(field)
            return field
        except json.JSONDecodeError:
            # If it's not valid JSON, try to parse it as a string representation of a list
            if isinstance(field, str) and field.startswith('[') and field.endswith(']'):
                # Remove brackets and split by commas
                items = field[1:-1].split(',')
                # Remove quotes and trim whitespace
                return [item.strip().strip('"\'') for item in items]
            return []
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text data by tokenizing, removing stopwords, and lemmatizing.
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Preprocessed text
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits (keep spaces)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words and len(token) > 1]
        
        # Lemmatize
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        
        # Join tokens back into a string
        return ' '.join(tokens)
    
    def engineer_features(self, do_train_test_split: bool = True, test_size: float = 0.25) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform feature engineering on the dataset.
        
        Args:
            do_train_test_split: Whether to split data into train/test sets
            test_size: Proportion of data to use for testing
            
        Returns:
            Tuple of (X_train, X_test) or (X, None) if no split
        """
        if self.data is None:
            self.logger.error("No data loaded. Call load_data() first.")
            raise ValueError("No data loaded")
        
        self.logger.info("Starting feature engineering")
        
        # Create a copy to avoid modifying the original
        data = self.data.copy()
        
        # Preprocess text columns
        text_cols = ['Item Description', 'clean_description', 'primary_function']
        for col in text_cols:
            if col in data.columns:
                self.logger.info(f"Preprocessing text column: {col}")
                data[f'{col}_processed'] = data[col].apply(self.preprocess_text)
        
        # Extract features from JSON fields
        self._extract_json_features(data)
        
        # Create numerical features from text
        self._create_text_features(data)
        
        # Extract n-grams from product descriptions
        self._extract_ngrams(data)
        
        # Prepare target variables
        target_cols = [
            'category_l1_name', 'category_l2_name', 'category_l3_name',
            'category_l4_name', 'category_l5_name'
        ]
        
        # Encode target variables
        for col in target_cols:
            if col in data.columns:
                self.logger.info(f"Encoding target variable: {col}")
                self.encoders[col] = LabelEncoder()
                data[f'{col}_encoded'] = self.encoders[col].fit_transform(data[col].fillna('Unknown'))
        
        # Identify feature columns
        feature_cols = [
            'word_count'
        ]
        
        # Add any additional feature columns created during engineering
        additional_feature_cols = [
            col for col in data.columns 
            if col.startswith('num_') or col.startswith('has_') or 
            col.startswith('feature_') or col.startswith('ngram_')
        ]
        feature_cols.extend(additional_feature_cols)
        
        # Ensure all feature columns are numeric
        for col in feature_cols:
            if col in data.columns:
                # Try to convert to numeric, set NaN if not convertible
                data[col] = pd.to_numeric(data[col], errors='coerce')
                # Fill NaN with 0
                data[col] = data[col].fillna(0)
        
        # NOTE: We remove text columns from direct features to avoid the string to float error
        # Text features will be handled through vectorization later
        
        # Remove any columns that don't exist
        feature_cols = [col for col in feature_cols if col in data.columns]
        
        # Store feature names
        self.feature_names = feature_cols.copy()
        
        X = data[feature_cols]
        
        # Create target variables dictionary
        y_dict = {}
        for col in target_cols:
            if f'{col}_encoded' in data.columns:
                y_dict[col] = data[f'{col}_encoded']
        
        if do_train_test_split:
            # Split data into train and test sets
            self.logger.info(f"Splitting data into train ({1-test_size:.1%}) and test ({test_size:.1%}) sets")
            
            # Use stratified split if possible
            stratify = None
            if 'category_l1_name_encoded' in data.columns:
                stratify = data['category_l1_name_encoded']
            
            # Split data
            X_train, X_test, y_train_dict, y_test_dict = {}, {}, {}, {}
            

            #####
            # Use stratified split if possible
            stratify = None
            if 'category_l1_name_encoded' in data.columns:
                # Check if all classes have at least 2 samples before using stratification
                value_counts = data['category_l1_name_encoded'].value_counts()
                if value_counts.min() >= 2:
                    stratify = data['category_l1_name_encoded']
                else:
                    self.logger.warning(f"Some classes have fewer than 2 samples. Using random split instead of stratified split.")

            # Split data
            X_indices = np.arange(len(X))
            self.train_indices, self.test_indices = train_test_split(
                X_indices, test_size=test_size, random_state=42, stratify=stratify
            )
            #####


            # X_indices = np.arange(len(X))
            # from sklearn.model_selection import train_test_split
            # self.train_indices, self.test_indices = train_test_split(
            #     X_indices, test_size=test_size, random_state=42, stratify=stratify
            # )
            
            X_train = X.iloc[self.train_indices]
            X_test = X.iloc[self.test_indices]
            
            # Split each target variable
            for target_name, y in y_dict.items():
                y_train_dict[target_name] = y.iloc[self.train_indices].values
                y_test_dict[target_name] = y.iloc[self.test_indices].values
            
            # Store split data
            self.X_train = X_train
            self.X_test = X_test
            self.y_trains = y_train_dict
            self.y_tests = y_test_dict
            
            self.logger.info(f"Feature engineering complete. Training set: {len(X_train)} rows, Testing set: {len(X_test)} rows")
            return X_train, X_test
        
        else:
            self.X_train = X
            for target_name, y in y_dict.items():
                self.y_trains[target_name] = y.values
            
            self.logger.info(f"Feature engineering complete. {len(X)} rows with {len(feature_cols)} features")
            return X, None
    
    def _extract_json_features(self, data: pd.DataFrame) -> None:
        """
        Extract features from JSON columns like attributes and specifications.
        
        Args:
            data: DataFrame to modify in-place
        """
        # Process attributes
        if 'attributes' in data.columns:
            self.logger.info("Extracting features from attributes")
            
            # Extract the number of attributes
            data['num_attributes'] = data['attributes'].apply(
                lambda x: len(x) if isinstance(x, (list, dict)) else 0
            )
            
            # Extract common attribute types
            common_attrs = ['material', 'color', 'size', 'weight', 'dimensions', 'type', 'model', 'brand']
            for attr in common_attrs:
                data[f'has_{attr}'] = data['attributes'].apply(
                    lambda x: self._check_attr_presence(x, attr)
                )
        
        # Process specifications
        if 'specifications' in data.columns:
            self.logger.info("Extracting features from specifications")
            
            # Extract the number of specifications
            data['num_specs'] = data['specifications'].apply(
                lambda x: len(x) if isinstance(x, (list, dict)) else 0
            )
            
            # Extract common specification types
            common_specs = ['width', 'height', 'length', 'weight', 'capacity', 'power', 'voltage']
            for spec in common_specs:
                data[f'has_{spec}_spec'] = data['specifications'].apply(
                    lambda x: self._check_attr_presence(x, spec)
                )
                
            # Extract numerical values from specifications when possible
            for spec in ['width', 'height', 'length', 'weight']:
                data[f'feature_{spec}_value'] = data['specifications'].apply(
                    lambda x: self._extract_numeric_spec(x, spec)
                )
    
    def _check_attr_presence(self, attrs, attr_name: str) -> bool:
        """Check if an attribute is present in the attributes dictionary or list."""
        if not attrs or not isinstance(attrs, (list, dict)):
            return False
        
        if isinstance(attrs, list):
            # Check if the attribute name appears in any string in the list
            return any(attr_name.lower() in str(item).lower() for item in attrs)
        
        elif isinstance(attrs, dict):
            # Check if the attribute name appears in any key
            for key in attrs.keys():
                if attr_name.lower() in key.lower():
                    return True
            
            # Also check if it appears in any value
            for value in attrs.values():
                if attr_name.lower() in str(value).lower():
                    return True
        
        return False
    
    def _extract_numeric_spec(self, specs, spec_name: str) -> float:
        """Extract numeric value from a specification if present."""
        if not specs or not isinstance(specs, dict):
            return 0.0
        
        # Look for the spec in keys
        for key, value in specs.items():
            if spec_name.lower() in key.lower():
                # Try to extract a numeric value
                if isinstance(value, (int, float)):
                    return float(value)
                elif isinstance(value, str):
                    # Extract digits and decimal points
                    numeric_str = re.search(r'(\d+\.?\d*)', value)
                    if numeric_str:
                        try:
                            return float(numeric_str.group(1))
                        except:
                            pass
        return 0.0
    
    def _create_text_features(self, data: pd.DataFrame) -> None:
        """
        Create additional features from text columns.
        
        Args:
            data: DataFrame to modify in-place
        """
        self.logger.info("Creating text features")
        
        # Process description length
        if 'Item Description' in data.columns:
            data['desc_length'] = data['Item Description'].apply(
                lambda x: len(str(x)) if pd.notna(x) else 0
            )
        
        # Process keyword features
        if 'extracted_keywords' in data.columns:
            # Number of keywords
            data['num_keywords'] = data['extracted_keywords'].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )
            
            # Calculate keyword density (keywords / description length)
            if 'desc_length' in data.columns:
                data['keyword_density'] = data.apply(
                    lambda row: row['num_keywords'] / max(1, row['desc_length']), axis=1
                )
            
            # Check for specific keywords that might indicate category
            # Group keywords by category
            keyword_groups = {
                'industrial': ['belt', 'conveyor', 'pump', 'valve', 'motor', 'filter', 'sensor', 'machine', 'equipment'],
                'electronics': ['server', 'computer', 'laptop', 'monitor', 'keyboard', 'software', 'license', 'cable'],
                'office': ['paper', 'pen', 'pencil', 'stapler', 'folder', 'binder', 'ink', 'toner', 'cartridge'],
                'safety': ['helmet', 'glove', 'mask', 'safety', 'protection', 'shield', 'goggle', 'extinguisher'],
                'tools': ['drill', 'hammer', 'saw', 'wrench', 'screwdriver', 'tool', 'measurement', 'gauge'],
                'construction': ['beam', 'concrete', 'cement', 'brick', 'lumber', 'drywall', 'insulation']
            }
            
            # Create features for each keyword group
            for group_name, keywords in keyword_groups.items():
                data[f'has_{group_name}_keywords'] = data['extracted_keywords'].apply(
                    lambda x: any(kw.lower() in str(k).lower() for k in x for kw in keywords) 
                    if isinstance(x, list) else False
                )
                
                data[f'num_{group_name}_keywords'] = data['extracted_keywords'].apply(
                    lambda x: sum(1 for k in x if any(kw.lower() in str(k).lower() for kw in keywords))
                    if isinstance(x, list) else 0
                )
    
    def _extract_ngrams(self, data: pd.DataFrame, n_range: Tuple[int, int] = (2, 3)) -> None:
        """
        Extract n-grams from processed text descriptions.
        
        Args:
            data: DataFrame to modify in-place
            n_range: Range of n-gram sizes to extract (min, max)
        """
        self.logger.info(f"Extracting {n_range[0]}-{n_range[1]}-grams from text descriptions")
        
        # Combine processed text fields if available
        if 'Item Description_processed' in data.columns:
            text_data = data['Item Description_processed'].fillna('').values
            
            # Create a set of n-grams for each document
            all_ngrams = []
            for text in text_data:
                words = text.split()
                doc_ngrams = set()
                
                for n in range(n_range[0], n_range[1] + 1):
                    if len(words) >= n:
                        for i in range(len(words) - n + 1):
                            ngram = ' '.join(words[i:i+n])
                            doc_ngrams.add(ngram)
                
                all_ngrams.append(doc_ngrams)
            
            # Find the most common n-grams across all documents
            ngram_counter = Counter()
            for doc_ngrams in all_ngrams:
                for ngram in doc_ngrams:
                    ngram_counter[ngram] += 1
            
            # Take the top 50 n-grams
            top_ngrams = [ngram for ngram, _ in ngram_counter.most_common(50)]
            
            # Create binary features for each top n-gram
            for i, ngram in enumerate(top_ngrams):
                data[f'ngram_{i}'] = data['Item Description_processed'].apply(
                    lambda x: 1 if pd.notna(x) and ngram in x else 0
                )
    
    def create_text_vectors(self, method: str = 'tfidf', max_features: int = 1000) -> None:
        """
        Create vector representations of text data.
        
        Args:
            method: Vectorization method ('tfidf' or 'count')
            max_features: Maximum number of features to extract
        """
        if self.X_train is None:
            self.logger.error("No training data. Call engineer_features() first.")
            raise ValueError("No training data")
        
        self.logger.info(f"Creating text vectors using {method} method with {max_features} max features")
        
        # Select text columns from the training data
        text_cols = [col for col in self.X_train.columns if '_processed' in col]
        
        for col in text_cols:
            self.logger.info(f"Vectorizing column: {col}")
            
            # Choose vectorizer based on method
            if method == 'tfidf':
                vectorizer = TfidfVectorizer(
                    max_features=max_features,
                    min_df=2,  # Ignore terms that appear in less than 2 documents
                    max_df=0.95,  # Ignore terms that appear in more than 95% of documents
                    sublinear_tf=True,  # Apply sublinear tf scaling (1 + log(tf))
                    ngram_range=(1, 2)  # Include unigrams and bigrams
                )
            else:  # count vectorizer
                vectorizer = CountVectorizer(
                    max_features=max_features,
                    min_df=2,
                    max_df=0.95,
                    ngram_range=(1, 2)
                )
            
            # Fit and transform the training data
            X_train_vec = vectorizer.fit_transform(self.X_train[col].fillna(''))
            
            # Transform the test data if available
            X_test_vec = None
            if self.X_test is not None:
                X_test_vec = vectorizer.transform(self.X_test[col].fillna(''))
            
            # Store the vectorizer
            self.vectorizers[col] = vectorizer
            
            # Add the vectorized features to the feature names
            feature_names = [f"{col}_{i}" for i in range(X_train_vec.shape[1])]
            self.feature_names.extend(feature_names)
            
            # Convert sparse matrices to DataFrames
            X_train_vec_df = pd.DataFrame(
                X_train_vec.toarray(),
                index=self.X_train.index,
                columns=feature_names
            )
            
            # Concatenate with the existing features
            self.X_train = pd.concat([self.X_train, X_train_vec_df], axis=1)
            
            # Do the same for the test data if available
            if X_test_vec is not None and self.X_test is not None:
                X_test_vec_df = pd.DataFrame(
                    X_test_vec.toarray(),
                    index=self.X_test.index,
                    columns=feature_names
                )
                self.X_test = pd.concat([self.X_test, X_test_vec_df], axis=1)
        
        self.logger.info(f"Text vectorization complete. Training data shape: {self.X_train.shape}")

    def _get_optimal_model_parameters(self, model_name: str, X: pd.DataFrame, y: np.ndarray) -> Dict:
        """
        Determine optimal parameters for a model using grid search.
        
        Args:
            model_name: Name of the model to optimize
            X: Feature data
            y: Target data
            
        Returns:
            Dictionary of optimal parameters
        """
        self.logger.info(f"Determining optimal parameters for {model_name}")
        
        # Define parameter grids for different models
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 4, 5]
            },
            'logistic_regression': {
                'C': [0.1, 1.0, 10.0],
                'solver': ['liblinear', 'lbfgs'],
                'max_iter': [1000]
            },
            'svm': {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            },
            'naive_bayes': {
                'alpha': [0.1, 0.5, 1.0]
            },
            'knn': {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance']
            }
        }
        
        if model_name not in param_grids:
            self.logger.warning(f"No parameter grid defined for {model_name}, using default parameters")
            return {}
        
        # Select base model class
        model_classes = {
            'random_forest': RandomForestClassifier,
            'gradient_boosting': GradientBoostingClassifier,
            'logistic_regression': LogisticRegression,
            'svm': SVC,
            'naive_bayes': MultinomialNB,
            'knn': KNeighborsClassifier
        }
        
        model_class = model_classes.get(model_name)
        if not model_class:
            return {}
        
        # Create base model instance
        if model_name == 'svm':
            base_model = model_class(probability=True, random_state=42)
        elif hasattr(model_class, 'random_state'):
            base_model = model_class(random_state=42)
        else:
            base_model = model_class()
        
        # Create grid search
        grid_search = GridSearchCV(
            base_model,
            param_grids[model_name],
            cv=3,
            scoring='f1_weighted',
            n_jobs=-1
        )
        
        # Run grid search
        try:
            grid_search.fit(X, y)
            self.logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
            return grid_search.best_params_
        except Exception as e:
            self.logger.error(f"Error in grid search for {model_name}: {str(e)}")
            return {}
    
    def train_models(self, target_col: str, models_to_train: Optional[List[str]] = None, 
                    optimize_params: bool = False) -> Dict:
        """
        Train multiple classification models for a specific target column.
        
        Args:
            target_col: Target column to predict (e.g., 'category_l1_name')
            models_to_train: List of model names to train, or None for all
            optimize_params: Whether to perform parameter optimization
            
        Returns:
            Dictionary of trained models with performance metrics
        """
        if self.X_train is None or not self.y_trains:
            self.logger.error("No training data. Call engineer_features() first.")
            raise ValueError("No training data")
        
        if target_col not in self.y_trains:
            self.logger.error(f"Target column {target_col} not found in training data")
            raise ValueError(f"Target column {target_col} not found")
        
        # Get the target data
        y_train = self.y_trains[target_col]
        y_test = self.y_tests.get(target_col) if self.y_tests else None
        
        self.logger.info(f"Training models for target: {target_col}")
        self.logger.info(f"Target distribution: {np.bincount(y_train)}")
        
        # Define available models
        available_models = {
            'random_forest': RandomForestClassifier(random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'svm': SVC(probability=True, random_state=42),
            'naive_bayes': MultinomialNB(),
            'mlp': MLPClassifier(random_state=42, max_iter=1000),
            'knn': KNeighborsClassifier()
        }
        
        # Select models to train
        if models_to_train is None:
            models_to_train = list(available_models.keys())
        
        # Dictionary to store trained models and their performance
        trained_models = {}
        
        # Train each model
        for model_name in models_to_train:
            if model_name not in available_models:
                self.logger.warning(f"Model {model_name} not available, skipping")
                continue
            
            self.logger.info(f"Training {model_name} model")
            
            try:
                # Get the base model
                model = available_models[model_name]
                
                # Optimize parameters if requested
                if optimize_params:
                    best_params = self._get_optimal_model_parameters(
                        model_name, self.X_train, y_train
                    )
                    if best_params:
                        model.set_params(**best_params)
                
                # Train the model
                model.fit(self.X_train, y_train)
                
                # Make predictions on training data
                y_train_pred = model.predict(self.X_train)
                
                # Calculate training metrics
                train_accuracy = accuracy_score(y_train, y_train_pred)
                train_f1 = f1_score(y_train, y_train_pred, average='weighted')
                train_precision = precision_score(y_train, y_train_pred, average='weighted')
                train_recall = recall_score(y_train, y_train_pred, average='weighted')
                
                # Make predictions on test data if available
                test_metrics = {}
                if self.X_test is not None and y_test is not None:
                    y_test_pred = model.predict(self.X_test)
                    
                    # Calculate test metrics
                    test_accuracy = accuracy_score(y_test, y_test_pred)
                    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
                    test_precision = precision_score(y_test, y_test_pred, average='weighted')
                    test_recall = recall_score(y_test, y_test_pred, average='weighted')
                    
                    test_metrics = {
                        'accuracy': test_accuracy,
                        'f1': test_f1,
                        'precision': test_precision,
                        'recall': test_recall
                    }
                
                # Store the model and its performance
                trained_models[model_name] = {
                    'model': model,
                    'train_metrics': {
                        'accuracy': train_accuracy,
                        'f1': train_f1,
                        'precision': train_precision,
                        'recall': train_recall
                    },
                    'test_metrics': test_metrics
                }
                
                self.logger.info(f"{model_name} training complete. Train accuracy: {train_accuracy:.4f}, Test accuracy: {test_metrics.get('accuracy', 'N/A')}")
                
            except Exception as e:
                self.logger.error(f"Error training {model_name} model: {str(e)}")
        
        # Store the trained models for this target
        self.models[target_col] = trained_models
        
        return trained_models
    
    def evaluate_models(self, target_col: str, plot_confusion_matrix: bool = True) -> Dict:
        """
        Evaluate trained models on test data and generate comparison plots.
        
        Args:
            target_col: Target column to evaluate
            plot_confusion_matrix: Whether to plot confusion matrices
            
        Returns:
            Dictionary of evaluation metrics
        """
        if target_col not in self.models:
            self.logger.error(f"No models trained for {target_col}. Call train_models() first.")
            raise ValueError(f"No models trained for {target_col}")
        
        if self.X_test is None or target_col not in self.y_tests:
            self.logger.error("No test data available for evaluation.")
            raise ValueError("No test data available")
        
        self.logger.info(f"Evaluating models for {target_col}")
        
        # Get the trained models
        models = self.models[target_col]
        
        # Check if we have any successfully trained models
        if not models:
            self.logger.warning(f"No successfully trained models for {target_col}")
            return {}
            
        # Get the test data
        X_test = self.X_test
        y_test = self.y_tests[target_col]
        
        # Dictionary to store evaluation results
        eval_results = {}
        
        # Metrics to calculate
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        # Prepare data for plots
        plot_data = {
            'model_names': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        # Evaluate each model
        for model_name, model_info in models.items():
            model = model_info['model']
            
            try:
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                # Add to plot data
                plot_data['model_names'].append(model_name)
                plot_data['accuracy'].append(accuracy)
                plot_data['precision'].append(precision)
                plot_data['recall'].append(recall)
                plot_data['f1'].append(f1)
                
                # Detailed classification report
                report = classification_report(y_test, y_pred, output_dict=True)
                
                # Confusion matrix
                if plot_confusion_matrix:
                    self._plot_confusion_matrix(
                        y_test, y_pred, model_name, target_col,
                        self.encoders.get(target_col)
                    )
                
                # Store evaluation results
                eval_results[model_name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'report': report
                }
                
                self.logger.info(f"{model_name} evaluation - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            except Exception as e:
                self.logger.error(f"Error evaluating {model_name} model: {str(e)}")
        
        # Create comparison plots only if we have any successful evaluations
        if plot_data['model_names']:
            self._plot_model_comparison(plot_data, target_col)
        else:
            self.logger.warning(f"No models could be evaluated for {target_col}")
        
        return eval_results
    
    def _plot_confusion_matrix(self, y_true, y_pred, model_name, target_col, encoder=None):
        """
        Plot confusion matrix for a model.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            target_col: Target column name
            encoder: Label encoder for class names
        """
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Get class names if encoder is available
        class_names = None
        if encoder is not None:
            class_names = encoder.classes_
        
        # Plot confusion matrix
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names
        )
        
        plt.title(f'Confusion Matrix - {model_name} - {target_col}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save the plot
        plot_dir = os.path.join(self.output_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f'cm_{target_col}_{model_name}.png'))
        plt.close()
    
    def _plot_model_comparison(self, plot_data, target_col):
        """
        Create model comparison plots.
        
        Args:
            plot_data: Dictionary containing plot data
            target_col: Target column name
        """
        # Check if we have any models to plot
        if not plot_data['model_names']:
            self.logger.warning(f"No successfully trained models for {target_col}. Skipping model comparison plots.")
            return
            
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Set width of bars
        bar_width = 0.2
        index = np.arange(len(plot_data['model_names']))
        
        # Plot bars for each metric
        plt.bar(index - bar_width*1.5, plot_data['accuracy'], bar_width, label='Accuracy')
        plt.bar(index - bar_width*0.5, plot_data['precision'], bar_width, label='Precision')
        plt.bar(index + bar_width*0.5, plot_data['recall'], bar_width, label='Recall')
        plt.bar(index + bar_width*1.5, plot_data['f1'], bar_width, label='F1')
        
        # Add labels and legend
        plt.xlabel('Model')
        plt.ylabel('Score')
        plt.title(f'Model Comparison - {target_col}')
        plt.xticks(index, plot_data['model_names'], rotation=45)
        plt.legend()
        plt.tight_layout()
        
        # Save the plot
        plot_dir = os.path.join(self.output_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f'model_comparison_{target_col}.png'))
        plt.close()
        
        # Create a radar chart for the best model
        if plot_data['f1']:  # Only if we have F1 scores
            best_model_idx = np.argmax(plot_data['f1'])
            best_model = plot_data['model_names'][best_model_idx]
            
            self._plot_radar_chart(
                best_model, 
                [plot_data['accuracy'][best_model_idx], 
                 plot_data['precision'][best_model_idx],
                 plot_data['recall'][best_model_idx],
                 plot_data['f1'][best_model_idx]],
                target_col
            )
    
    def _plot_radar_chart(self, model_name, metrics, target_col):
        """
        Create a radar chart for model metrics.
        
        Args:
            model_name: Name of the model
            metrics: List of metrics [accuracy, precision, recall, f1]
            target_col: Target column name
        """
        # Create figure
        plt.figure(figsize=(8, 8))
        
        # Set radar chart parameters
        categories = ['Accuracy', 'Precision', 'Recall', 'F1']
        N = len(categories)
        
        # Create angles for the radar chart
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the circle
        
        # Add the metrics to the plot
        values = metrics + [metrics[0]]  # Close the circle
        
        # Create the plot
        ax = plt.subplot(111, polar=True)
        plt.xticks(angles[:-1], categories)
        ax.set_rlabel_position(0)
        plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ['0.2', '0.4', '0.6', '0.8', '1.0'], color='grey', size=8)
        plt.ylim(0, 1)
        
        # Plot the metrics
        ax.plot(angles, values, linewidth=1, linestyle='solid')
        ax.fill(angles, values, alpha=0.1)
        
        # Add title
        plt.title(f'Performance Metrics - {model_name} - {target_col}')
        
        # Save the plot
        plot_dir = os.path.join(self.output_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f'radar_{target_col}_{model_name}.png'))
        plt.close()
    
    def reduce_dimensions(self, method: str = 'pca', n_components: int = 2) -> np.ndarray:
        """
        Reduce the dimensionality of the feature space for visualization.
        
        Args:
            method: Dimensionality reduction method ('pca', 'tsne', 'umap')
            n_components: Number of components to reduce to
            
        Returns:
            Array of reduced dimensions
        """
        if self.X_train is None:
            self.logger.error("No training data. Call engineer_features() first.")
            raise ValueError("No training data")
        
        self.logger.info(f"Reducing dimensions using {method} to {n_components} components")
        
        # Combine training and test data for visualization
        X_all = self.X_train.copy()
        if self.X_test is not None:
            X_all = pd.concat([self.X_train, self.X_test])
        
        # Apply dimensionality reduction
        if method == 'pca':
            reducer = PCA(n_components=n_components, random_state=42)
        elif method == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=42)
        elif method == 'umap' and UMAP_AVAILABLE:
            reducer = umap.UMAP(n_components=n_components, random_state=42)
        else:
            self.logger.warning(f"Unknown or unavailable method: {method}, using PCA")
            reducer = PCA(n_components=n_components, random_state=42)
        
        # Fit and transform
        try:
            reduced_data = reducer.fit_transform(X_all)
            
            self.logger.info(f"Dimension reduction complete. Shape: {reduced_data.shape}")
            
            return reduced_data
        except Exception as e:
            self.logger.error(f"Error reducing dimensions: {str(e)}")
            return np.array([])
    
    def visualize_clusters(self, target_col: str, n_clusters: int = 5, 
                          dim_reduction_method: str = 'pca') -> None:
        """
        Perform clustering and visualize the results.
        
        Args:
            target_col: Target column to color the clusters by
            n_clusters: Number of clusters for K-means
            dim_reduction_method: Method for dimensionality reduction
        """
        if self.X_train is None or target_col not in self.y_trains:
            self.logger.error("No training data or target column not found.")
            raise ValueError("No training data or target not found")
        
        self.logger.info(f"Visualizing clusters for {target_col}")
        
        try:
            # Reduce dimensions for visualization
            reduced_data = self.reduce_dimensions(method=dim_reduction_method, n_components=2)
            
            if len(reduced_data) == 0:
                self.logger.error("Dimension reduction failed, aborting visualization")
                return
            
            # Prepare data for plotting
            if self.X_test is not None:
                all_indices = np.concatenate([self.train_indices, self.test_indices])
            else:
                all_indices = self.train_indices
            
            # Get target values
            all_targets = np.concatenate([
                self.y_trains[target_col],
                self.y_tests[target_col] if self.y_tests and target_col in self.y_tests else []
            ])
            
            # Create figure
            plt.figure(figsize=(15, 10))
            
            # Plot 1: Scatter plot colored by true categories
            plt.subplot(1, 2, 1)
            scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=all_targets, cmap='tab10', alpha=0.7)
            
            # Add a colorbar with class labels if available
            if target_col in self.encoders:
                encoder = self.encoders[target_col]
                class_names = encoder.classes_
                cbar = plt.colorbar(scatter, ticks=range(len(class_names)))
                cbar.set_ticklabels(class_names)
            else:
                plt.colorbar(scatter)
            
            plt.title(f'Data Distribution by {target_col}')
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
            
            # Plot 2: Cluster visualization
            plt.subplot(1, 2, 2)
            
            # Apply K-means clustering
            n_clusters = min(n_clusters, len(reduced_data))  # Ensure we don't try more clusters than data points
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(reduced_data)
            
            # Plot clusters
            scatter2 = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=cluster_labels, cmap='tab10', alpha=0.7)
            
            # Plot cluster centers
            centers = kmeans.cluster_centers_
            plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5, marker='x')
            
            plt.colorbar(scatter2)
            plt.title(f'K-means Clustering (k={n_clusters})')
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
            
            plt.tight_layout()
            
            # Save the plot
            plot_dir = os.path.join(self.output_dir, 'plots')
            os.makedirs(plot_dir, exist_ok=True)
            plt.savefig(os.path.join(plot_dir, f'clusters_{target_col}.png'))
            plt.close()
            
            # Calculate cluster purity
            self._calculate_cluster_purity(cluster_labels, all_targets, target_col)
            
        except Exception as e:
            self.logger.error(f"Error visualizing clusters: {str(e)}")
            return
    
    def _calculate_cluster_purity(self, cluster_labels, target_labels, target_col):
        """
        Calculate and plot cluster purity metrics.
        
        Args:
            cluster_labels: Cluster assignments
            target_labels: True target labels
            target_col: Target column name
        """
        self.logger.info("Calculating cluster purity metrics")
        
        # Create a cross-tabulation of clusters vs. true labels
        cross_tab = pd.crosstab(cluster_labels, target_labels)
        
        # Calculate purity for each cluster
        purity_values = []
        for cluster_idx in range(len(cross_tab)):
            if cluster_idx in cross_tab.index:
                cluster_size = cross_tab.loc[cluster_idx].sum()
                most_common = cross_tab.loc[cluster_idx].max()
                purity = most_common / cluster_size if cluster_size > 0 else 0
                purity_values.append(purity)
        
        # Calculate average purity
        avg_purity = np.mean(purity_values)
        
        self.logger.info(f"Average cluster purity: {avg_purity:.4f}")
        
        # Create a heatmap of the cross-tabulation
        plt.figure(figsize=(12, 8))
        
        # If label encoder is available, use class names
        if target_col in self.encoders:
            encoder = self.encoders[target_col]
            class_names = encoder.classes_
            cross_tab.columns = [f"{class_names[i]}" for i in cross_tab.columns]
        
        sns.heatmap(cross_tab, annot=True, fmt='d', cmap='YlGnBu')
        plt.title(f'Cluster Composition - {target_col} (Avg. Purity: {avg_purity:.4f})')
        plt.ylabel('Cluster')
        plt.xlabel('True Label')
        
        # Save the plot
        plot_dir = os.path.join(self.output_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f'cluster_purity_{target_col}.png'))
        plt.close()
    
    def save_models(self) -> None:
        """Save trained models to disk."""
        if not self.models:
            self.logger.warning("No models to save")
            return
        
        model_dir = os.path.join(self.output_dir, 'models')
        os.makedirs(model_dir, exist_ok=True)
        
        self.logger.info(f"Saving models to {model_dir}")
        
        for target_col, models in self.models.items():
            target_dir = os.path.join(model_dir, target_col)
            os.makedirs(target_dir, exist_ok=True)
            
            # Save each model
            for model_name, model_info in models.items():
                model_path = os.path.join(target_dir, f"{model_name}.pkl")
                
                try:
                    with open(model_path, 'wb') as f:
                        pickle.dump(model_info['model'], f)
                    self.logger.info(f"Saved model: {model_path}")
                except Exception as e:
                    self.logger.error(f"Error saving model {model_path}: {str(e)}")
        
        # Save encoders
        encoder_path = os.path.join(model_dir, 'encoders.pkl')
        try:
            with open(encoder_path, 'wb') as f:
                pickle.dump(self.encoders, f)
            self.logger.info(f"Saved encoders: {encoder_path}")
        except Exception as e:
            self.logger.error(f"Error saving encoders: {str(e)}")
        
        # Save vectorizers
        vectorizer_path = os.path.join(model_dir, 'vectorizers.pkl')
        try:
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(self.vectorizers, f)
            self.logger.info(f"Saved vectorizers: {vectorizer_path}")
        except Exception as e:
            self.logger.error(f"Error saving vectorizers: {str(e)}")
    
    def load_models(self, model_dir: str) -> bool:
        """
        Load trained models from disk.
        
        Args:
            model_dir: Directory containing saved models
            
        Returns:
            True if models were loaded successfully, False otherwise
        """
        if not os.path.isdir(model_dir):
            self.logger.error(f"Model directory not found: {model_dir}")
            return False
        
        self.logger.info(f"Loading models from {model_dir}")
        
        # Reset models
        self.models = {}
        
        # Load encoders
        encoder_path = os.path.join(model_dir, 'encoders.pkl')
        if os.path.isfile(encoder_path):
            try:
                with open(encoder_path, 'rb') as f:
                    self.encoders = pickle.load(f)
                self.logger.info(f"Loaded encoders from {encoder_path}")
            except Exception as e:
                self.logger.error(f"Error loading encoders: {str(e)}")
                return False
        
        # Load vectorizers
        vectorizer_path = os.path.join(model_dir, 'vectorizers.pkl')
        if os.path.isfile(vectorizer_path):
            try:
                with open(vectorizer_path, 'rb') as f:
                    self.vectorizers = pickle.load(f)
                self.logger.info(f"Loaded vectorizers from {vectorizer_path}")
            except Exception as e:
                self.logger.error(f"Error loading vectorizers: {str(e)}")
                return False
        
        # Load models for each target
        for target_dir in os.listdir(model_dir):
            target_path = os.path.join(model_dir, target_dir)
            
            if os.path.isdir(target_path):
                self.models[target_dir] = {}
                
                # Load each model for this target
                for model_file in os.listdir(target_path):
                    if model_file.endswith('.pkl'):
                        model_name = os.path.splitext(model_file)[0]
                        model_path = os.path.join(target_path, model_file)
                        
                        try:
                            with open(model_path, 'rb') as f:
                                model = pickle.load(f)
                            
                            self.models[target_dir][model_name] = {
                                'model': model,
                                'train_metrics': {},
                                'test_metrics': {}
                            }
                            
                            self.logger.info(f"Loaded model: {model_path}")
                        except Exception as e:
                            self.logger.error(f"Error loading model {model_path}: {str(e)}")
        
        self.logger.info(f"Loaded models for {len(self.models)} targets")
        return True
    
    def update_model_with_feedback(self, target_col: str, model_name: str, 
                                  new_data: pd.DataFrame, correct_labels: np.ndarray) -> bool:
        """
        Update a trained model with user feedback.
        
        Args:
            target_col: Target column name
            model_name: Name of the model to update
            new_data: New data with features
            correct_labels: Correct labels for the new data
            
        Returns:
            True if the model was updated successfully, False otherwise
        """
        if target_col not in self.models or model_name not in self.models[target_col]:
            self.logger.error(f"Model {model_name} for {target_col} not found")
            return False
        
        self.logger.info(f"Updating {model_name} model for {target_col} with {len(new_data)} new samples")
        
        # Get the model
        model_info = self.models[target_col][model_name]
        model = model_info['model']
        
        # Prepare the new data
        # Apply vectorization if necessary
        processed_data = new_data.copy()
        
        # Apply text preprocessing
        text_cols = ['Item Description', 'clean_description', 'primary_function']
        for col in text_cols:
            if col in processed_data.columns:
                processed_data[f'{col}_processed'] = processed_data[col].apply(self.preprocess_text)
        
        # Apply vectorization for text columns
        for col, vectorizer in self.vectorizers.items():
            if col in processed_data.columns:
                # Transform the text data
                X_vec = vectorizer.transform(processed_data[col].fillna(''))
                
                # Convert to DataFrame
                feature_names = [f"{col}_{i}" for i in range(X_vec.shape[1])]
                X_vec_df = pd.DataFrame(
                    X_vec.toarray(),
                    index=processed_data.index,
                    columns=feature_names
                )
                
                # Concatenate with processed data
                processed_data = pd.concat([processed_data, X_vec_df], axis=1)
        
        # Extract features matching the original training data
        X_new = processed_data[self.feature_names]
        
        # Update the model
        try:
            # Different update methods depending on model type
            if hasattr(model, 'partial_fit'):
                # Models that support partial_fit (incremental learning)
                # Get unique classes
                classes = None
                if target_col in self.encoders:
                    classes = np.arange(len(self.encoders[target_col].classes_))
                
                model.partial_fit(X_new, correct_labels, classes=classes)
                
                self.logger.info(f"Model updated using partial_fit")
                
            elif isinstance(model, (RandomForestClassifier, GradientBoostingClassifier)):
                # For ensemble models, we can train more estimators and combine them
                # Train a new estimator on the feedback data
                if isinstance(model, RandomForestClassifier):
                    new_estimator = RandomForestClassifier(
                        n_estimators=20,  # Smaller ensemble for the update
                        random_state=42
                    )
                else:  # GradientBoostingClassifier
                    new_estimator = GradientBoostingClassifier(
                        n_estimators=20,
                        random_state=42
                    )
                
                new_estimator.fit(X_new, correct_labels)
                
                # Combine estimators (for RandomForest)
                if isinstance(model, RandomForestClassifier):
                    model.estimators_ += new_estimator.estimators_
                    model.n_estimators = len(model.estimators_)
                
                self.logger.info(f"Model updated by adding new estimators")
                
            else:
                # For models that don't support incremental learning,
                # we'll retrain on combined data
                # Get original training data
                X_orig = self.X_train
                y_orig = self.y_trains[target_col]
                
                # Combine with new data
                X_combined = pd.concat([X_orig, X_new])
                y_combined = np.concatenate([y_orig, correct_labels])
                
                # Retrain the model
                model.fit(X_combined, y_combined)
                
                self.logger.info(f"Model updated by retraining on combined data")
            
            # Store the updated model
            self.models[target_col][model_name]['model'] = model
            
            # Save the updated model
            model_dir = os.path.join(self.output_dir, 'models', target_col)
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f"{model_name}.pkl")
            
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            self.logger.info(f"Saved updated model: {model_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating model: {str(e)}")
            return False
    
    def predict(self, data: pd.DataFrame, target_col: str, model_name: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new data.
        
        Args:
            data: Data to make predictions on
            target_col: Target column to predict
            model_name: Name of the model to use, or None for best model
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if target_col not in self.models:
            self.logger.error(f"No models trained for {target_col}")
            raise ValueError(f"No models trained for {target_col}")
        
        # Select the model
        if model_name is None:
            # Find the best model based on test metrics
            best_model_name = None
            best_f1 = -1
            
            for name, info in self.models[target_col].items():
                if 'test_metrics' in info and 'f1' in info['test_metrics']:
                    f1 = info['test_metrics']['f1']
                    if f1 > best_f1:
                        best_f1 = f1
                        best_model_name = name
            
            if best_model_name is None:
                # Fallback to first model
                best_model_name = list(self.models[target_col].keys())[0]
            
            model_name = best_model_name
        
        if model_name not in self.models[target_col]:
            self.logger.error(f"Model {model_name} not found for {target_col}")
            raise ValueError(f"Model {model_name} not found")
        
        self.logger.info(f"Making predictions with {model_name} model for {target_col}")
        
        # Get the model
        model = self.models[target_col][model_name]['model']
        
        # Prepare the data
        processed_data = data.copy()
        
        # Apply text preprocessing
        text_cols = ['Item Description', 'clean_description', 'primary_function']
        for col in text_cols:
            if col in processed_data.columns:
                processed_data[f'{col}_processed'] = processed_data[col].apply(self.preprocess_text)
        
        # Apply vectorization for text columns
        for col, vectorizer in self.vectorizers.items():
            if col in processed_data.columns:
                # Transform the text data
                X_vec = vectorizer.transform(processed_data[col].fillna(''))
                
                # Convert to DataFrame
                feature_names = [f"{col}_{i}" for i in range(X_vec.shape[1])]
                X_vec_df = pd.DataFrame(
                    X_vec.toarray(),
                    index=processed_data.index,
                    columns=feature_names
                )
                
                # Concatenate with processed data
                processed_data = pd.concat([processed_data, X_vec_df], axis=1)
        
        # Extract features matching the original training data
        # Make sure all required feature columns exist
        for feature in self.feature_names:
            if feature not in processed_data.columns:
                processed_data[feature] = 0
        
        X = processed_data[self.feature_names]
        
        # Make predictions
        try:
            y_pred = model.predict(X)
            
            # Get prediction probabilities if available
            y_prob = None
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X)
            
            # Convert numeric predictions to original categories if encoder is available
            if target_col in self.encoders:
                encoder = self.encoders[target_col]
                y_pred_labels = encoder.inverse_transform(y_pred)
                
                self.logger.info(f"Predicted {len(y_pred)} samples")
                return y_pred_labels, y_prob
            
            self.logger.info(f"Predicted {len(y_pred)} samples")
            return y_pred, y_prob
            
        except Exception as e:
            self.logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def generate_report(self, target_cols: List[str]) -> str:
        """
        Generate a comprehensive report of the classification results.
        
        Args:
            target_cols: List of target columns to include in the report
            
        Returns:
            Report text
        """
        if not self.models:
            self.logger.error("No models trained. Cannot generate report.")
            return "No models trained. Cannot generate report."
        
        report_lines = [
            "# Procurement Data Classification Report",
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Dataset Information",
            f"- Total samples: {len(self.data) if self.data is not None else 'N/A'}",
            f"- Training samples: {len(self.X_train) if self.X_train is not None else 'N/A'}",
            f"- Testing samples: {len(self.X_test) if self.X_test is not None else 'N/A'}",
            f"- Features: {len(self.feature_names)} features",
            "",
            "## Model Performance Summary"
        ]
        
        for target_col in target_cols:
            if target_col not in self.models:
                continue
            
            report_lines.append(f"### Target: {target_col}")
            report_lines.append("")
            
            # Create a table of model performance
            report_lines.append("| Model | Accuracy | Precision | Recall | F1 Score |")
            report_lines.append("| ----- | -------- | --------- | ------ | -------- |")
            
            for model_name, model_info in self.models[target_col].items():
                if 'test_metrics' in model_info and model_info['test_metrics']:
                    metrics = model_info['test_metrics']
                    report_lines.append(
                        f"| {model_name} | {metrics.get('accuracy', 'N/A'):.4f} | "
                        f"{metrics.get('precision', 'N/A'):.4f} | {metrics.get('recall', 'N/A'):.4f} | "
                        f"{metrics.get('f1', 'N/A'):.4f} |"
                    )
                else:
                    report_lines.append(f"| {model_name} | N/A | N/A | N/A | N/A |")
            
            report_lines.append("")
        
        report_lines.append("## Feature Importance")
        report_lines.append("")
        
        # Add feature importance plots if available
        feature_importance_added = False
        
        for target_col in target_cols:
            if target_col not in self.models:
                continue
            
            for model_name, model_info in self.models[target_col].items():
                model = model_info['model']
                
                if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
                    feature_importance_added = True
                    self._plot_feature_importance(model, target_col, model_name)
                    report_lines.append(f"![Feature Importance for {model_name} on {target_col}](plots/feature_importance_{target_col}_{model_name}.png)")
                    report_lines.append("")
        
        if not feature_importance_added:
            report_lines.append("No feature importance information available.")
            report_lines.append("")
        
        report_lines.append("## Dimensionality Reduction and Clustering")
        report_lines.append("")
        
        # Add cluster visualization plots
        for target_col in target_cols:
            if os.path.isfile(os.path.join(self.output_dir, 'plots', f'clusters_{target_col}.png')):
                report_lines.append(f"![Clusters for {target_col}](plots/clusters_{target_col}.png)")
                report_lines.append("")
            
            if os.path.isfile(os.path.join(self.output_dir, 'plots', f'cluster_purity_{target_col}.png')):
                report_lines.append(f"![Cluster Purity for {target_col}](plots/cluster_purity_{target_col}.png)")
                report_lines.append("")
        
        report_lines.append("## Conclusion")
        report_lines.append("")
        report_lines.append("This report presents the results of the procurement data classification system. "
                          "The system has been trained to classify products into various category levels using "
                          "machine learning models. The performance metrics and visualizations show the effectiveness "
                          "of the models in categorizing the products.")
        report_lines.append("")
        report_lines.append("The system supports:")
        report_lines.append("- Multiple classification models")
        report_lines.append("- Advanced text processing and feature engineering")
        report_lines.append("- Dimensionality reduction and visualization")
        report_lines.append("- Clustering for pattern discovery")
        report_lines.append("- User feedback for incremental learning")
        
        # Write the report to disk
        report_text = "\n".join(report_lines)
        report_path = os.path.join(self.output_dir, 'classification_report.md')
        
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        self.logger.info(f"Generated report: {report_path}")
        
        return report_text
    
    def _plot_feature_importance(self, model, target_col, model_name):
        """
        Plot feature importance for a model.
        
        Args:
            model: Trained model
            target_col: Target column name
            model_name: Name of the model
        """
        plt.figure(figsize=(12, 8))
        
        feature_importance = None
        
        # Extract feature importance
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # For models like LogisticRegression
            if len(model.coef_.shape) == 1:
                feature_importance = model.coef_
            else:
                # For multi-class, take the mean of absolute values
                feature_importance = np.mean(np.abs(model.coef_), axis=0)
        
        if feature_importance is not None and len(feature_importance) > 0:
            # Get feature names
            feature_names = self.feature_names[:len(feature_importance)]
            
            # Sort by importance
            indices = np.argsort(feature_importance)[::-1]
            
            # Take top 20 features
            top_indices = indices[:20]
            top_features = [feature_names[i] for i in top_indices]
            top_importance = feature_importance[top_indices]
            
            # Plot importance
            plt.barh(range(len(top_features)), top_importance, align='center')
            plt.yticks(range(len(top_features)), top_features)
            plt.title(f'Feature Importance - {model_name} - {target_col}')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.tight_layout()
            
            # Save the plot
            plot_dir = os.path.join(self.output_dir, 'plots')
            os.makedirs(plot_dir, exist_ok=True)
            plt.savefig(os.path.join(plot_dir, f'feature_importance_{target_col}_{model_name}.png'))
            plt.close()
    
    def run_pipeline(self, target_cols: Optional[List[str]] = None) -> None:
        """
        Run the complete classification pipeline.
        
        Args:
            target_cols: List of target columns to train models for, or None for all
        """
        self.logger.info("Starting classification pipeline")
        
        # Load data
        self.load_data()
        
        # Engineer features
        self.engineer_features()
        
        # Create text vectors
        self.create_text_vectors()
        
        # Determine target columns if not provided
        if target_cols is None:
            target_cols = [
                col for col in self.y_trains.keys()
            ]
        
        # Train models for each target
        for target_col in target_cols:
            if target_col not in self.y_trains:
                self.logger.warning(f"Target column {target_col} not found in training data, skipping")
                continue
            
            self.logger.info(f"Processing target: {target_col}")
            
            try:
                # Train models
                trained_models = self.train_models(target_col)
                
                # Skip evaluation if no models were successfully trained
                if not trained_models:
                    self.logger.warning(f"No models were successfully trained for {target_col}, skipping evaluation")
                    continue
                
                # Evaluate models
                self.evaluate_models(target_col)
                
                # Dimensionality reduction and clustering
                self.visualize_clusters(target_col)
            except Exception as e:
                self.logger.error(f"Error processing target {target_col}: {str(e)}")
                continue
        
        # Save models
        self.save_models()
        
        # Generate report
        self.generate_report(target_cols)
        
        self.logger.info("Classification pipeline complete")




# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Procurement Data Classification System')
    parser.add_argument('--data', type=str, default="../output/enriched3/enriched_products.csv", help='Path to the CSV data file')
    parser.add_argument('--output', type=str, default='output', help='Output directory for results')
    parser.add_argument('--targets', type=str, nargs='+', default=None, 
                      help='Target columns to train models for (e.g., category_l1_name)')
    
    args = parser.parse_args()
    
    # Initialize classifier
    classifier = ProcurementClassifier(args.data, args.output)
    
    # Run pipeline
    classifier.run_pipeline(args.targets)
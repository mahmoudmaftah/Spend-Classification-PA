"""
PAVUS AI Procurement Categorization

This module implements the core categorization algorithm that combines
traditional machine learning with LLM-enhanced features to categorize
procurement transactions at scale.

Features:
- Hybrid categorization algorithm (LLM + ML)
- Multi-level hierarchical classification
- Confidence scoring for predictions
- Manual override capabilities
- Proper utilization of enriched data features
"""

import pandas as pd
import numpy as np
import os
import json
import logging
import pickle
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("procurement_categorization.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('procurement_categorization')

class ProcurementCategorization:
    """
    A module for categorizing procurement data using a hybrid approach
    combining traditional ML and LLM-enhanced features.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the categorization module.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        
        # Set up default configuration
        self._set_default_config()
        
        # Initialize models
        self.models = {
            'level1': None,
            'level2': None,
            'level3': None,
            'level4': None,  # Added level 4 support
            'level5': None   # Added level 5 support
        }
        
        # Initialize label encoders
        self.label_encoders = {
            'level1': LabelEncoder(),
            'level2': LabelEncoder(),
            'level3': LabelEncoder(),
            'level4': LabelEncoder(),  # Added level 4 support
            'level5': LabelEncoder()   # Added level 5 support
        }
        
        # Initialize vectorizers
        self.vectorizers = {
            'description': None
        }
        
        # Load taxonomy if provided
        self.taxonomy = None
        if 'taxonomy_path' in self.config and os.path.exists(self.config['taxonomy_path']):
            with open(self.config['taxonomy_path'], 'r') as f:
                self.taxonomy = json.load(f)
                logger.info(f"Loaded taxonomy with {len(self.taxonomy)} categories")
        
        logger.info("Procurement Categorization module initialized")
    
    def _set_default_config(self):
        """Set default configuration parameters"""
        self.config.setdefault('output_dir', 'output/models')
        self.config.setdefault('model_type', 'random_forest')
        self.config.setdefault('min_confidence_threshold', 0.6)
        self.config.setdefault('description_col', 'Item Description')
        self.config.setdefault('supplier_col', 'Vendor Name')
        self.config.setdefault('confidence_col', 'category_confidence')
        self.config.setdefault('final_category_id_col', 'final_category_id')
        self.config.setdefault('final_category_level_col', 'final_category_level')
        self.config.setdefault('category_path_col', 'category_path')
        self.config.setdefault('use_llm_features', True)
        self.config.setdefault('use_supplier_features', True)  # Option to include supplier data
        self.config.setdefault('n_estimators', 100)
        self.config.setdefault('max_depth', 20)
        self.config.setdefault('random_state', 42)
        self.config.setdefault('test_size', 0.2)
        self.config.setdefault('max_levels', 5)  # Support up to 5 levels
    
    def prepare_features(self, df: pd.DataFrame, training: bool = True, 
                         supplier_df: Optional[pd.DataFrame] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Prepare features for the categorization model.
        
        Args:
            df: DataFrame containing product data with descriptions
            training: Whether this is for training (True) or prediction (False)
            supplier_df: Optional DataFrame with supplier enriched data
            
        Returns:
            Tuple of (feature_matrix, feature_info)
        """
        description_col = self.config['description_col']
        supplier_col = self.config['supplier_col']
        
        logger.info(f"Preparing features for {'training' if training else 'prediction'}")
        
        # Initialize feature info dictionary
        feature_info = {}
        
        # 1. Text features from product description
        if training or self.vectorizers['description'] is None:
            self.vectorizers['description'] = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            description_features = self.vectorizers['description'].fit_transform(df[description_col].fillna(''))
            feature_info['description_vocab'] = self.vectorizers['description'].vocabulary_
        else:
            description_features = self.vectorizers['description'].transform(df[description_col].fillna(''))
        
        logger.info(f"Generated {description_features.shape[1]} text features from descriptions")
        
        # 2. Supplier one-hot encoding (if available and enabled)
        supplier_dummies = pd.DataFrame(index=df.index)
        if self.config['use_supplier_features'] and supplier_col in df.columns:
            if training:
                # Create and store one-hot encoding for suppliers
                supplier_dummies = pd.get_dummies(df[supplier_col], prefix='supplier')
                feature_info['supplier_columns'] = supplier_dummies.columns.tolist()
            else:
                # Use previously defined supplier columns and fill with zeros for new suppliers
                supplier_dummies = pd.get_dummies(df[supplier_col], prefix='supplier')
                if 'supplier_columns' in feature_info:
                    for col in feature_info['supplier_columns']:
                        if col not in supplier_dummies.columns:
                            supplier_dummies[col] = 0
                    supplier_dummies = supplier_dummies[feature_info['supplier_columns']]
            
            logger.info(f"Generated {supplier_dummies.shape[1]} supplier features")
        
        # 3. Add supplier enriched features if available
        if self.config['use_supplier_features'] and supplier_df is not None and supplier_col in df.columns:
            # Map supplier features to product data
            supplier_features_df = pd.DataFrame(index=df.index)
            
            # Create numeric features from supplier data
            supplier_numeric_cols = [
                'sustainability_rating', 'digital_presence_score', 
                'financial_health', 'innovation_score'
            ]
            
            # Create categorical features from supplier data
            supplier_cat_cols = [
                'industry', 'market_segment', 'likely_size', 
                'public_private_status', 'supplier_risk_level',
                'customer_base', 'global_reach', 'pricing_tier'
            ]
            
            # Join supplier data to product data
            if all(col in supplier_df.columns for col in supplier_numeric_cols + supplier_cat_cols):
                # Merge supplier data with product data
                merged_df = df.merge(
                    supplier_df[['Vendor Name'] + supplier_numeric_cols + supplier_cat_cols],
                    left_on=supplier_col,
                    right_on='Vendor Name',
                    how='left'
                )
                
                # Fill missing values
                for col in supplier_numeric_cols:
                    if col in merged_df.columns:
                        merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce').fillna(0)
                        supplier_features_df[f'supplier_{col}'] = merged_df[col]
                
                # One-hot encode categorical supplier features
                for col in supplier_cat_cols:
                    if col in merged_df.columns:
                        if training:
                            cat_dummies = pd.get_dummies(merged_df[col], prefix=f'supplier_{col}')
                            feature_info[f'supplier_{col}_columns'] = cat_dummies.columns.tolist()
                        else:
                            if f'supplier_{col}_columns' in feature_info:
                                cat_dummies = pd.get_dummies(merged_df[col], prefix=f'supplier_{col}')
                                for dummy_col in feature_info[f'supplier_{col}_columns']:
                                    if dummy_col not in cat_dummies.columns:
                                        cat_dummies[dummy_col] = 0
                                cat_dummies = cat_dummies[feature_info[f'supplier_{col}_columns']]
                            else:
                                continue
                        
                        supplier_features_df = pd.concat([supplier_features_df, cat_dummies], axis=1)
                
                logger.info(f"Generated {supplier_features_df.shape[1]} enriched supplier features")
            else:
                logger.warning("Required supplier columns not found in supplier data")
        else:
            supplier_features_df = pd.DataFrame(index=df.index)
        
        # 4. LLM-derived features (if available and enabled)
        llm_dummies = pd.DataFrame(index=df.index)
        if self.config['use_llm_features']:
            # Check for LLM-derived columns from enrichment
            potential_llm_cols = [
                'category_l1', 'category_l2', 'category_l3',
                'primary_function', 'attributes', 'specifications',
                'word_count'  # Include word count as a feature
            ]
            
            # Get available LLM feature columns
            available_llm_cols = [col for col in potential_llm_cols if col in df.columns]
            
            if available_llm_cols:
                # Process each LLM feature column
                llm_feature_cols = []
                
                for col in available_llm_cols:
                    # Add word_count directly if it exists
                    if col == 'word_count' and col in df.columns:
                        llm_dummies[col] = df[col]
                        continue
                        
                    # Handle JSON string columns (attributes, specifications)
                    if col in ['attributes', 'specifications']:
                        # Parse JSON strings to lists
                        try:
                            df[f'{col}_parsed'] = df[col].apply(
                                lambda x: json.loads(x) if isinstance(x, str) and x else []
                            )
                            
                            # Count number of items
                            df[f'{col}_count'] = df[f'{col}_parsed'].apply(len)
                            llm_feature_cols.append(f'{col}_count')
                            
                            # Flatten lists and create bag-of-words for top items
                            all_items = []
                            for items in df[f'{col}_parsed'].tolist():
                                if isinstance(items, list):
                                    all_items.extend(items)
                                    
                            if all_items:
                                top_items = pd.Series(all_items).value_counts().head(50).index.tolist()
                                
                                for item in top_items:
                                    item_key = str(item).replace(" ", "_").replace(".", "").replace(",", "")
                                    df[f'{col}_{item_key}'] = df[f'{col}_parsed'].apply(
                                        lambda x: 1 if isinstance(x, list) and item in x else 0
                                    )
                                    llm_feature_cols.append(f'{col}_{item_key}')
                        except Exception as e:
                            logger.warning(f"Error processing LLM column {col}: {str(e)}")
                    else:
                        # Process categorical columns
                        llm_feature_cols.append(col)
                        # Convert to string to handle any non-string values
                        df[col] = df[col].astype(str)
                
                # Create one-hot encoding for categorical LLM features
                if llm_feature_cols:
                    # Handle the case where columns need one-hot encoding
                    categorical_cols = [col for col in llm_feature_cols 
                                       if col not in ['word_count'] and 
                                       not col.endswith('_count')]
                    
                    if categorical_cols:
                        if training:
                            # Create and store one-hot encoding
                            cat_dummies = pd.get_dummies(df[categorical_cols], prefix_sep='_')
                            feature_info['llm_cat_columns'] = cat_dummies.columns.tolist()
                            llm_dummies = pd.concat([llm_dummies, cat_dummies], axis=1)
                        else:
                            # Use previously defined columns
                            try:
                                cat_dummies = pd.get_dummies(df[categorical_cols], prefix_sep='_')
                                if 'llm_cat_columns' in feature_info:
                                    for col in feature_info['llm_cat_columns']:
                                        if col not in cat_dummies.columns:
                                            cat_dummies[col] = 0
                                    cat_dummies = cat_dummies[feature_info['llm_cat_columns']]
                                    llm_dummies = pd.concat([llm_dummies, cat_dummies], axis=1)
                            except KeyError:
                                # If we're missing some columns, create empty DataFrame
                                if 'llm_cat_columns' in feature_info:
                                    cat_dummies = pd.DataFrame(0, index=df.index, 
                                                               columns=feature_info['llm_cat_columns'])
                                    llm_dummies = pd.concat([llm_dummies, cat_dummies], axis=1)
                    
                    # Add numeric columns directly
                    numeric_cols = [col for col in llm_feature_cols 
                                   if col in ['word_count'] or 
                                   col.endswith('_count')]
                    
                    for col in numeric_cols:
                        if col in df.columns:
                            llm_dummies[col] = df[col]
                    
                    # Add any binary indicator columns directly
                    binary_cols = [col for col in df.columns 
                                  if col not in categorical_cols and
                                  col not in numeric_cols and
                                  col.startswith(('attributes_', 'specifications_'))]
                    
                    for col in binary_cols:
                        llm_dummies[col] = df[col]
                    
                    logger.info(f"Generated {llm_dummies.shape[1]} LLM-derived features")
                
                # 5. Extract features from category_path if available
                if self.config['category_path_col'] in df.columns:
                    try:
                        # Parse the category path JSON
                        df['category_path_parsed'] = df[self.config['category_path_col']].apply(
                            lambda x: json.loads(x) if isinstance(x, str) and x else []
                        )
                        
                        # Extract features from the path
                        max_level = self.config['max_levels']
                        for level in range(1, max_level + 1):
                            # Create a feature for each level's category ID
                            df[f'path_id_l{level}'] = df['category_path_parsed'].apply(
                                lambda x: x[level-1]['id'] if isinstance(x, list) and len(x) >= level else None
                            )
                            
                            # One-hot encode these category IDs
                            if training:
                                path_dummies = pd.get_dummies(df[f'path_id_l{level}'], prefix=f'path_l{level}')
                                feature_info[f'path_l{level}_columns'] = path_dummies.columns.tolist()
                                llm_dummies = pd.concat([llm_dummies, path_dummies], axis=1)
                            else:
                                if f'path_l{level}_columns' in feature_info:
                                    path_dummies = pd.get_dummies(df[f'path_id_l{level}'], prefix=f'path_l{level}')
                                    for col in feature_info[f'path_l{level}_columns']:
                                        if col not in path_dummies.columns:
                                            path_dummies[col] = 0
                                    path_dummies = path_dummies[feature_info[f'path_l{level}_columns']]
                                    llm_dummies = pd.concat([llm_dummies, path_dummies], axis=1)
                        
                        logger.info(f"Generated features from category paths")
                    except Exception as e:
                        logger.warning(f"Error processing category path: {str(e)}")
            else:
                logger.warning("No LLM-derived feature columns found in the data")
        else:
            logger.info("LLM-derived features disabled in configuration")
        
        # Combine all features
        # Convert sparse matrix to dense for compatibility with other features
        dense_description = description_features.toarray()
        feature_arrays = [dense_description]
        
        # Add supplier features if not empty
        if not supplier_dummies.empty:
            feature_arrays.append(supplier_dummies.values)
        
        # Add supplier enriched features if not empty
        if not supplier_features_df.empty:
            feature_arrays.append(supplier_features_df.values)
        
        # Add LLM features if not empty
        if not llm_dummies.empty:
            feature_arrays.append(llm_dummies.values)
        
        # Stack all features horizontally
        combined_features = np.hstack(feature_arrays)
        
        logger.info(f"Final feature matrix shape: {combined_features.shape}")
        
        # Store feature information
        feature_info['n_description_features'] = description_features.shape[1]
        feature_info['n_supplier_features'] = supplier_dummies.shape[1] if not supplier_dummies.empty else 0
        feature_info['n_supplier_enriched_features'] = supplier_features_df.shape[1] if not supplier_features_df.empty else 0
        feature_info['n_llm_features'] = llm_dummies.shape[1] if not llm_dummies.empty else 0
        feature_info['total_features'] = combined_features.shape[1]
        
        return combined_features, feature_info
    
    def train_models(self, df: pd.DataFrame, supplier_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Train the categorization models for all levels using the final_category_id and level.
        
        Args:
            df: DataFrame containing product data with category assignments
            supplier_df: Optional DataFrame with supplier enriched data
            
        Returns:
            Dictionary with training results
        """
        results = {}
        
        # Check if we have the required columns
        required_cols = [
            self.config['final_category_id_col'],
            self.config['final_category_level_col']
        ]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check if we have enough data
        if len(df) < 10:
            logger.error(f"Not enough data to train models. Got only {len(df)} records.")
            raise ValueError(f"Not enough data to train models. Got only {len(df)} records.")
        
        # Prepare features
        X, feature_info = self.prepare_features(df, training=True, supplier_df=supplier_df)
        results['feature_info'] = feature_info
        
        # Create training data for each level
        max_level = self.config['max_levels']
        
        # Convert category_path to list of dicts if it's a string
        if self.config['category_path_col'] in df.columns:
            df['category_path_parsed'] = df[self.config['category_path_col']].apply(
                lambda x: json.loads(x) if isinstance(x, str) and x.strip() else []
            )
        
        # Train models for each level
        for level in range(1, max_level + 1):
            logger.info(f"Training model for level {level}")
            
            # Extract training data for this level
            level_df = df.copy()
            
            # Create target variable for this level
            if 'category_path_parsed' in level_df.columns:
                # Extract category ID for this level from the path
                level_df[f'target_category_l{level}'] = level_df['category_path_parsed'].apply(
                    lambda x: x[level-1]['id'] if isinstance(x, list) and len(x) >= level else None
                )
            else:
                # Use the final category ID where the level matches
                level_df[f'target_category_l{level}'] = level_df.apply(
                    lambda row: row[self.config['final_category_id_col']] 
                    if row[self.config['final_category_level_col']] == level 
                    else None, 
                    axis=1
                )
            
            # Drop rows with missing categories
            valid_mask = level_df[f'target_category_l{level}'].notna() & (level_df[f'target_category_l{level}'] != '')
            
            if valid_mask.sum() < 10:
                logger.warning(f"Not enough data for level {level}. Only {valid_mask.sum()} valid records.")
                results[f'level{level}'] = {
                    'trained': False,
                    'reason': f'Not enough data (only {valid_mask.sum()} records)'
                }
                continue
            
            X_valid = X[valid_mask]
            y_valid = level_df.loc[valid_mask, f'target_category_l{level}']
            
            if len(np.unique(y_valid)) < 2:
                logger.warning(f"Not enough unique categories at level {level}. Found {len(np.unique(y_valid))} categories.")
                results[f'level{level}'] = {
                    'trained': False,
                    'reason': 'Not enough unique categories'
                }
                continue
            
            # Encode labels
            self.label_encoders[f'level{level}'].fit(y_valid)
            y_encoded = self.label_encoders[f'level{level}'].transform(y_valid)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_valid, y_encoded, 
                test_size=self.config['test_size'], 
                random_state=self.config['random_state'],
                stratify=y_encoded if len(np.unique(y_encoded)) > 1 else None
            )
            
            # Create and train the model
            if self.config['model_type'] == 'random_forest':
                model = RandomForestClassifier(
                    n_estimators=self.config['n_estimators'],
                    max_depth=self.config['max_depth'],
                    random_state=self.config['random_state'],
                    n_jobs=-1,
                    class_weight='balanced'
                )
            else:
                # Default to RandomForest if unsupported model type
                logger.warning(f"Unsupported model type {self.config['model_type']}. Defaulting to random_forest.")
                model = RandomForestClassifier(
                    n_estimators=self.config['n_estimators'],
                    max_depth=self.config['max_depth'],
                    random_state=self.config['random_state'],
                    n_jobs=-1,
                    class_weight='balanced'
                )
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Evaluate the model
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # Get predicted probabilities for the predicted classes
            pred_confidences = [proba[cls] for proba, cls in zip(y_pred_proba, y_pred)]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average='weighted'
            )
            
            # Store the trained model
            self.models[f'level{level}'] = model
            
            # Store results
            results[f'level{level}'] = {
                'trained': True,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'avg_confidence': np.mean(pred_confidences),
                'n_classes': len(np.unique(y_encoded)),
                'class_distribution': np.bincount(y_encoded).tolist(),
                'class_names': self.label_encoders[f'level{level}'].classes_.tolist()
            }
            
            logger.info(f"Level {level} model trained with accuracy {accuracy:.4f}, F1 {f1:.4f}")
        
        return results
    
    def predict(self, df: pd.DataFrame, supplier_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Generate category predictions for all levels.
        
        Args:
            df: DataFrame containing product data
            supplier_df: Optional DataFrame with supplier enriched data
            
        Returns:
            DataFrame with added category predictions and confidence scores
        """
        # Check if models are trained
        if not any(self.models.values()):
            logger.error("No models trained. Call train_models() first.")
            raise ValueError("No models trained. Call train_models() first.")
        
        logger.info(f"Generating predictions for {len(df)} records")
        
        # Create output DataFrame
        output_df = df.copy()
        
        # Prepare features
        X, _ = self.prepare_features(df, training=False, supplier_df=supplier_df)
        
        # Generate predictions for each level
        for level in range(1, self.config['max_levels'] + 1):
            level_key = f'level{level}'
            if self.models[level_key] is None:
                logger.warning(f"Model for {level_key} not trained. Skipping predictions.")
                output_df[f'predicted_category_l{level}'] = None
                output_df[f'predicted_confidence_l{level}'] = 0.0
                continue
            
            # Get predictions and confidence scores
            y_pred = self.models[level_key].predict(X)
            y_pred_proba = self.models[level_key].predict_proba(X)
            
            # Get predicted probabilities for the predicted classes
            pred_confidences = [proba[pred] for proba, pred in zip(y_pred_proba, y_pred)]
            
            # Convert encoded predictions back to original categories
            y_pred_categories = self.label_encoders[level_key].inverse_transform(y_pred)
            
            # Add predictions to output DataFrame
            output_df[f'predicted_category_l{level}'] = y_pred_categories
            output_df[f'predicted_confidence_l{level}'] = pred_confidences
            
            logger.info(f"Added predictions for level {level}")
        
        # Validate hierarchical structure
        self._validate_hierarchy(output_df)
        
        # Generate final classification path
        self._generate_classification_path(output_df)
        
        return output_df
    
    def _validate_hierarchy(self, df: pd.DataFrame) -> None:
        """
        Validate and adjust predictions to ensure a valid hierarchical structure.
        
        Args:
            df: DataFrame with predictions to be validated
        """
        if self.taxonomy is None:
            logger.warning("No taxonomy available. Skipping hierarchy validation.")
            return
        
        logger.info("Validating hierarchical structure of predictions")
        
        # Create dictionaries for parent-child relationships from taxonomy
        parent_map = {}
        children_map = {}
        
        for category in self.taxonomy:
            category_id = category.get('id')
            parent_id = category.get('parent_id')
            level = category.get('level')
            
            if category_id and parent_id:
                parent_map[category_id] = parent_id
                
                if parent_id not in children_map:
                    children_map[parent_id] = []
                children_map[parent_id].append(category_id)
        
        # Check each record for hierarchical consistency (bottom-up approach)
        max_level = self.config['max_levels']
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Validating hierarchy"):
            # Start from the highest level with a prediction and work backward
            for level in range(max_level, 1, -1):
                pred_col = f'predicted_category_l{level}'
                pred_conf_col = f'predicted_confidence_l{level}'
                parent_pred_col = f'predicted_category_l{level-1}'
                parent_pred_conf_col = f'predicted_confidence_l{level-1}'
                
                if pred_col in df.columns and parent_pred_col in df.columns:
                    child_id = row[pred_col]
                    
                    # Skip if there's no prediction for this level
                    if pd.isna(child_id) or child_id == '':
                        continue
                    
                    # Check if the child has a known parent
                    if child_id in parent_map:
                        expected_parent = parent_map[child_id]
                        current_parent = row[parent_pred_col]
                        
                        # If the parent prediction doesn't match the expected parent
                        if current_parent != expected_parent:
                            child_conf = row[pred_conf_col]
                            parent_conf = row[parent_pred_conf_col]
                            
                            # Decide which to adjust based on confidence
                            if child_conf > parent_conf:
                                # Level N prediction is more confident, adjust level N-1
                                df.at[idx, parent_pred_col] = expected_parent
                                logger.debug(f"Adjusted L{level-1} category for record {idx} based on L{level} confidence")
                            else:
                                # Find a compatible child category
                                compatible_child = None
                                highest_confidence = 0
                                
                                # Get predicted probabilities for level N
                                if self.models[f'level{level}'] is not None:
                                    X_single, _ = self.prepare_features(df.iloc[[idx]], training=False)
                                    probas = self.models[f'level{level}'].predict_proba(X_single)[0]
                                    classes = self.label_encoders[f'level{level}'].classes_
                                    
                                    # Find the highest confidence child category compatible with parent
                                    for i, prob in enumerate(probas):
                                        child_id_candidate = self.label_encoders[f'level{level}'].inverse_transform([i])[0]
                                        if (child_id_candidate in parent_map and 
                                            parent_map[child_id_candidate] == current_parent and 
                                            prob > highest_confidence):
                                            compatible_child = child_id_candidate
                                            highest_confidence = prob
                                
                                if compatible_child and highest_confidence > 0.3:  # Minimum threshold
                                    df.at[idx, pred_col] = compatible_child
                                    df.at[idx, pred_conf_col] = highest_confidence
                                    logger.debug(f"Adjusted L{level} category for record {idx} to be compatible with L{level-1}")
                                elif expected_parent in children_map:  
                                    # No compatible child found with sufficient confidence
                                    # If we can't find a compatible child, we might need to nullify the child prediction
                                    df.at[idx, pred_col] = None
                                    df.at[idx, pred_conf_col] = 0.0
                                    logger.debug(f"Removed incompatible L{level} prediction for record {idx}")
        
        logger.info("Hierarchy validation completed")
    
    def _generate_classification_path(self, df: pd.DataFrame) -> None:
        """
        Generate classification paths from predicted categories.
        
        Args:
            df: DataFrame with predictions
        """
        logger.info("Generating classification paths")
        
        # Create a column for the classification path
        df['predicted_category_path'] = None
        
        # For each record, build the path from level 1 to the deepest level with a prediction
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating paths"):
            path = []
            final_level = 0
            
            # Build the path starting from level 1
            for level in range(1, self.config['max_levels'] + 1):
                pred_col = f'predicted_category_l{level}'
                conf_col = f'predicted_confidence_l{level}'
                
                if pred_col in df.columns and pd.notna(row[pred_col]) and row[pred_col] != '':
                    # Get category details from taxonomy if available
                    category_info = {
                        'id': row[pred_col],
                        'level': level,
                        'confidence': row[conf_col] if conf_col in df.columns else None
                    }
                    
                    # Add name and description if we have taxonomy information
                    if self.taxonomy is not None:
                        for cat in self.taxonomy:
                            if cat.get('id') == row[pred_col]:
                                category_info['name'] = cat.get('name')
                                category_info['description'] = cat.get('description')
                                category_info['code'] = cat.get('code')
                                break
                    
                    path.append(category_info)
                    final_level = level
                else:
                    # Stop when we reach a level without a prediction
                    break
            
            # Store the path and final level
            if path:
                df.at[idx, 'predicted_category_path'] = json.dumps(path)
                df.at[idx, 'predicted_final_level'] = final_level
                df.at[idx, 'predicted_final_category'] = path[-1]['id'] if path else None
        
        logger.info("Classification paths generated")
    
    def generate_confidence_metrics(self, predictions_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate metrics on prediction confidence.
        
        Args:
            predictions_df: DataFrame with predictions
            
        Returns:
            Dictionary with confidence metrics
        """
        metrics = {}
        
        for level in range(1, self.config['max_levels'] + 1):
            conf_col = f'predicted_confidence_l{level}'
            if conf_col not in predictions_df.columns:
                continue
                
            level_metrics = {
                'avg_confidence': predictions_df[conf_col].mean(),
                'median_confidence': predictions_df[conf_col].median(),
                'min_confidence': predictions_df[conf_col].min(),
                'max_confidence': predictions_df[conf_col].max(),
                'low_confidence_count': (predictions_df[conf_col] < self.config['min_confidence_threshold']).sum(),
                'low_confidence_percent': (predictions_df[conf_col] < self.config['min_confidence_threshold']).mean() * 100
            }
            
            metrics[f'level{level}'] = level_metrics
        
        return metrics
    
    def save_models(self, output_dir: Optional[str] = None) -> None:
        """
        Save trained models and related artifacts.
        
        Args:
            output_dir: Directory to save models (default from config)
        """
        if output_dir is None:
            output_dir = self.config['output_dir']
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save models
        for level, model in self.models.items():
            if model is not None:
                model_path = os.path.join(output_dir, f'{level}_model.joblib')
                joblib.dump(model, model_path)
                logger.info(f"Saved {level} model to {model_path}")
        
        # Save label encoders
        for level, encoder in self.label_encoders.items():
            if hasattr(encoder, 'classes_') and len(encoder.classes_) > 0:
                encoder_path = os.path.join(output_dir, f'{level}_encoder.joblib')
                joblib.dump(encoder, encoder_path)
                logger.info(f"Saved {level} label encoder to {encoder_path}")
        
        # Save vectorizers
        for name, vectorizer in self.vectorizers.items():
            if vectorizer is not None:
                vectorizer_path = os.path.join(output_dir, f'{name}_vectorizer.joblib')
                joblib.dump(vectorizer, vectorizer_path)
                logger.info(f"Saved {name} vectorizer to {vectorizer_path}")
        
        # Save configuration
        config_path = os.path.join(output_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        logger.info(f"Saved configuration to {config_path}")
    
    def load_models(self, model_dir: str) -> bool:
        """
        Load trained models and related artifacts.
        
        Args:
            model_dir: Directory containing saved models
            
        Returns:
            Boolean indicating success
        """
        try:
            # Load configuration
            config_path = os.path.join(model_dir, 'config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.config.update(json.load(f))
                logger.info(f"Loaded configuration from {config_path}")
            
            # Load models
            for level in [f'level{i}' for i in range(1, self.config['max_levels'] + 1)]:
                model_path = os.path.join(model_dir, f'{level}_model.joblib')
                if os.path.exists(model_path):
                    self.models[level] = joblib.load(model_path)
                    logger.info(f"Loaded {level} model from {model_path}")
            
            # Load label encoders
            for level in [f'level{i}' for i in range(1, self.config['max_levels'] + 1)]:
                encoder_path = os.path.join(model_dir, f'{level}_encoder.joblib')
                if os.path.exists(encoder_path):
                    self.label_encoders[level] = joblib.load(encoder_path)
                    logger.info(f"Loaded {level} label encoder from {encoder_path}")
            
            # Load vectorizers
            for name in ['description']:
                vectorizer_path = os.path.join(model_dir, f'{name}_vectorizer.joblib')
                if os.path.exists(vectorizer_path):
                    self.vectorizers[name] = joblib.load(vectorizer_path)
                    logger.info(f"Loaded {name} vectorizer from {vectorizer_path}")
            
            return any(model is not None for model in self.models.values())
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False
    
    def visualize_category_distribution(self, df: pd.DataFrame, output_path: Optional[str] = None) -> None:
        """
        Visualize the distribution of categories.
        
        Args:
            df: DataFrame with category assignments
            output_path: Optional path to save visualization
        """
        # Create a figure with subplots for each level
        fig, axes = plt.subplots(self.config['max_levels'], 1, figsize=(12, 6 * self.config['max_levels']))
        
        # Ensure axes is a list for consistent indexing
        if self.config['max_levels'] == 1:
            axes = [axes]
        
        for i, level in enumerate(range(1, self.config['max_levels'] + 1)):
            # Get category count from either final predictions or target values
            if f'predicted_category_l{level}' in df.columns:
                category_col = f'predicted_category_l{level}'
            elif f'target_category_l{level}' in df.columns:
                category_col = f'target_category_l{level}'
            else:
                # Skip if we don't have categories for this level
                axes[i].text(0.5, 0.5, f"No data for Level {level}", 
                            ha='center', va='center', fontsize=14)
                continue
            
            # Get category counts
            category_counts = df[category_col].value_counts().sort_values(ascending=False)
            
            # Limit to top 20 for readability
            if len(category_counts) > 20:
                category_counts = category_counts.head(20)
                title_suffix = " (Top 20)"
            else:
                title_suffix = ""
            
            # Plot
            sns.barplot(x=category_counts.values, y=category_counts.index, ax=axes[i])
            axes[i].set_title(f"Level {level} Category Distribution{title_suffix}")
            axes[i].set_xlabel("Count")
            axes[i].set_ylabel("Category")
            
            # Add count labels
            for j, v in enumerate(category_counts.values):
                axes[i].text(v + 0.5, j, str(v), va='center')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            logger.info(f"Saved category distribution visualization to {output_path}")
        
        plt.close()
    
    def visualize_feature_importance(self, level: int = 1, top_n: int = 20, 
                                     output_path: Optional[str] = None) -> None:
        """
        Visualize feature importance for a specific level.
        
        Args:
            level: The category level to visualize
            top_n: Number of top features to show
            output_path: Optional path to save visualization
        """
        level_key = f'level{level}'
        
        if self.models[level_key] is None:
            logger.warning(f"No trained model for {level_key}")
            return
        
        # Extract feature importance
        importance = self.models[level_key].feature_importances_
        
        # Get feature names
        feature_names = []
        
        # Description features (from TF-IDF)
        if self.vectorizers['description'] is not None:
            feature_names.extend(self.vectorizers['description'].get_feature_names_out())
        
        # If we don't have enough feature names, use generic names
        if len(feature_names) < len(importance):
            feature_names = [f"Feature {i}" for i in range(len(importance))]
        
        # Create DataFrame for visualization
        importance_df = pd.DataFrame({
            'Feature': feature_names[:len(importance)],
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        # Limit to top N features
        importance_df = importance_df.head(top_n)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title(f"Top {top_n} Feature Importance for Level {level}")
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            logger.info(f"Saved feature importance visualization to {output_path}")
        
        plt.close()
    
    def run_categorization_pipeline(self, training_data: pd.DataFrame, 
                                   test_data: Optional[pd.DataFrame] = None,
                                   supplier_data: Optional[pd.DataFrame] = None,
                                   output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the complete categorization pipeline.
        
        Args:
            training_data: DataFrame with category assignments for training
            test_data: Optional DataFrame for prediction
            supplier_data: Optional DataFrame with supplier enriched data
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary with pipeline results
        """
        if output_dir is None:
            output_dir = self.config['output_dir']
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Train models
            logger.info("Training categorization models")
            training_results = self.train_models(training_data, supplier_data)
            
            # Save models
            self.save_models(output_dir)
            
            # Generate visualizations
            os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
            
            # Category distribution
            self.visualize_category_distribution(
                training_data,
                os.path.join(output_dir, 'visualizations', 'category_distribution.png')
            )
            
            # Feature importance for each level
            for level in range(1, self.config['max_levels'] + 1):
                if self.models[f'level{level}'] is not None:
                    self.visualize_feature_importance(
                        level=level,
                        output_path=os.path.join(output_dir, 'visualizations', f'feature_importance_l{level}.png')
                    )
            
            # Run predictions if test data provided
            prediction_results = None
            if test_data is not None:
                logger.info(f"Generating predictions for {len(test_data)} test records")
                predictions_df = self.predict(test_data, supplier_data)
                
                # Save predictions
                predictions_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
                
                # Generate confidence metrics
                confidence_metrics = self.generate_confidence_metrics(predictions_df)
                
                prediction_results = {
                    'predictions_df': predictions_df,
                    'confidence_metrics': confidence_metrics
                }
            
            logger.info("Categorization pipeline completed successfully")
            
            return {
                'training_results': training_results,
                'prediction_results': prediction_results,
                'models': self.models,
                'config': self.config
            }
            
        except Exception as e:
            logger.error(f"Error in categorization pipeline: {str(e)}")
            raise

# Example usage
if __name__ == '__main__':
    # Sample usage of the categorization module
    
    # Load enriched data (assuming it was generated by ProcurementDataEnrichment)
    try:
        enriched_products = pd.read_csv('output/enriched2/enriched_products.csv')
        enriched_suppliers = pd.read_csv('output/enriched2/enriched_suppliers.csv')
        taxonomy_path = 'output/enriched2/taxonomy.json'
        
        # Configure categorization module
        config = {
            'output_dir': 'output/models',
            'taxonomy_path': taxonomy_path,
            'use_llm_features': True,
            'use_supplier_features': True,
            'min_confidence_threshold': 0.6,
            'max_levels': 1
        }
        
        # Initialize categorization module
        categorizer = ProcurementCategorization(config)
        
        # Split data for training and testing
        from sklearn.model_selection import train_test_split
        train_data, test_data = train_test_split(
            enriched_products, test_size=0.2, random_state=42
        )
        
        # Run categorization pipeline
        results = categorizer.run_categorization_pipeline(
            training_data=train_data,
            test_data=test_data,
            supplier_data=enriched_suppliers,
            output_dir='output/models'
        )
        
        # Print summary
        for level in range(1, 6):  # Up to 5 levels supported
            level_key = f'level{level}'
            if level_key in results['training_results'] and results['training_results'][level_key].get('trained', False):
                print(f"Level {level} model:")
                print(f"  - Accuracy: {results['training_results'][level_key]['accuracy']:.4f}")
                print(f"  - F1 Score: {results['training_results'][level_key]['f1']:.4f}")
                print(f"  - Classes: {results['training_results'][level_key]['n_classes']}")
        
        if results['prediction_results']:
            for level in range(1, 6):
                level_key = f'level{level}'
                if level_key in results['prediction_results']['confidence_metrics']:
                    metrics = results['prediction_results']['confidence_metrics'][level_key]
                    print(f"Level {level} predictions:")
                    print(f"  - Average confidence: {metrics['avg_confidence']:.4f}")
                    print(f"  - Low confidence predictions: {metrics['low_confidence_percent']:.2f}%")
        
    except FileNotFoundError:
        print("Required files not found. Please run data enrichment first.")
    except Exception as e:
        print(f"Error in example: {str(e)}")
"""
PAVUS AI Procurement Data Pipeline

This module provides a complete data ingestion and preprocessing pipeline 
for procurement transaction data, preparing it for AI-driven categorization.

Features:
- Data loading from various formats (CSV, Excel)
- Data validation and cleaning
- Supplier name standardization
- Product description normalization and feature extraction
- Creation of supplier and product reference tables
"""

import pandas as pd
import numpy as np
import re
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("procurement_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('procurement_pipeline')

class ProcurementDataPipeline:
    """
    A complete pipeline for loading, cleaning, and preprocessing procurement data.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the pipeline with configuration settings.
        
        Args:
            config: Dictionary containing configuration parameters.
        """
        self.config = config or {}
        self.raw_data = None
        self.processed_data = None
        self.supplier_reference = None
        self.product_reference = None
        
        # Load default configuration if not provided
        self._set_default_config()
        
        logger.info("Procurement Data Pipeline initialized")
    
    def _set_default_config(self):
        """Set default configuration parameters if not provided."""
        # Define date columns and their format
        self.config.setdefault('date_columns', ['GL Posting Date'])
        self.config.setdefault('date_format', '%Y-%m-%d')
        
        # Define supplier columns
        self.config.setdefault('supplier_id_column', 'Vendor ID')
        self.config.setdefault('supplier_name_column', 'Vendor Name')
        
        # Define product columns
        self.config.setdefault('product_id_column', 'Item Number')
        self.config.setdefault('product_desc_column', 'Item Description')
        
        # Define cost columns
        self.config.setdefault('unit_cost_column', 'Unit Cost')
        self.config.setdefault('extended_cost_column', 'Extended Cost')
        
        # Define quantity columns
        self.config.setdefault('qty_shipped_column', 'QTY Shipped')
        self.config.setdefault('qty_invoiced_column', 'QTY Invoiced')
        
        # Define columns to keep in processed data
        self.config.setdefault('columns_to_keep', [
            'GL Posting Date', 'Item Number', 'Item Description',
            'QTY Shipped', 'QTY Invoiced', 'Unit Cost', 'Extended Cost',
            'Vendor ID', 'Vendor Name', 'PO Number'
        ])
    
    def load_data(self, file_path: str, file_type: Optional[str] = None) -> pd.DataFrame:
        """
        Load procurement data from file.
        
        Args:
            file_path: Path to the data file.
            file_type: Type of file ('csv', 'excel', etc.). If None, inferred from extension.
            
        Returns:
            DataFrame containing the loaded data.
        """
        if file_type is None:
            # Infer file type from extension
            _, extension = os.path.splitext(file_path)
            file_type = extension.lower().replace('.', '')
        
        try:
            if file_type in ['csv', 'txt']:
                # Try different encodings and delimiters
                try:
                    self.raw_data = pd.read_csv(file_path, encoding='utf-8')
                except UnicodeDecodeError:
                    try:
                        self.raw_data = pd.read_csv(file_path, encoding='latin1')
                    except:
                        # Try comma and tab delimiters
                        try:
                            self.raw_data = pd.read_csv(file_path, delimiter=',', encoding='latin1')
                        except:
                            self.raw_data = pd.read_csv(file_path, delimiter='\t', encoding='latin1')
            
            elif file_type in ['xlsx', 'xls']:
                self.raw_data = pd.read_excel(file_path)
            
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Initial validation of loaded data
            if self.raw_data.empty:
                raise ValueError("Loaded data is empty")
            
            logger.info(f"Successfully loaded data from {file_path} with {len(self.raw_data)} rows")
            return self.raw_data
            
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {str(e)}")
            raise
    
    def validate_data(self) -> Tuple[bool, List[str]]:
        """
        Perform initial validation checks on the raw data.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        if self.raw_data is None:
            return False, ["No data loaded. Call load_data() first."]
        
        errors = []
        
        # Check required columns
        required_columns = [
            self.config['supplier_id_column'],
            self.config['supplier_name_column'],
            self.config['product_id_column'],
            self.config['product_desc_column']
        ]
        
        missing_columns = [col for col in required_columns if col not in self.raw_data.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {', '.join(missing_columns)}")
        
        # Check for completely empty columns
        empty_columns = [col for col in self.raw_data.columns 
                         if self.raw_data[col].isna().all()]
        if empty_columns:
            errors.append(f"Completely empty columns: {', '.join(empty_columns)}")
        
        # Check for duplicate transactions (if we have a unique identifier)
        if 'POP Receipt Number' in self.raw_data.columns and 'Item Number' in self.raw_data.columns:
            potential_dupes = self.raw_data.duplicated(subset=['POP Receipt Number', 'Item Number'], keep=False)
            dupe_count = potential_dupes.sum()
            if dupe_count > 0:
                errors.append(f"Found {dupe_count} potential duplicate transactions")
        
        is_valid = len(errors) == 0
        
        if is_valid:
            logger.info("Data validation passed")
        else:
            logger.warning(f"Data validation failed with {len(errors)} issues")
            for error in errors:
                logger.warning(f"Validation error: {error}")
        
        return is_valid, errors
    
    def clean_data(self) -> pd.DataFrame:
        """
        Clean the raw data by fixing data types, handling missing values,
        and performing basic transformations.
        
        Returns:
            Cleaned DataFrame
        """
        if self.raw_data is None:
            logger.error("No data loaded. Call load_data() first.")
            raise ValueError("No data loaded")
        
        logger.info("Starting data cleaning process")
        
        # Create a copy to avoid modifying the original
        df = self.raw_data.copy()
        
        # Handle missing values
        for col in df.columns:
            null_count = df[col].isna().sum()
            if null_count > 0:
                logger.info(f"Found {null_count} missing values in column {col}")
                
                # For numeric columns, fill with 0
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(0)
                
                # For string columns, fill with empty string
                elif pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                    df[col] = df[col].fillna('')
        
        # Convert date columns
        for date_col in self.config['date_columns']:
            if date_col in df.columns:
                try:
                    df[date_col] = pd.to_datetime(df[date_col], format=self.config['date_format'])
                except Exception as e:
                    logger.warning(f"Failed to convert {date_col} to datetime: {str(e)}")
        
        # Ensure numeric columns are numeric
        numeric_columns = [
            self.config['unit_cost_column'],
            self.config['extended_cost_column'],
            self.config['qty_shipped_column'],
            self.config['qty_invoiced_column']
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Ensure string columns are strings
        string_columns = [
            self.config['supplier_id_column'],
            self.config['supplier_name_column'],
            self.config['product_id_column'],
            self.config['product_desc_column']
        ]
        
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].astype(str)
        
        # Keep only relevant columns if specified
        if 'columns_to_keep' in self.config:
            # Filter to only include columns that exist in the DataFrame
            valid_columns = [col for col in self.config['columns_to_keep'] if col in df.columns]
            df = df[valid_columns]
        
        # Store the cleaned data
        self.processed_data = df
        
        logger.info(f"Data cleaning completed. Processed data has {len(df)} rows and {len(df.columns)} columns")
        return df

    def standardize_supplier_names(self) -> pd.DataFrame:
        """
        Create standardized supplier names to handle variations of the same supplier.
        
        Returns:
            DataFrame with standardized supplier information
        """
        if self.processed_data is None:
            logger.warning("No processed data available. Running clean_data() first.")
            self.clean_data()
        
        df = self.processed_data.copy()
        supplier_col = self.config['supplier_name_column']
        supplier_id_col = self.config['supplier_id_column']
        
        if supplier_col not in df.columns:
            logger.error(f"Supplier column {supplier_col} not found in data")
            raise ValueError(f"Supplier column {supplier_col} not found in data")
        
        logger.info("Starting supplier name standardization")
        
        # Extract unique suppliers
        suppliers = df[[supplier_id_col, supplier_col]].drop_duplicates()
        
        # Basic standardization
        suppliers['standard_name'] = suppliers[supplier_col].apply(self._standardize_name)
        
        # Create mapping dictionary from original to standardized names
        name_mapping = dict(zip(suppliers[supplier_col], suppliers['standard_name']))
        
        # Apply standardization to the main dataframe
        df['standardized_supplier'] = df[supplier_col].map(name_mapping)
        
        # Create supplier reference table
        self.supplier_reference = suppliers.copy()
        
        logger.info(f"Supplier standardization complete. Found {len(suppliers)} unique suppliers")
        
        # Update processed data
        self.processed_data = df
        
        return self.supplier_reference
    
    def _standardize_name(self, name: str) -> str:
        """
        Apply standardization rules to a supplier name.
        
        Args:
            name: Original supplier name
            
        Returns:
            Standardized name
        """
        if not isinstance(name, str):
            return str(name)
        
        # Convert to uppercase
        std_name = name.upper()
        
        # Remove common legal entity indicators
        entity_types = [' INC', ' LLC', ' LTD', ' CORP', ' CO', ' COMPANY', 
                        ' CORPORATION', ' INCORPORATED']
        for entity in entity_types:
            std_name = std_name.replace(entity, '')
        
        # Remove punctuation
        std_name = re.sub(r'[^\w\s]', '', std_name)
        
        # Remove extra whitespace
        std_name = ' '.join(std_name.split())
        
        return std_name

    def extract_product_features(self) -> pd.DataFrame:
        """
        Extract features from product descriptions to aid in categorization.
        
        Returns:
            DataFrame with product features
        """
        if self.processed_data is None:
            logger.warning("No processed data available. Running clean_data() first.")
            self.clean_data()
        
        df = self.processed_data.copy()
        product_id_col = self.config['product_id_column']
        product_desc_col = self.config['product_desc_column']
        
        logger.info("Starting product feature extraction")
        
        # Extract unique products
        products = df[[product_id_col, product_desc_col]].drop_duplicates()
        
        # Clean descriptions
        products['clean_description'] = products[product_desc_col].apply(
            lambda x: ' '.join(str(x).upper().split()) if pd.notna(x) else '')
        
        # Extract basic features
        products['word_count'] = products['clean_description'].apply(lambda x: len(x.split()))
        products['extracted_keywords'] = products['clean_description'].apply(self._extract_keywords)
        
        # Create product reference
        self.product_reference = products
        
        logger.info(f"Product feature extraction complete. Found {len(products)} unique products")
        
        return self.product_reference
    
    def _extract_keywords(self, description: str) -> List[str]:
        """
        Extract key terms from product description.
        
        Args:
            description: Product description
            
        Returns:
            List of extracted keywords
        """
        if not isinstance(description, str) or not description:
            return []
        
        # Remove common filler words
        filler_words = ['THE', 'AND', 'FOR', 'WITH', 'A', 'OF', 'TO', 'IN', 'ON']
        words = description.split()
        keywords = [word for word in words if word not in filler_words and len(word) > 2]
        
        return keywords
    
    def create_spend_analysis_base(self) -> pd.DataFrame:
        """
        Create the base dataset for spend analysis, integrating all processed data.
        
        Returns:
            DataFrame ready for categorization
        """
        if self.processed_data is None:
            logger.warning("No processed data. Running full preprocessing pipeline.")
            self.clean_data()
            self.standardize_supplier_names()
            self.extract_product_features()
        
        df = self.processed_data.copy()
        
        # Add supplier standardization if not already done
        if 'standardized_supplier' not in df.columns:
            self.standardize_supplier_names()
            df = self.processed_data.copy()
        
        # Merge product features
        if self.product_reference is not None:
            product_features = self.product_reference[[
                self.config['product_id_column'], 
                'clean_description', 
                'extracted_keywords'
            ]]
            
            df = df.merge(
                product_features,
                on=self.config['product_id_column'],
                how='left'
            )
        
        logger.info("Created base dataset for spend analysis and categorization")
        
        return df
    
    def save_processed_data(self, output_path: str) -> None:
        """
        Save the processed data to disk.
        
        Args:
            output_path: Path to save the data
        """
        if self.processed_data is None:
            logger.error("No processed data to save")
            raise ValueError("No processed data to save")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Determine file format from extension
        _, extension = os.path.splitext(output_path)
        if extension.lower() == '.csv':
            self.processed_data.to_csv(output_path, index=False)
        elif extension.lower() in ['.xlsx', '.xls']:
            self.processed_data.to_excel(output_path, index=False)
        else:
            logger.warning(f"Unknown file extension {extension}, defaulting to CSV")
            self.processed_data.to_csv(output_path, index=False)
        
        logger.info(f"Saved processed data to {output_path}")
    
    def save_reference_tables(self, output_dir: str) -> None:
        """
        Save supplier and product reference tables.
        
        Args:
            output_dir: Directory to save the tables
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if self.supplier_reference is not None:
            supplier_path = os.path.join(output_dir, 'supplier_reference.csv')
            self.supplier_reference.to_csv(supplier_path, index=False)
            logger.info(f"Saved supplier reference to {supplier_path}")
        
        if self.product_reference is not None:
            product_path = os.path.join(output_dir, 'product_reference.csv')
            self.product_reference.to_csv(product_path, index=False)
            logger.info(f"Saved product reference to {product_path}")
    
    def run_full_pipeline(self, file_path: str, output_dir: str) -> Dict[str, pd.DataFrame]:
        """
        Run the complete data pipeline from raw data to processed outputs.
        
        Args:
            file_path: Path to input data file
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary containing all processed DataFrames
        """
        try:
            # Load data
            self.load_data(file_path)
            
            # Validate data
            is_valid, errors = self.validate_data()
            if not is_valid:
                logger.warning(f"Data validation issues: {errors}")
                logger.info("Proceeding with pipeline despite validation issues")
            
            # Clean data
            self.clean_data()
            
            # Process suppliers
            self.standardize_supplier_names()
            
            # Process products
            self.extract_product_features()
            
            # Create spend analysis base
            spend_analysis_base = self.create_spend_analysis_base()
            
            # Save outputs
            os.makedirs(output_dir, exist_ok=True)
            self.save_processed_data(os.path.join(output_dir, 'processed_data.csv'))
            self.save_reference_tables(output_dir)
            
            logger.info("Full pipeline completed successfully")
            
            return {
                'raw_data': self.raw_data,
                'processed_data': self.processed_data,
                'supplier_reference': self.supplier_reference,
                'product_reference': self.product_reference,
                'spend_analysis_base': spend_analysis_base
            }
            
        except Exception as e:
            logger.error(f"Error in pipeline: {str(e)}")
            raise


# Example usage
if __name__ == '__main__':
    # Initialize pipeline
    pipeline = ProcurementDataPipeline()
    
    # Run pipeline on sample data
    results = pipeline.run_full_pipeline(
        file_path='data/data.csv',
        output_dir='processed_data'
    )
    
    # Display summary
    print(f"Processed {len(results['processed_data'])} transactions")
    print(f"Found {len(results['supplier_reference'])} unique suppliers")
    print(f"Found {len(results['product_reference'])} unique products")
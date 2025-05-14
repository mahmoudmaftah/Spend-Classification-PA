"""
PAVUS AI Procurement Data Enrichment

This module enhances the preprocessed procurement data by leveraging OpenAI's API
to extract additional attributes and context for suppliers and products.

Features:
- Supplier attribute extraction (industry, market segment, size)
- Product attribute extraction (category indicators, specifications, function)
- Classification suggestions based on product descriptions
"""

import pandas as pd
import numpy as np
import os
import json
import time
from typing import Dict, List, Optional, Tuple, Union
import logging
from dotenv import load_dotenv
import openai
import re
import concurrent.futures
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("procurement_enrichment.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('procurement_enrichment')

class ProcurementDataEnrichment:
    """
    A module for enriching procurement data using OpenAI's API.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4-turbo"):
        """
        Initialize the enrichment module.
        
        Args:
            api_key: OpenAI API key (if None, will look for OPENAI_API_KEY in environment)
            model: OpenAI model to use for enrichment
        """
        # Load environment variables from .env file if it exists
        load_dotenv()
        
        # Use provided API key or get from environment
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("No OpenAI API key provided. Set OPENAI_API_KEY environment variable or pass key to constructor.")
        
        self.model = model
        self.client = openai.OpenAI(api_key=self.api_key) if self.api_key else None
        
        # Rate limiting parameters
        self.request_delay = 0.1  # seconds between requests
        self.max_retries = 5
        self.retry_delay = 2  # seconds between retries
        self.max_workers = 5  # maximum number of concurrent requests
        
        logger.info(f"Procurement Data Enrichment initialized with model {model}")
    
    def enrich_suppliers(self, supplier_df: pd.DataFrame, supplier_name_col: str = 'Vendor Name') -> pd.DataFrame:
        """
        Enrich supplier data with additional attributes using OpenAI.
        
        Args:
            supplier_df: DataFrame containing supplier information
            supplier_name_col: Column containing supplier names
            
        Returns:
            DataFrame with enriched supplier data
        """
        if not self.client:
            logger.error("OpenAI client not initialized. Provide a valid API key.")
            raise ValueError("OpenAI client not initialized")
        
        logger.info(f"Starting supplier enrichment for {len(supplier_df)} suppliers")
        
        # Create a copy of the input dataframe
        enriched_df = supplier_df.copy()
        
        # Add columns for enriched data
        enriched_df['industry'] = None
        enriched_df['market_segment'] = None
        enriched_df['likely_size'] = None
        enriched_df['product_categories'] = None
        enriched_df['confidence_score'] = None
        
        enriched_df['year_founded'] = None
        enriched_df['public_private_status'] = None
        enriched_df['competitive_positioning'] = None
        enriched_df['sustainability_rating'] = None
        enriched_df['digital_presence_score'] = None
        enriched_df['supplier_risk_level'] = None
        enriched_df['financial_health'] = None
        enriched_df['innovation_score'] = None
        enriched_df['customer_base'] = None
        enriched_df['global_reach'] = None
        enriched_df['pricing_tier'] = None
        
        # Define the enrichment function for a single supplier
        def enrich_single_supplier(supplier_name):
            prompt = f"""
            As a procurement data analyst, analyze the following supplier name and provide key business attributes.
            
            Supplier Name: {supplier_name}
            
            Please return ONLY a JSON object with the following fields:
            - industry: The primary industry of this supplier
            - market_segment: The market segment they likely serve
            - likely_size: Likely size of the business (Small, Medium, Large, or Enterprise)
            - product_categories: A list of up to 5 product categories they likely provide based on their name and industry
            - confidence_score: Your confidence in this assessment (0.0-1.0)
            - year_founded: Estimated year of establishment (or "Unknown")
            - public_private_status: Whether likely public or private
            - competitive_positioning: Market leader, challenger, niche player, etc.
            - sustainability_rating: Score from 1-10 on likely sustainability practices
            - digital_presence_score: Score from 1-10 on digital presence strength
            - supplier_risk_level: Low, Medium, or High
            - financial_health: Score from 1-10 on estimated financial stability
            - innovation_score: Score from 1-10 on innovation level
            - customer_base: B2B, B2C, or Both
            - global_reach: Local, Regional, National, or Multinational
            - pricing_tier: Budget, Mid-range, or Premium
            
            Return ONLY the JSON object with no preamble or explanation.
            """
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a procurement data analysis assistant that provides structured information."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2
                )
                
                # Extract the JSON from the response
                result_text = response.choices[0].message.content.strip()
                
                # Clean up the response to ensure it's valid JSON
                result_text = result_text.replace("```json", "").replace("```", "").strip()
                
                # Parse the JSON
                result = json.loads(result_text)
                
                # Add a small delay to avoid rate limiting
                time.sleep(self.request_delay)
                
                return result
                
            except Exception as e:
                logger.warning(f"Error enriching supplier '{supplier_name}': {str(e)}")
                return {
                    "industry": None,
                    "market_segment": None,
                    "likely_size": None,
                    "product_categories": [],
                    "confidence_score": 0.0,
                    "year_founded": None,
                    "public_private_status": None,
                    "competitive_positioning": None,
                    "sustainability_rating": None,
                    "digital_presence_score": None,
                    "supplier_risk_level": None,
                    "financial_health": None,
                    "innovation_score": None,
                    "customer_base": None,
                    "global_reach": None,
                    "pricing_tier": None
                }
        
        # Process suppliers in parallel with a thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all suppliers to the thread pool
            future_to_index = {
                executor.submit(enrich_single_supplier, row[supplier_name_col]): i 
                for i, row in enriched_df.iterrows()
            }
            
            # Process results as they complete
            for future in tqdm(concurrent.futures.as_completed(future_to_index), 
                              total=len(enriched_df), 
                              desc="Enriching suppliers"):
                index = future_to_index[future]
                try:
                    result = future.result()
                    enriched_df.at[index, 'industry'] = result.get('industry')
                    enriched_df.at[index, 'market_segment'] = result.get('market_segment')
                    enriched_df.at[index, 'likely_size'] = result.get('likely_size')
                    enriched_df.at[index, 'product_categories'] = json.dumps(result.get('product_categories', []))
                    enriched_df.at[index, 'confidence_score'] = result.get('confidence_score', 0.0)
                    enriched_df.at[index, 'year_founded'] = result.get('year_founded')
                    enriched_df.at[index, 'public_private_status'] = result.get('public_private_status')
                    enriched_df.at[index, 'competitive_positioning'] = result.get('competitive_positioning')
                    enriched_df.at[index, 'sustainability_rating'] = result.get('sustainability_rating')
                    enriched_df.at[index, 'digital_presence_score'] = result.get('digital_presence_score')
                    enriched_df.at[index, 'supplier_risk_level'] = result.get('supplier_risk_level')
                    enriched_df.at[index, 'financial_health'] = result.get('financial_health')
                    enriched_df.at[index, 'innovation_score'] = result.get('innovation_score')
                    enriched_df.at[index, 'customer_base'] = result.get('customer_base')
                    enriched_df.at[index, 'global_reach'] = result.get('global_reach')
                    enriched_df.at[index, 'pricing_tier'] = result.get('pricing_tier')
                except Exception as e:
                    logger.error(f"Error processing supplier at index {index}: {str(e)}")
        
        logger.info(f"Supplier enrichment completed for {len(enriched_df)} suppliers")
        
        return enriched_df
    
    def enrich_products(self, product_df: pd.DataFrame, 
                        product_id_col: str = 'Item Number', 
                        product_desc_col: str = 'Item Description') -> pd.DataFrame:
        """
        Enrich product data with additional attributes using OpenAI.
        
        Args:
            product_df: DataFrame containing product information
            product_id_col: Column containing product IDs
            product_desc_col: Column containing product descriptions
            
        Returns:
            DataFrame with enriched product data
        """
        if not self.client:
            logger.error("OpenAI client not initialized. Provide a valid API key.")
            raise ValueError("OpenAI client not initialized")
        
        logger.info(f"Starting product enrichment for {len(product_df)} products")
        
        # Create a copy of the input dataframe
        enriched_df = product_df.copy()
        
        # Add columns for enriched data
        enriched_df['category_l1'] = None
        enriched_df['category_l2'] = None
        enriched_df['category_l3'] = None
        enriched_df['attributes'] = None
        enriched_df['specifications'] = None
        enriched_df['primary_function'] = None
        enriched_df['confidence_score'] = None
        
        # Define the enrichment function for a single product
        def enrich_single_product(product_id, product_desc):
            prompt = f"""
            As a procurement data analyst, analyze the following product description and provide categorization and attributes.
            
            Product ID: {product_id}
            Product Description: {product_desc}
            
            Please return ONLY a JSON object with the following fields:
            - category_l1: The primary category of this product (high level)
            - category_l2: The secondary category (more specific)
            - category_l3: The tertiary category (most specific)
            - attributes: A list of key attributes extracted from the description
            - specifications: Any technical specifications mentioned
            - primary_function: The main function or purpose of this product
            - confidence_score: Your confidence in this assessment (0.0-1.0)
            
            Return ONLY the JSON object with no preamble or explanation.
            """
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a procurement data analysis assistant that provides structured information."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2
                )
                
                # Extract the JSON from the response
                result_text = response.choices[0].message.content.strip()
                
                # Clean up the response to ensure it's valid JSON
                result_text = result_text.replace("```json", "").replace("```", "").strip()
                
                # Parse the JSON
                result = json.loads(result_text)
                
                # Add a small delay to avoid rate limiting
                time.sleep(self.request_delay)
                
                return result
                
            except Exception as e:
                logger.warning(f"Error enriching product '{product_id}': {str(e)}")
                return {
                    "category_l1": None,
                    "category_l2": None,
                    "category_l3": None,
                    "attributes": [],
                    "specifications": [],
                    "primary_function": None,
                    "confidence_score": 0.0
                }
        
        # Process products in parallel with a thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all products to the thread pool
            future_to_index = {
                executor.submit(enrich_single_product, row[product_id_col], row[product_desc_col]): i 
                for i, row in enriched_df.iterrows()
            }
            
            # Process results as they complete
            for future in tqdm(concurrent.futures.as_completed(future_to_index), 
                              total=len(enriched_df), 
                              desc="Enriching products"):
                index = future_to_index[future]
                try:
                    result = future.result()
                    enriched_df.at[index, 'category_l1'] = result.get('category_l1')
                    enriched_df.at[index, 'category_l2'] = result.get('category_l2')
                    enriched_df.at[index, 'category_l3'] = result.get('category_l3')
                    enriched_df.at[index, 'attributes'] = json.dumps(result.get('attributes', []))
                    enriched_df.at[index, 'specifications'] = json.dumps(result.get('specifications', []))
                    enriched_df.at[index, 'primary_function'] = result.get('primary_function')
                    enriched_df.at[index, 'confidence_score'] = result.get('confidence_score', 0.0)
                except Exception as e:
                    logger.error(f"Error processing product at index {index}: {str(e)}")
        
        logger.info(f"Product enrichment completed for {len(enriched_df)} products")
        
        return enriched_df
    
    def generate_category_suggestions(self, product_df: pd.DataFrame, 
                                     existing_taxonomy: Optional[List[Dict]] = None) -> List[Dict]:
        """
        Generate category taxonomy suggestions based on product data.
        
        Args:
            product_df: DataFrame containing product information
            existing_taxonomy: Optional existing taxonomy to be enhanced
            
        Returns:
            List of category dictionaries forming a taxonomy
        """

        use_local_taxonomy = True  # Set to True to use local taxonomy file
        if use_local_taxonomy:
            # Load local taxonomy from file
            with open('taxonomy.json', 'r') as f:
                existing_taxonomy = json.load(f)
            logger.info("Loaded local taxonomy from taxonomy.json")
            return existing_taxonomy


        if not self.client:
            logger.error("OpenAI client not initialized. Provide a valid API key.")
            raise ValueError("OpenAI client not initialized")
        
        logger.info(f"Generating category suggestions based on {len(product_df)} products")
        
        # Extract sample of product descriptions (up to 50)
        sample_size = min(50, len(product_df))
        sample_products = product_df.sample(sample_size)
        
        # Prepare descriptions text
        sample_descriptions = "\n".join([
            f"- {desc}" for desc in sample_products['Item Description'].tolist()
        ])
        
        # Construct the prompt
        existing_taxonomy_text = ""
        if existing_taxonomy:
            existing_taxonomy_text = json.dumps(existing_taxonomy, indent=2)
            
        prompt = f"""
        As a procurement categorization expert, analyze the following sample of product descriptions and generate a hierarchical taxonomy for procurement spend categorization.

        SAMPLE PRODUCTS:
        {sample_descriptions}

        {f'EXISTING TAXONOMY (to be enhanced/extended): {existing_taxonomy_text}' if existing_taxonomy else ''}

        Based on these product descriptions, please create a 3-level hierarchical taxonomy optimized for procurement spend analysis. The taxonomy should follow this structure:
        - Level 1: Broad categories (e.g., "Equipment", "Supplies")
        - Level 2: Sub-categories within Level 1
        - Level 3: Detailed product groups within Level 2

        Return ONLY a JSON array where each element has this structure:
        {{
            "id": "unique_id",
            "name": "Category Name",
            "level": 1,
            "parent_id": null,
            "description": "Description of this category"
        }}

        For level 2 and 3 categories, the parent_id should reference the id of its parent category.
        Return ONLY the JSON array with no preamble or explanation.
        """

        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a procurement taxonomy specialist that creates structured category hierarchies."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=4000
            )
            
            # Extract the JSON from the response
            result_text = response.choices[0].message.content.strip()
            
            # Clean up the response to ensure it's valid JSON
            result_text = result_text.replace("```json", "").replace("```", "").strip()
            
            # Parse the JSON
            taxonomy = json.loads(result_text)
            
            logger.info(f"Generated taxonomy with {len(taxonomy)} categories")
            
            return taxonomy
            
        except Exception as e:
            logger.error(f"Error generating category suggestions: {str(e)}")
            return []
    
    # def map_products_to_taxonomy(self, product_df: pd.DataFrame, 
    #                            taxonomy: List[Dict],
    #                            product_id_col: str = 'Item Number',
    #                            product_desc_col: str = 'Item Description') -> pd.DataFrame:
    #     """
    #     Map products to the provided taxonomy categories.
        
    #     Args:
    #         product_df: DataFrame containing product information
    #         taxonomy: List of taxonomy category dictionaries
    #         product_id_col: Column containing product IDs
    #         product_desc_col: Column containing product descriptions
            
    #     Returns:
    #         DataFrame with mapped categories
    #     """
    #     if not self.client:
    #         logger.error("OpenAI client not initialized. Provide a valid API key.")
    #         raise ValueError("OpenAI client not initialized")
        
    #     logger.info(f"Mapping {len(product_df)} products to taxonomy categories")
        
    #     # Create a copy of the input dataframe
    #     mapped_df = product_df.copy()
        
    #     # Add columns for mapped categories
    #     mapped_df['category_id_l1'] = None
    #     mapped_df['category_id_l2'] = None
    #     mapped_df['category_id_l3'] = None
    #     mapped_df['category_confidence'] = None
        
    #     # Create a formatted version of the taxonomy for the prompt
    #     taxonomy_by_level = {1: [], 2: [], 3: []}
    #     for cat in taxonomy:
    #         level = cat.get('level')
    #         if level in taxonomy_by_level:
    #             taxonomy_by_level[level].append(cat)
        
    #     # Format taxonomy for prompt
    #     taxonomy_text = ""
    #     for level in [1, 2, 3]:
    #         taxonomy_text += f"\nLEVEL {level} CATEGORIES:\n"
    #         for cat in taxonomy_by_level[level]:
    #             parent_info = ""
    #             if cat.get('parent_id'):
    #                 parent = next((c for c in taxonomy if c.get('id') == cat.get('parent_id')), None)
    #                 if parent:
    #                     parent_info = f" (under {parent.get('name')})"
    #             taxonomy_text += f"- {cat.get('id')}: {cat.get('name')}{parent_info} - {cat.get('description')}\n"
        
    #     # Define the mapping function for a single product
    #     def map_single_product(product_id, product_desc):
    #         prompt = f"""
    #         As a procurement categorization expert, map the following product to the most appropriate categories in our taxonomy.

    #         PRODUCT ID: {product_id}
    #         PRODUCT DESCRIPTION: {product_desc}

    #         TAXONOMY:
    #         {taxonomy_text}

    #         Please return ONLY a JSON object with the following fields:
    #         - category_id_l1: The ID of the most appropriate Level 1 category
    #         - category_id_l2: The ID of the most appropriate Level 2 category (must be a child of the Level 1 category)
    #         - category_id_l3: The ID of the most appropriate Level 3 category (must be a child of the Level 2 category)
    #         - confidence: Your confidence in this mapping (0.0-1.0)

    #         Return ONLY the JSON object with no preamble or explanation.
    #         """
            
    #         try:
    #             response = self.client.chat.completions.create(
    #                 model=self.model,
    #                 messages=[
    #                     {"role": "system", "content": "You are a procurement categorization assistant that maps products to taxonomy categories."},
    #                     {"role": "user", "content": prompt}
    #                 ],
    #                 temperature=0.2
    #             )
                
    #             # Extract the JSON from the response
    #             result_text = response.choices[0].message.content.strip()
                
    #             # Clean up the response to ensure it's valid JSON
    #             result_text = result_text.replace("```json", "").replace("```", "").strip()
                
    #             # Parse the JSON
    #             result = json.loads(result_text)
                
    #             # Add a small delay to avoid rate limiting
    #             time.sleep(self.request_delay)
                
    #             return result
                
    #         except Exception as e:
    #             logger.warning(f"Error mapping product '{product_id}': {str(e)}")
    #             return {
    #                 "category_id_l1": None,
    #                 "category_id_l2": None,
    #                 "category_id_l3": None,
    #                 "confidence": 0.0
    #             }
        
    #     # Process products in parallel with a thread pool
    #     with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
    #         # Submit all products to the thread pool
    #         future_to_index = {
    #             executor.submit(map_single_product, row[product_id_col], row[product_desc_col]): i 
    #             for i, row in mapped_df.iterrows()
    #         }
            
    #         # Process results as they complete
    #         for future in tqdm(concurrent.futures.as_completed(future_to_index), 
    #                           total=len(mapped_df), 
    #                           desc="Mapping products to taxonomy"):
    #             index = future_to_index[future]
    #             try:
    #                 result = future.result()
    #                 mapped_df.at[index, 'category_id_l1'] = result.get('category_id_l1')
    #                 mapped_df.at[index, 'category_id_l2'] = result.get('category_id_l2')
    #                 mapped_df.at[index, 'category_id_l3'] = result.get('category_id_l3')
    #                 mapped_df.at[index, 'category_confidence'] = result.get('confidence', 0.0)
    #             except Exception as e:
    #                 logger.error(f"Error processing mapping at index {index}: {str(e)}")
        
    #     logger.info(f"Product mapping completed for {len(mapped_df)} products")
        
    #     return mapped_df

    def map_products_to_taxonomy(self, product_df: pd.DataFrame, 
                            taxonomy: List[Dict],
                            product_id_col: str = 'Item Number',
                            product_desc_col: str = 'Item Description') -> pd.DataFrame:
        """
        Map products to the provided taxonomy categories using a hierarchical approach.
        
        Args:
            product_df: DataFrame containing product information
            taxonomy: List of taxonomy category dictionaries
            product_id_col: Column containing product IDs
            product_desc_col: Column containing product descriptions
            
        Returns:
            DataFrame with mapped categories
        """
        if not self.client:
            logger.error("OpenAI client not initialized. Provide a valid API key.")
            raise ValueError("OpenAI client not initialized")
        
        logger.info(f"Mapping {len(product_df)} products to taxonomy categories")
        
        # Create a copy of the input dataframe
        mapped_df = product_df.copy()
        
        # Add columns for mapped categories and paths
        mapped_df['category_path'] = None
        mapped_df['final_category_id'] = None
        mapped_df['final_category_level'] = None
        mapped_df['category_confidence'] = None
        
        # Build a taxonomy tree from the list for easier navigation
        taxonomy_tree = self._build_taxonomy_tree(taxonomy)
        
        # Define the mapping function for a single product
        def map_single_product(product_id, product_desc):
            try:
                # Start the classification process
                classification_result = self._classify_product_hierarchically(
                    product_desc, 
                    taxonomy_tree
                )
                
                # Extract the classification path and final category
                category_path = classification_result.get('classification_path', [])
                final_classification = classification_result.get('final_classification')
                
                # Format the result
                result = {
                    "category_path": category_path,
                    "final_category_id": final_classification.get('id') if final_classification else None,
                    "final_category_level": final_classification.get('level') if final_classification else None,
                    "confidence": classification_result.get('confidence', 0.7)  # Default confidence
                }
                
                return result
                    
            except Exception as e:
                logger.warning(f"Error mapping product '{product_id}': {str(e)}")
                return {
                    "category_path": [],
                    "final_category_id": None,
                    "final_category_level": None,
                    "confidence": 0.0
                }
        
        # Process products in parallel with a thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all products to the thread pool
            future_to_index = {
                executor.submit(map_single_product, row[product_id_col], row[product_desc_col]): i 
                for i, row in mapped_df.iterrows()
            }
            
            # Process results as they complete
            for future in tqdm(concurrent.futures.as_completed(future_to_index), 
                            total=len(mapped_df), 
                            desc="Mapping products to taxonomy"):
                index = future_to_index[future]
                try:
                    result = future.result()
                    mapped_df.at[index, 'category_path'] = json.dumps(result.get('category_path', []))
                    mapped_df.at[index, 'final_category_id'] = result.get('final_category_id')
                    mapped_df.at[index, 'final_category_level'] = result.get('final_category_level')
                    mapped_df.at[index, 'category_confidence'] = result.get('confidence', 0.0)
                except Exception as e:
                    logger.error(f"Error processing mapping at index {index}: {str(e)}")
        
        logger.info(f"Product mapping completed for {len(mapped_df)} products")
        
        return mapped_df
    
    def map_products_to_taxonomy2(self, product_df: pd.DataFrame, 
                            taxonomy: List[Dict],
                            product_id_col: str = 'Item Number',
                            product_desc_col: str = 'Item Description') -> pd.DataFrame:
        """
        Map products to the provided taxonomy categories using a hierarchical approach,
        adding the category names for each level directly to the dataframe.
        
        Args:
            product_df: DataFrame containing product information
            taxonomy: List of taxonomy category dictionaries
            product_id_col: Column containing product IDs
            product_desc_col: Column containing product descriptions
            
        Returns:
            DataFrame with mapped categories and category names
        """
        if not self.client:
            logger.error("OpenAI client not initialized. Provide a valid API key.")
            raise ValueError("OpenAI client not initialized")
        
        logger.info(f"Mapping {len(product_df)} products to taxonomy categories with category names")
        
        # Create a copy of the input dataframe
        mapped_df = product_df.copy()
        
        # Add columns for mapped category names
        mapped_df['category_l1_name'] = None
        mapped_df['category_l2_name'] = None
        mapped_df['category_l3_name'] = None
        mapped_df['category_l4_name'] = None
        mapped_df['category_l5_name'] = None
        mapped_df['category_confidence'] = None
        
        # Build a taxonomy tree from the list for easier navigation
        taxonomy_tree = self._build_taxonomy_tree(taxonomy)
        
        # Create a lookup dictionary for faster node access
        node_id_to_name = {node['id']: node['name'] for node in taxonomy}
        
        # Define the mapping function for a single product
        def map_single_product(product_id, product_desc):
            try:
                # Start the classification process
                classification_result = self._classify_product_hierarchically(
                    product_desc, 
                    taxonomy_tree
                )
                
                # Extract the classification path
                category_path = classification_result.get('classification_path', [])
                
                # Format the result with category names for each level
                result = {
                    "category_confidence": classification_result.get('confidence', 0.0)
                }
                
                # Extract category names for each level (up to 5 levels)
                for i, category in enumerate(category_path[:5], 1):
                    result[f"category_l{i}_name"] = category.get('name')
                
                return result
                    
            except Exception as e:
                logger.warning(f"Error mapping product '{product_id}': {str(e)}")
                return {
                    "category_l1_name": None,
                    "category_l2_name": None,
                    "category_l3_name": None,
                    "category_l4_name": None,
                    "category_l5_name": None,
                    "category_confidence": 0.0
                }
        
        # Process products in parallel with a thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all products to the thread pool
            future_to_index = {
                executor.submit(map_single_product, row[product_id_col], row[product_desc_col]): i 
                for i, row in mapped_df.iterrows()
            }
            
            # Process results as they complete
            for future in tqdm(concurrent.futures.as_completed(future_to_index), 
                            total=len(mapped_df), 
                            desc="Mapping products to taxonomy categories"):
                index = future_to_index[future]
                try:
                    result = future.result()
                    mapped_df.at[index, 'category_l1_name'] = result.get('category_l1_name')
                    mapped_df.at[index, 'category_l2_name'] = result.get('category_l2_name')
                    mapped_df.at[index, 'category_l3_name'] = result.get('category_l3_name')
                    mapped_df.at[index, 'category_l4_name'] = result.get('category_l4_name')
                    mapped_df.at[index, 'category_l5_name'] = result.get('category_l5_name')
                    mapped_df.at[index, 'category_confidence'] = result.get('category_confidence', 0.0)
                except Exception as e:
                    logger.error(f"Error processing mapping at index {index}: {str(e)}")
        
        logger.info(f"Product mapping with category names completed for {len(mapped_df)} products")
        
        return mapped_df

    def _build_taxonomy_tree(self, taxonomy: List[Dict]) -> Dict:
        """
        Build a taxonomy tree structure for easier navigation.
        
        Args:
            taxonomy: List of taxonomy node dictionaries
            
        Returns:
            Dictionary with taxonomy tree structure
        """
        # Initialize structure
        tree = {
            'nodes_by_id': {},
            'nodes_by_level': {},
            'children_by_parent': {}
        }
        
        # Process all nodes
        for node in taxonomy:
            node_id = node.get('id')
            level = node.get('level')
            parent_id = node.get('parent_id')
            
            # Store node by ID
            tree['nodes_by_id'][node_id] = node
            
            # Store node by level
            if level not in tree['nodes_by_level']:
                tree['nodes_by_level'][level] = []
            tree['nodes_by_level'][level].append(node)
            
            # Store parent-child relationships
            if parent_id:
                if parent_id not in tree['children_by_parent']:
                    tree['children_by_parent'][parent_id] = []
                tree['children_by_parent'][parent_id].append(node)
        
        logger.info(f"Built taxonomy tree with {len(tree['nodes_by_id'])} nodes across {len(tree['nodes_by_level'])} levels")
        return tree

    def _classify_product_hierarchically(self, product_desc: str, taxonomy_tree: Dict) -> Dict:
        """
        Classify a product by navigating through the taxonomy tree level by level.
        
        Args:
            product_desc: Product description
            taxonomy_tree: Taxonomy tree structure
            
        Returns:
            Dictionary with classification result
        """
        # Start with level 1 nodes
        min_level = min(taxonomy_tree['nodes_by_level'].keys())
        current_nodes = taxonomy_tree['nodes_by_level'].get(min_level, [])
        classification_path = []
        
        # Navigate down the tree until we reach a leaf node
        while current_nodes:
            # Select the best category at the current level
            selected_node = self._select_best_category(product_desc, current_nodes)
            
            if not selected_node:
                break
            
            # Add to classification path
            classification_path.append({
                'id': selected_node.get('id'),
                'name': selected_node.get('name'),
                'level': selected_node.get('level'),
                'description': selected_node.get('description'),
                'code': selected_node.get('code', None)
            })
            
            # Get children of the selected node
            node_id = selected_node.get('id')
            current_nodes = taxonomy_tree['children_by_parent'].get(node_id, [])
        
        # Return the classification result
        result = {
            'classification_path': classification_path,
            'final_classification': classification_path[-1] if classification_path else None,
            'confidence': 0.9 if classification_path else 0.0  # Simplified confidence
        }
        
        return result

    def _select_best_category(self, product_desc: str, categories: List[Dict]) -> Dict:
        """
        Use GPT-4 to select the best category for a product from given options.
        
        Args:
            product_desc: Product description
            categories: List of category dictionaries to choose from
            
        Returns:
            Selected category dictionary or None if no selection could be made
        """
        if not categories:
            return None
        
        # Format options for GPT-4
        options_text = "\n".join([
            f"{i+1}. {cat.get('name')}: {cat.get('description', 'No description')} (ID: {cat.get('id')})" 
            for i, cat in enumerate(categories)
        ])
        
        # Create the prompt
        prompt = f"""
        As a procurement categorization expert, select the most appropriate category for this product:
        
        PRODUCT DESCRIPTION: {product_desc}
        
        Available categories:
        {options_text}
        
        Select the most appropriate category by responding with ONLY the ID number of the category.
        Do not include any explanation or additional text in your response, just the ID number.
        """
        
        try:
            # Call the OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a procurement categorization assistant that maps products to taxonomy categories."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=20
            )
            
            # Extract the category ID from the response
            gpt_response = response.choices[0].message.content.strip()
            
            # Try to find the selected option
            for category in categories:
                if category.get('id') in gpt_response:
                    return category
            
            # If we couldn't match the response directly to an id,
            # log a warning and return the first option as a fallback
            logger.warning(f"Couldn't match GPT-4 response '{gpt_response}' to any category id. Using first option as fallback.")
            return categories[0] if categories else None
            
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            # Return the first option as a fallback in case of error
            return categories[0] if categories else None
        
        # Add a small delay to avoid rate limiting
        time.sleep(self.request_delay)
    
    def save_enriched_data(self, supplier_df: pd.DataFrame, product_df: pd.DataFrame, 
                          taxonomy: List[Dict], output_dir: str) -> None:
        """
        Save enriched data to disk.
        
        Args:
            supplier_df: Enriched supplier DataFrame
            product_df: Enriched product DataFrame
            taxonomy: Generated taxonomy
            output_dir: Directory to save the data
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save supplier data
        supplier_df.to_csv(os.path.join(output_dir, 'enriched_suppliers.csv'), index=False)
        logger.info(f"Saved enriched supplier data to {os.path.join(output_dir, 'enriched_suppliers.csv')}")
        
        # Save product data
        product_df.to_csv(os.path.join(output_dir, 'enriched_products.csv'), index=False)
        logger.info(f"Saved enriched product data to {os.path.join(output_dir, 'enriched_products.csv')}")
        
        # Save taxonomy
        with open(os.path.join(output_dir, 'taxonomy.json'), 'w') as f:
            json.dump(taxonomy, f, indent=2)
        logger.info(f"Saved taxonomy to {os.path.join(output_dir, 'taxonomy.json')}")
    
    def run_enrichment_pipeline(self, supplier_df: pd.DataFrame, product_df: pd.DataFrame,
                              output_dir: str) -> Dict:
        """
        Run the complete enrichment pipeline.
        
        Args:
            supplier_df: DataFrame containing supplier information
            product_df: DataFrame containing product information
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary containing all enriched data
        """
        try:
            # this has already been done.


            # check if we need to enrich suppliers and products
            enrich_suppliers = False
            enrich_products = False

            enriched_products = None
            enriched_suppliers = None

            if os.path.exists(os.path.join(output_dir, 'enriched_suppliers.csv')):
                logger.info(f"Enriched supplier data already exists at {os.path.join(output_dir, 'enriched_suppliers.csv')}")
                enrich_suppliers = False
                enriched_suppliers = pd.read_csv(os.path.join(output_dir, 'enriched_suppliers.csv'))
            else:
                logger.info(f"Enriched supplier data does not exist at {os.path.join(output_dir, 'enriched_suppliers.csv')}")
                enrich_suppliers = True

            if os.path.exists(os.path.join(output_dir, 'enriched_products.csv')):
                logger.info(f"Enriched product data already exists at {os.path.join(output_dir, 'enriched_products.csv')}")
                enrich_products = False
                enriched_products = pd.read_csv(os.path.join(output_dir, 'enriched_products.csv'))
            else:
                logger.info(f"Enriched product data does not exist at {os.path.join(output_dir, 'enriched_products.csv')}")
                enrich_products = True



            if enrich_suppliers:
                # Enrich suppliers
                enriched_suppliers = self.enrich_suppliers(supplier_df)
                # save enriched suppliers to disk
                enriched_suppliers.to_csv(os.path.join(output_dir, 'enriched_suppliers.csv'), index=False)
                logger.info(f"Saved enriched supplier data to {os.path.join(output_dir, 'enriched_suppliers.csv')}")
            

            if enrich_products:
                # Enrich products
                enriched_products = self.enrich_products(product_df)
                # save enriched products to disk
                enriched_products.to_csv(os.path.join(output_dir, 'enriched_products.csv'), index=False)
                logger.info(f"Saved enriched product data to {os.path.join(output_dir, 'enriched_products.csv')}")
            
            if True:
                # Generate taxonomy
                taxonomy = self.generate_category_suggestions(product_df)
                # save taxonomy to disk
                with open(os.path.join(output_dir, 'taxonomy.json'), 'w') as f:
                    json.dump(taxonomy, f, indent=2)
                logger.info(f"Saved taxonomy to {os.path.join(output_dir, 'taxonomy.json')}")

            
            
            # Map products to taxonomy
            mapped_products = self.map_products_to_taxonomy2(enriched_products, taxonomy)
            
            # Save all data
            self.save_enriched_data(enriched_suppliers, mapped_products, taxonomy, output_dir)
            
            logger.info("Enrichment pipeline completed successfully")
            
            return {
                'enriched_suppliers': enriched_suppliers,
                'enriched_products': enriched_products,
                'mapped_products': mapped_products,
                'taxonomy': taxonomy
            }
            
        except Exception as e:
            logger.error(f"Error in enrichment pipeline: {str(e)}")
            raise

# Example usage
if __name__ == '__main__':


    # this is the limit for suppliers to enrich (for testing purposes)
    max_sumpplier = 5
    # this is the limit for products to enrich (for testing purposes)
    max_product = 2000


    # Load processed data (assuming it was generated by ProcurementDataPipeline)
    supplier_df = pd.read_csv('processed_data/supplier_reference.csv')
    product_df = pd.read_csv('processed_data/product_reference.csv')


    if max_sumpplier > 0:
        # only keep the first few rows for testing
        supplier_df = supplier_df.head(max_sumpplier)

    if max_product > 0:
        # start with from row 2000 to row 4000
        product_df = product_df.iloc[2000:2000+max_product]


    
    # Initialize enrichment module
    enrichment = ProcurementDataEnrichment()

    # Ensure output directory exists
    output_dir = 'output/enriched4'
    os.makedirs(output_dir, exist_ok=True)
    # Run enrichment pipeline
    results = enrichment.run_enrichment_pipeline(
        supplier_df=supplier_df,
        product_df=product_df,
        output_dir='output/enriched4'
    )
    

"""
PAVUS AI Procurement Dashboard - Data Generator

This script transforms the enriched products and suppliers datasets into
structured datasets ready for dashboard visualization.

Input:
- enriched_products.csv: Output from the procurement categorization system
- enriched_suppliers.csv: Output from the procurement data enrichment system
- transaction_data.csv: Raw procurement transaction data (if available)

Output:
- categorized_spend.csv: Categorized transaction data
- category_analytics.csv: Analytics per category
- supplier_analytics.csv: Analytics per supplier
- model_performance.csv: Categorization model performance metrics
- cost_saving_opportunities.csv: Identified savings opportunities
- data_quality.csv: Data quality metrics
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import random
from collections import defaultdict

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

class DashboardDataGenerator:
    """
    Generates structured datasets for the procurement dashboard from enriched data.
    """
    
    def __init__(self, 
                 enriched_products_path, 
                 enriched_suppliers_path, 
                 transaction_data_path=None,
                 output_dir='dashboard_data'):
        """
        Initialize the data generator.
        
        Args:
            enriched_products_path: Path to enriched products CSV
            enriched_suppliers_path: Path to enriched suppliers CSV
            transaction_data_path: Optional path to transaction data
            output_dir: Directory to save generated datasets
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load enriched products data
        print(f"Loading enriched products from {enriched_products_path}")
        self.products_df = pd.read_csv(enriched_products_path)
        
        # Load enriched suppliers data
        print(f"Loading enriched suppliers from {enriched_suppliers_path}")
        self.suppliers_df = pd.read_csv(enriched_suppliers_path)
        
        # Load transaction data if available, or create simulated data
        if transaction_data_path and os.path.exists(transaction_data_path):
            print(f"Loading transaction data from {transaction_data_path}")
            self.transactions_df = pd.read_csv(transaction_data_path)
        else:
            print("Transaction data not provided. Generating simulated transaction data.")
            self.transactions_df = self._generate_simulated_transactions()
        
        # Parse JSON fields
        self._parse_json_fields()
        
        print(f"Loaded {len(self.products_df)} products, {len(self.suppliers_df)} suppliers, " 
              f"and {len(self.transactions_df)} transactions")
    
    def _parse_json_fields(self):
        """Parse JSON string fields into Python objects."""
        # Parse product JSON fields
        json_columns = ['category_path', 'attributes', 'specifications']
        
        for col in json_columns:
            if col in self.products_df.columns:
                self.products_df[f'{col}_parsed'] = self.products_df[col].apply(
                    lambda x: json.loads(x) if isinstance(x, str) and x.strip() else []
                )
    
    def _generate_simulated_transactions(self):
        """
        Generate simulated transaction data based on products and suppliers.
        
        Returns:
            DataFrame with simulated transactions
        """
        # Create a base dataframe with product and supplier combinations
        transactions = []
        
        # Get product IDs and supplier IDs
        product_ids = self.products_df['Item Number'].unique()
        supplier_ids = self.suppliers_df['Vendor ID'].unique()
        
        # Random dates in 2024-2025
        dates = pd.date_range(start='2024-01-01', end='2025-05-01')
        
        # Generate random transactions
        n_transactions = min(len(product_ids) * 5, 10000)  # 5 transactions per product on average, up to 10,000
        
        for _ in range(n_transactions):
            # Random product
            product_idx = np.random.randint(0, len(product_ids))
            product_id = product_ids[product_idx]
            
            # Get the corresponding supplier for this product
            # In a real scenario, products could come from multiple suppliers
            # Here we're simplifying by using a deterministic mapping
            supplier_idx = product_idx % len(supplier_ids)
            supplier_id = supplier_ids[supplier_idx]
            
            # Random date
            date_idx = np.random.randint(0, len(dates))
            transaction_date = dates[date_idx].strftime('%Y-%m-%d')
            
            # Random quantity and unit cost
            quantity = np.random.randint(1, 20)
            unit_cost = np.round(np.random.uniform(50, 500), 2)
            extended_cost = quantity * unit_cost
            
            # Create transaction
            transaction = {
                'transaction_id': f"TX{np.random.randint(100000, 999999)}",
                'date': transaction_date,
                'po_number': f"PO{np.random.randint(100000, 999999)}",
                'supplier_id': supplier_id,
                'item_number': product_id,
                'quantity': quantity,
                'unit_cost': unit_cost,
                'extended_cost': extended_cost,
                'currency': 'USD'
            }
            
            transactions.append(transaction)
        
        # Convert to DataFrame
        transactions_df = pd.DataFrame(transactions)
        
        # Add supplier names
        supplier_name_map = dict(zip(self.suppliers_df['Vendor ID'], self.suppliers_df['Vendor Name']))
        transactions_df['supplier_name'] = transactions_df['supplier_id'].map(supplier_name_map)
        
        # Add product descriptions
        product_desc_map = dict(zip(self.products_df['Item Number'], self.products_df['Item Description']))
        transactions_df['item_description'] = transactions_df['item_number'].map(product_desc_map)
        
        return transactions_df
    
    def generate_categorized_spend(self):
        """
        Generate the categorized spend dataset by combining transactions and categorization.
        
        Returns:
            DataFrame with categorized transactions
        """
        print("Generating categorized spend dataset...")
        
        # Start with transaction data
        categorized_df = self.transactions_df.copy()
        
        # Add categorization data from products
        product_category_map = {}
        product_attributes_map = {}
        
        for _, product in self.products_df.iterrows():
            product_id = product['Item Number']
            
            # Build category data
            if 'category_path_parsed' in self.products_df.columns and not pd.isna(product.get('category_path_parsed', [])):
                category_path = product['category_path_parsed']
                final_category_id = product.get('final_category_id', '')
                final_category_level = product.get('final_category_level', 0)
            else:
                # Fallback if no parsed path
                category_path = []
                final_category_id = ''
                final_category_level = 0
            
            # Build product attributes
            attributes = product.get('attributes_parsed', [])
            specifications = product.get('specifications_parsed', {})
            primary_function = product.get('primary_function', '')
            
            product_category_map[product_id] = {
                'category_path': category_path,
                'final_category_id': final_category_id,
                'final_category_level': final_category_level,
                'categorization_confidence': product.get('category_confidence', np.random.uniform(0.7, 0.98))
            }
            
            product_attributes_map[product_id] = {
                'primary_function': primary_function,
                'specifications': specifications,
                'attributes': attributes
            }
        
        # Add category data to transactions
        categorized_df['category_path'] = categorized_df['item_number'].apply(
            lambda x: json.dumps(product_category_map.get(x, {}).get('category_path', []))
        )
        
        categorized_df['final_category_id'] = categorized_df['item_number'].apply(
            lambda x: product_category_map.get(x, {}).get('final_category_id', '')
        )
        
        categorized_df['final_category_level'] = categorized_df['item_number'].apply(
            lambda x: product_category_map.get(x, {}).get('final_category_level', 0)
        )
        
        categorized_df['categorization_confidence'] = categorized_df['item_number'].apply(
            lambda x: product_category_map.get(x, {}).get('categorization_confidence', 0.7)
        )
        
        # Determine if review is needed based on confidence
        categorized_df['needs_review'] = categorized_df['categorization_confidence'] < 0.7
        
        # Add categorization method (90% ML, 10% manual)
        categorized_df['categorization_method'] = np.random.choice(
            ['ML', 'Manual'], 
            size=len(categorized_df),
            p=[0.9, 0.1]
        )
        
        # Add product attributes as JSON strings
        categorized_df['product_attributes'] = categorized_df['item_number'].apply(
            lambda x: json.dumps(product_attributes_map.get(x, {}))
        )
        
        # Add supplier insights
        supplier_insights_map = {}
        
        for _, supplier in self.suppliers_df.iterrows():
            supplier_id = supplier['Vendor ID']
            
            insights = {
                'industry': supplier.get('industry', ''),
                'market_segment': supplier.get('market_segment', ''),
                'company_size': supplier.get('likely_size', ''),
                'risk_level': supplier.get('supplier_risk_level', ''),
                'sustainability_rating': supplier.get('sustainability_rating', 0),
                'financial_health': supplier.get('financial_health', 0)
            }
            
            supplier_insights_map[supplier_id] = insights
        
        categorized_df['supplier_insights'] = categorized_df['supplier_id'].apply(
            lambda x: json.dumps(supplier_insights_map.get(x, {}))
        )
        
        # Save to CSV
        output_path = os.path.join(self.output_dir, 'categorized_spend.csv')
        categorized_df.to_csv(output_path, index=False)
        print(f"Saved categorized spend dataset with {len(categorized_df)} records to {output_path}")
        
        return categorized_df
    
    def generate_category_analytics(self):
        """
        Generate analytics by category.
        
        Returns:
            DataFrame with category analytics
        """
        print("Generating category analytics dataset...")
        
        # Start with categorized spend data
        if not hasattr(self, 'categorized_df'):
            self.categorized_df = self.generate_categorized_spend()
        
        # Initialize data structure for category analytics
        category_data = {}
        
        # Parse category paths
        self.categorized_df['category_path_parsed'] = self.categorized_df['category_path'].apply(
            lambda x: json.loads(x) if isinstance(x, str) and x.strip() else []
        )
        
        # Group by final category
        category_groups = self.categorized_df.groupby('final_category_id')
        
        for category_id, group in category_groups:
            if not category_id or category_id == '':
                continue
                
            # Get first row to extract category details
            first_row = group.iloc[0]
            category_path_parsed = first_row['category_path_parsed']
            
            # Skip if no valid category path
            if not category_path_parsed:
                continue
            
            # Get category details
            final_level = int(first_row['final_category_level'])
            category_name = ''
            parent_category_id = ''
            parent_category_name = ''
            category_path_text = ''
            category_code = ''
            
            for path_item in category_path_parsed:
                level = path_item.get('level', 0)
                
                if level == final_level:
                    category_name = path_item.get('name', '')
                    category_code = path_item.get('code', '')
                
                if level == final_level - 1:
                    parent_category_id = path_item.get('id', '')
                    parent_category_name = path_item.get('name', '')
                
                if category_path_text:
                    category_path_text += ' > '
                category_path_text += path_item.get('name', '')
            
            # Calculate spend metrics
            total_spend = group['extended_cost'].sum()
            
            # Simulate YTD and previous year data
            spend_year_to_date = total_spend * np.random.uniform(0.2, 0.4)
            spend_previous_year = total_spend * np.random.uniform(0.8, 1.2)
            spend_change_percent = ((total_spend - spend_previous_year) / spend_previous_year) * 100
            
            transaction_count = len(group)
            average_transaction_value = total_spend / transaction_count if transaction_count > 0 else 0
            
            # Calculate supplier metrics
            suppliers = group['supplier_id'].unique()
            supplier_count = len(suppliers)
            
            # Top suppliers
            supplier_spend = group.groupby('supplier_id').agg({
                'extended_cost': 'sum',
                'supplier_name': 'first'
            }).sort_values('extended_cost', ascending=False)
            
            top_suppliers = []
            for supplier_id, row in supplier_spend.head(3).iterrows():
                supplier_spend_amount = row['extended_cost']
                supplier_percent = (supplier_spend_amount / total_spend) * 100
                
                top_suppliers.append({
                    'id': supplier_id,
                    'name': row['supplier_name'],
                    'spend': supplier_spend_amount,
                    'percent': supplier_percent
                })
            
            supplier_concentration = sum(supplier['percent'] for supplier in top_suppliers)
            
            # Price analysis
            unit_prices = group['unit_cost']
            price_range = {
                'min': unit_prices.min(),
                'max': unit_prices.max(),
                'average': unit_prices.mean(),
                'median': unit_prices.median()
            }
            
            # Calculate price volatility (coefficient of variation)
            if unit_prices.mean() > 0:
                price_volatility = unit_prices.std() / unit_prices.mean()
            else:
                price_volatility = 0
            
            # Categorization stats
            confidence_values = group['categorization_confidence']
            categorization_confidence = {
                'average': confidence_values.mean(),
                'low_confidence_items': (confidence_values < 0.7).sum(),
                'manually_reviewed': (group['categorization_method'] == 'Manual').sum()
            }
            
            # Store category data
            category_data[category_id] = {
                'category_id': category_id,
                'category_name': category_name,
                'category_level': final_level,
                'parent_category_id': parent_category_id,
                'parent_category_name': parent_category_name,
                'category_path': category_path_text,
                'category_code': category_code,
                
                # Spend Metrics
                'total_spend': total_spend,
                'spend_year_to_date': spend_year_to_date,
                'spend_previous_year': spend_previous_year,
                'spend_change_percent': spend_change_percent,
                'transaction_count': transaction_count,
                'average_transaction_value': average_transaction_value,
                
                # Supplier Metrics
                'supplier_count': supplier_count,
                'top_suppliers': json.dumps(top_suppliers),
                'supplier_concentration': supplier_concentration,
                
                # Price Analysis
                'price_range': json.dumps(price_range),
                'price_volatility': price_volatility,
                
                # Categorization Stats
                'categorization_confidence': json.dumps(categorization_confidence)
            }
        
        # Convert to DataFrame
        category_analytics_df = pd.DataFrame.from_dict(category_data, orient='index')
        
        # Save to CSV
        output_path = os.path.join(self.output_dir, 'category_analytics.csv')
        category_analytics_df.to_csv(output_path, index=False)
        print(f"Saved category analytics dataset with {len(category_analytics_df)} categories to {output_path}")
        
        return category_analytics_df
    
    def generate_supplier_analytics(self):
        """
        Generate analytics by supplier.
        
        Returns:
            DataFrame with supplier analytics
        """
        print("Generating supplier analytics dataset...")
        
        # Start with categorized spend data
        if not hasattr(self, 'categorized_df'):
            self.categorized_df = self.generate_categorized_spend()
        
        # Initialize data structure for supplier analytics
        supplier_data = {}
        
        # Group by supplier
        supplier_groups = self.categorized_df.groupby('supplier_id')
        
        for supplier_id, group in supplier_groups:
            # Get supplier details from enriched data
            supplier_info = self.suppliers_df[self.suppliers_df['Vendor ID'] == supplier_id]
            
            if supplier_info.empty:
                continue
                
            supplier_info = supplier_info.iloc[0]
            
            # Basic Supplier Info
            standard_name = supplier_info.get('standard_name', '')
            supplier_name = supplier_info.get('Vendor Name', '')
            industry = supplier_info.get('industry', '')
            market_segment = supplier_info.get('market_segment', '')
            company_size = supplier_info.get('likely_size', '')
            location = "Unknown"  # This would come from additional data
            
            # Randomize relationship status
            relationship_status = np.random.choice(['Preferred', 'Approved', 'One-time', 'Strategic'], p=[0.3, 0.4, 0.2, 0.1])
            
            # Spend Metrics
            total_spend = group['extended_cost'].sum()
            
            # Simulate YTD and previous year data
            spend_year_to_date = total_spend * np.random.uniform(0.2, 0.4)
            spend_previous_year = total_spend * np.random.uniform(0.8, 1.2)
            spend_change_percent = ((total_spend - spend_previous_year) / spend_previous_year) * 100
            
            transaction_count = len(group)
            average_transaction_value = total_spend / transaction_count if transaction_count > 0 else 0
            
            # Category Distribution
            category_spend = group.groupby(['final_category_id', 'category_path_parsed']).agg({
                'extended_cost': 'sum'
            }).reset_index().sort_values('extended_cost', ascending=False)
            
            category_distribution = []
            for _, row in category_spend.head(5).iterrows():
                category_id = row['final_category_id']
                category_path = row['category_path_parsed']
                
                if isinstance(category_path, str):
                    category_path = json.loads(category_path)
                
                if not category_path:
                    continue
                    
                # Get the name of the last item in the path
                category_name = category_path[-1].get('name', '') if category_path else ''
                category_spend_amount = row['extended_cost']
                category_percent = (category_spend_amount / total_spend) * 100
                
                category_distribution.append({
                    'category_id': category_id,
                    'name': category_name,
                    'spend': category_spend_amount,
                    'percent': category_percent
                })
            
            # Category concentration
            category_concentration = sum(cat['percent'] for cat in category_distribution)
            
            # Risk and Performance - use enriched data where available
            risk_level = supplier_info.get('supplier_risk_level', np.random.choice(['Low', 'Medium', 'High'], p=[0.6, 0.3, 0.1]))
            
            # Simulate risk factors
            risk_factors_options = [
                "Single source for key items", 
                "Regional supplier",
                "Financial stability concerns",
                "Delivery performance issues",
                "Quality variations",
                "Limited capacity",
                "Contract expiration approaching",
                "Compliance issues"
            ]
            
            n_risk_factors = np.random.randint(0, 3)
            risk_factors = random.sample(risk_factors_options, n_risk_factors)
            
            # Get ratings from enriched data
            sustainability_rating = supplier_info.get('sustainability_rating', np.random.randint(1, 10))
            digital_presence_score = supplier_info.get('digital_presence_score', np.random.randint(1, 10))
            financial_health = supplier_info.get('financial_health', np.random.randint(1, 10))
            innovation_score = supplier_info.get('innovation_score', np.random.randint(1, 10))
            
            # Contract Information (simulated)
            contract_status = np.random.choice(['Active', 'Expiring', 'Expired', 'In Negotiation'], p=[0.7, 0.1, 0.1, 0.1])
            
            # Generate random expiration date in the next 2 years
            expiration_days = np.random.randint(30, 730)
            contract_expiration = (datetime.now() + pd.Timedelta(days=expiration_days)).strftime('%Y-%m-%d')
            
            # Payment terms
            payment_terms = np.random.choice(['Net 30', 'Net 45', 'Net 60'])
            
            # Available discounts
            discount_options = ["2% 10 Net 30", "Volume discount > $50,000", "Early payment discount", "Annual rebate"]
            n_discounts = np.random.randint(0, 3)
            discounts_available = random.sample(discount_options, n_discounts)
            
            # Store supplier data
            supplier_data[supplier_id] = {
                'supplier_id': supplier_id,
                'supplier_name': supplier_name,
                'standard_name': standard_name,
                
                # Basic Supplier Info
                'industry': industry,
                'market_segment': market_segment,
                'company_size': company_size,
                'location': location,
                'relationship_status': relationship_status,
                
                # Spend Metrics
                'total_spend': total_spend,
                'spend_year_to_date': spend_year_to_date,
                'spend_previous_year': spend_previous_year,
                'spend_change_percent': spend_change_percent,
                'transaction_count': transaction_count,
                'average_transaction_value': average_transaction_value,
                
                # Category Distribution
                'category_distribution': json.dumps(category_distribution),
                'category_concentration': category_concentration,
                
                # Risk and Performance
                'risk_level': risk_level,
                'risk_factors': json.dumps(risk_factors),
                'sustainability_rating': sustainability_rating,
                'digital_presence_score': digital_presence_score,
                'financial_health': financial_health,
                'innovation_score': innovation_score,
                
                # Contract Information
                'contract_status': contract_status,
                'contract_expiration': contract_expiration,
                'payment_terms': payment_terms,
                'discounts_available': json.dumps(discounts_available)
            }
        
        # Convert to DataFrame
        supplier_analytics_df = pd.DataFrame.from_dict(supplier_data, orient='index')
        
        # Save to CSV
        output_path = os.path.join(self.output_dir, 'supplier_analytics.csv')
        supplier_analytics_df.to_csv(output_path, index=False)
        print(f"Saved supplier analytics dataset with {len(supplier_analytics_df)} suppliers to {output_path}")
        
        return supplier_analytics_df
    
    def generate_model_performance(self):
        """
        Generate model performance metrics.
        
        Returns:
            DataFrame with model performance data
        """
        print("Generating model performance dataset...")
        
        # This would ideally come from actual model evaluation
        # Here we'll simulate some reasonable performance metrics
        
        model_performance = {
            'model_version': '1.2.5',
            'training_date': datetime.now().strftime('%Y-%m-%d'),
            'data_points_trained': len(self.products_df),
            
            # Overall Performance
            'overall_accuracy': np.random.uniform(0.85, 0.92),
            'overall_f1_score': np.random.uniform(0.83, 0.90),
            'low_confidence_predictions': np.random.uniform(0.05, 0.12),
            
            # Performance by Level (as JSON)
            'level_performance': json.dumps([
                {'level': 1, 'accuracy': np.random.uniform(0.94, 0.98), 'f1_score': np.random.uniform(0.93, 0.97), 'number_of_classes': 8},
                {'level': 2, 'accuracy': np.random.uniform(0.91, 0.95), 'f1_score': np.random.uniform(0.90, 0.94), 'number_of_classes': 28},
                {'level': 3, 'accuracy': np.random.uniform(0.88, 0.92), 'f1_score': np.random.uniform(0.86, 0.91), 'number_of_classes': 104},
                {'level': 4, 'accuracy': np.random.uniform(0.84, 0.89), 'f1_score': np.random.uniform(0.82, 0.87), 'number_of_classes': 312},
                {'level': 5, 'accuracy': np.random.uniform(0.79, 0.84), 'f1_score': np.random.uniform(0.76, 0.82), 'number_of_classes': 785}
            ]),
            
            # Feature Importance (as JSON)
            'feature_importance': json.dumps([
                {'feature': 'product_description_tfidf', 'importance': np.random.uniform(0.40, 0.50)},
                {'feature': 'supplier_industry', 'importance': np.random.uniform(0.15, 0.20)},
                {'feature': 'primary_function', 'importance': np.random.uniform(0.12, 0.18)},
                {'feature': 'attributes', 'importance': np.random.uniform(0.08, 0.15)},
                {'feature': 'specifications', 'importance': np.random.uniform(0.05, 0.12)}
            ]),
            
            # Common Misclassifications (as JSON)
            'common_misclassifications': json.dumps([
                {'from_category': '102993', 'to_category': '102994', 'count': np.random.randint(30, 60), 'confusion_rate': np.random.uniform(0.10, 0.20)},
                {'from_category': '103245', 'to_category': '103246', 'count': np.random.randint(20, 50), 'confusion_rate': np.random.uniform(0.08, 0.15)},
                {'from_category': '104578', 'to_category': '104580', 'count': np.random.randint(15, 40), 'confusion_rate': np.random.uniform(0.05, 0.12)}
            ])
        }
        
        # Convert to DataFrame
        model_performance_df = pd.DataFrame([model_performance])
        
        # Save to CSV
        output_path = os.path.join(self.output_dir, 'model_performance.csv')
        model_performance_df.to_csv(output_path, index=False)
        print(f"Saved model performance dataset to {output_path}")
        
        return model_performance_df
    
    def generate_cost_saving_opportunities(self):
        """
        Generate cost saving opportunities based on categorized spend analysis.
        
        Returns:
            DataFrame with cost saving opportunities
        """
        print("Generating cost saving opportunities dataset...")
        
        # This would typically come from an analysis algorithm
        # Here we'll generate sample opportunities
        
        # Start with categorized spend data
        if not hasattr(self, 'categorized_df'):
            self.categorized_df = self.generate_categorized_spend()
        
        # If we don't have category analytics yet, generate them
        if not hasattr(self, 'category_analytics_df'):
            self.category_analytics_df = self.generate_category_analytics()
        
        # Generate opportunities based on different types
        opportunities = []
        
        # 1. Price Standardization Opportunities
        # Look for categories with high price variance
        for _, category in self.category_analytics_df.iterrows():
            category_id = category['category_id']
            category_name = category['category_name']
            price_range = json.loads(category['price_range']) if isinstance(category['price_range'], str) else {}
            price_volatility = category.get('price_volatility', 0)
            
            # Only consider categories with significant volatility
            if price_volatility > 0.15 and category['supplier_count'] > 1:
                # Get all transactions for this category
                category_transactions = self.categorized_df[self.categorized_df['final_category_id'] == category_id]
                
                # Group by supplier
                supplier_prices = category_transactions.groupby('supplier_id').agg({
                    'unit_cost': 'mean',
                    'supplier_name': 'first',
                    'extended_cost': 'sum',
                    'quantity': 'sum'
                })
                
                # Only consider if we have multiple suppliers
                if len(supplier_prices) > 1:
                    # Find lowest-priced supplier
                    lowest_price_supplier = supplier_prices.loc[supplier_prices['unit_cost'].idxmin()]
                    
                    # Calculate potential savings
                    avg_price = supplier_prices['unit_cost'].mean()
                    target_price = lowest_price_supplier['unit_cost']
                    total_volume = supplier_prices['quantity'].sum()
                    current_spend = supplier_prices['extended_cost'].sum()
                    potential_target_spend = target_price * total_volume
                    estimated_savings = current_spend - potential_target_spend
                    
                                            # Only consider significant savings
                    if estimated_savings > 5000:
                        suppliers_detail = []
                        for supplier_id, row in supplier_prices.iterrows():
                            suppliers_detail.append({
                                'id': supplier_id,
                                'name': row['supplier_name'],
                                'avg_price': row['unit_cost']
                            })
                        
                        opportunity = {
                            'opportunity_id': f"OPP{np.random.randint(100000, 999999)}",
                            'opportunity_type': 'Price Standardization',
                            'description': f"Standardize pricing for {category_name} across suppliers",
                            'affected_category': json.dumps({'id': category_id, 'name': category_name}),
                            'estimated_savings': estimated_savings,
                            'confidence_level': 'High' if price_volatility > 0.25 else 'Medium',
                            
                            # Opportunity Details
                            'current_state': json.dumps({
                                'suppliers': suppliers_detail,
                                'volume': total_volume,
                                'current_spend': current_spend
                            }),
                            
                            'recommended_action': json.dumps({
                                'target_price': target_price,
                                'preferred_supplier': lowest_price_supplier['supplier_name'],
                                'implementation_difficulty': 'Medium',
                                'next_steps': [
                                    "Review contracts with current suppliers",
                                    "Negotiate price match with preferred suppliers",
                                    "Consolidate volume with best-price supplier"
                                ]
                            }),
                            
                            # Supporting Evidence
                            'evidence': json.dumps({
                                'price_trend': 'Stable',
                                'market_conditions': 'Competitive',
                                'benchmark_data': f"Industry average price is ${target_price * 1.05:.2f}"
                            })
                        }
                        
                        opportunities.append(opportunity)
        
        # 2. Supplier Consolidation Opportunities
        # Look for categories with multiple suppliers
        for _, category in self.category_analytics_df.iterrows():
            category_id = category['category_id']
            category_name = category['category_name']
            supplier_count = category['supplier_count']
            
            # Only consider categories with multiple suppliers
            if supplier_count > 3:
                # Get top suppliers from the category data
                top_suppliers = json.loads(category['top_suppliers']) if isinstance(category['top_suppliers'], str) else []
                
                # Only consider if we have significant fragmentation
                if len(top_suppliers) >= 3:
                    # Calculate potential savings from consolidation
                    # Assume 5-10% savings potential from consolidation
                    savings_percent = np.random.uniform(0.05, 0.10)
                    total_spend = category['total_spend']
                    estimated_savings = total_spend * savings_percent
                    
                    # Only consider significant savings
                    if estimated_savings > 10000:
                        opportunity = {
                            'opportunity_id': f"OPP{np.random.randint(100000, 999999)}",
                            'opportunity_type': 'Supplier Consolidation',
                            'description': f"Consolidate {category_name} spending with fewer suppliers",
                            'affected_category': json.dumps({'id': category_id, 'name': category_name}),
                            'estimated_savings': estimated_savings,
                            'confidence_level': 'Medium',
                            
                            # Opportunity Details
                            'current_state': json.dumps({
                                'suppliers': top_suppliers,
                                'supplier_count': supplier_count,
                                'current_spend': total_spend
                            }),
                            
                            'recommended_action': json.dumps({
                                'target_suppliers': [s['id'] for s in top_suppliers[:2]],
                                'implementation_difficulty': 'Medium',
                                'next_steps': [
                                    "Analyze supplier performance data",
                                    "Identify top-performing suppliers",
                                    "Develop consolidation strategy",
                                    "Initiate new contract negotiations"
                                ]
                            }),
                            
                            # Supporting Evidence
                            'evidence': json.dumps({
                                'consolidation_potential': f"{savings_percent * 100:.1f}% based on industry benchmarks",
                                'volume_leverage': 'High potential with consolidated volumes',
                                'service_impact': 'Minimal expected service disruption'
                            })
                        }
                        
                        opportunities.append(opportunity)
        
        # 3. Category Consolidation Opportunities
        # Look for similar categories
        # This would require more advanced analysis, but here we'll simulate
        
        # Sample similar category pairs
        similar_categories = [
            (('102993', 'Leather straps'), ('102994', 'Nylon straps')),
            (('103245', 'Conveyor belts'), ('103246', 'Conveyor systems')),
            (('104578', 'Maintenance services'), ('104580', 'Repair services'))
        ]
        
        for (cat1_id, cat1_name), (cat2_id, cat2_name) in similar_categories:
            # Check if both categories exist in our data
            cat1_data = self.category_analytics_df[self.category_analytics_df['category_id'] == cat1_id]
            cat2_data = self.category_analytics_df[self.category_analytics_df['category_id'] == cat2_id]
            
            if not cat1_data.empty and not cat2_data.empty:
                cat1_spend = cat1_data.iloc[0]['total_spend']
                cat2_spend = cat2_data.iloc[0]['total_spend']
                total_spend = cat1_spend + cat2_spend
                
                # Assume 3-7% savings from category consolidation
                savings_percent = np.random.uniform(0.03, 0.07)
                estimated_savings = total_spend * savings_percent
                
                if estimated_savings > 7500:
                    opportunity = {
                        'opportunity_id': f"OPP{np.random.randint(100000, 999999)}",
                        'opportunity_type': 'Category Consolidation',
                        'description': f"Consolidate similar categories: {cat1_name} and {cat2_name}",
                        'affected_category': json.dumps([
                            {'id': cat1_id, 'name': cat1_name},
                            {'id': cat2_id, 'name': cat2_name}
                        ]),
                        'estimated_savings': estimated_savings,
                        'confidence_level': 'Medium',
                        
                        # Opportunity Details
                        'current_state': json.dumps({
                            'categories': [
                                {'id': cat1_id, 'name': cat1_name, 'spend': cat1_spend},
                                {'id': cat2_id, 'name': cat2_name, 'spend': cat2_spend}
                            ],
                            'total_spend': total_spend
                        }),
                        
                        'recommended_action': json.dumps({
                            'implementation_difficulty': 'High',
                            'next_steps': [
                                "Review specifications across both categories",
                                "Identify common suppliers",
                                "Develop consolidated specifications",
                                "Issue RFP for consolidated categories"
                            ]
                        }),
                        
                        # Supporting Evidence
                        'evidence': json.dumps({
                            'similarity_score': np.random.uniform(0.75, 0.95),
                            'common_suppliers': np.random.randint(2, 5),
                            'industry_benchmark': f"{savings_percent * 100:.1f}% savings typical for similar consolidations"
                        })
                    }
                    
                    opportunities.append(opportunity)
        
        # Convert to DataFrame
        opportunities_df = pd.DataFrame(opportunities)
        
        # Save to CSV
        output_path = os.path.join(self.output_dir, 'cost_saving_opportunities.csv')
        opportunities_df.to_csv(output_path, index=False)
        print(f"Saved cost saving opportunities dataset with {len(opportunities_df)} opportunities to {output_path}")
        
        return opportunities_df
    
    def generate_data_quality(self):
        """
        Generate data quality metrics.
        
        Returns:
            DataFrame with data quality metrics
        """
        print("Generating data quality metrics dataset...")
        
        # Start with categorized spend data
        if not hasattr(self, 'categorized_df'):
            self.categorized_df = self.generate_categorized_spend()
        
        # Calculate data quality metrics
        total_transactions = len(self.categorized_df)
        total_spend = self.categorized_df['extended_cost'].sum()
        
        # Categorization completeness
        categorized_mask = self.categorized_df['final_category_id'].notna() & (self.categorized_df['final_category_id'] != '')
        categorized_transactions = categorized_mask.sum()
        categorization_rate = categorized_transactions / total_transactions if total_transactions > 0 else 0
        
        high_confidence_mask = self.categorized_df['categorization_confidence'] >= 0.8
        high_confidence_categories = high_confidence_mask.sum() / total_transactions if total_transactions > 0 else 0
        
        # Simulate data completeness metrics
        completeness_metrics = {
            'supplier_data': np.random.uniform(0.97, 1.0),
            'product_descriptions': np.random.uniform(0.95, 0.99),
            'pricing_data': np.random.uniform(0.98, 1.0),
            'category_data': categorization_rate
        }
        
        # Simulate data quality issues
        quality_issues = [
            {
                'issue_type': 'Missing Product Descriptions',
                'count': int(total_transactions * (1 - completeness_metrics['product_descriptions'])),
                'impact': 'Medium'
            },
            {
                'issue_type': 'Duplicate Transactions',
                'count': int(total_transactions * np.random.uniform(0.001, 0.003)),
                'impact': 'Low'
            },
            {
                'issue_type': 'Inconsistent Supplier Names',
                'count': int(len(self.suppliers_df) * np.random.uniform(0.05, 0.1)),
                'impact': 'Medium'
            },
            {
                'issue_type': 'Uncategorized Items',
                'count': total_transactions - categorized_transactions,
                'impact': 'High'
            }
        ]
        
        # Simulate improvement metrics
        recent_improvements = [
            {
                'metric': 'Categorization Rate',
                'previous': categorization_rate - np.random.uniform(0.05, 0.1),
                'current': categorization_rate,
                'improvement': np.random.uniform(0.05, 0.1)
            },
            {
                'metric': 'High Confidence Categories',
                'previous': high_confidence_categories - np.random.uniform(0.05, 0.1),
                'current': high_confidence_categories,
                'improvement': np.random.uniform(0.05, 0.1)
            }
        ]
        
        # Create data quality record
        data_quality = {
            'scan_date': datetime.now().strftime('%Y-%m-%d'),
            'total_transactions': total_transactions,
            'total_spend': total_spend,
            
            # Categorization Completeness
            'categorized_transactions': categorized_transactions,
            'categorization_rate': categorization_rate,
            'high_confidence_categories': high_confidence_categories,
            
            # Data Completeness
            'completeness_metrics': json.dumps(completeness_metrics),
            
            # Data Quality Issues
            'quality_issues': json.dumps(quality_issues),
            
            # Recent Improvements
            'recent_improvements': json.dumps(recent_improvements)
        }
        
        # Convert to DataFrame
        data_quality_df = pd.DataFrame([data_quality])
        
        # Save to CSV
        output_path = os.path.join(self.output_dir, 'data_quality.csv')
        data_quality_df.to_csv(output_path, index=False)
        print(f"Saved data quality metrics to {output_path}")
        
        return data_quality_df
    
    def generate_all_datasets(self):
        """
        Generate all dashboard datasets.
        
        Returns:
            Dictionary with all generated DataFrames
        """
        print("Generating all dashboard datasets...")
        
        # Generate datasets in order
        categorized_spend = self.generate_categorized_spend()
        category_analytics = self.generate_category_analytics()
        supplier_analytics = self.generate_supplier_analytics()
        model_performance = self.generate_model_performance()
        cost_saving_opportunities = self.generate_cost_saving_opportunities()
        data_quality = self.generate_data_quality()
        
        print("\nAll dashboard datasets generated successfully!")
        print(f"Files saved to: {self.output_dir}")
        
        return {
            'categorized_spend': categorized_spend,
            'category_analytics': category_analytics,
            'supplier_analytics': supplier_analytics,
            'model_performance': model_performance,
            'cost_saving_opportunities': cost_saving_opportunities,
            'data_quality': data_quality
        }

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate dashboard datasets from enriched procurement data')
    parser.add_argument('--products', default="../output/enriched2/enriched_products.csv", help='Path to enriched products CSV')
    parser.add_argument('--suppliers', default="../output/enriched2/enriched_suppliers.csv", help='Path to enriched suppliers CSV')
    parser.add_argument('--transactions', default="../processed_data/processed_data.csv" , help='Optional path to transaction data CSV')
    parser.add_argument('--output', default='dashboard_data', help='Output directory for dashboard datasets')
    
    args = parser.parse_args()
    
    # Create data generator
    generator = DashboardDataGenerator(
        enriched_products_path=args.products,
        enriched_suppliers_path=args.suppliers,
        transaction_data_path=args.transactions,
        output_dir=args.output
    )
    
    # Generate all datasets
    generator.generate_all_datasets()

                        
 
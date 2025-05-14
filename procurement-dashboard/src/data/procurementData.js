// Main Categorized Spend Dataset
export const categorizedSpend = {
  "transaction_id": "TX123456",
  "date": "2024-03-15",
  "po_number": "PO987654",
  "supplier_id": "AKRB",
  "supplier_name": "Akron Belting",
  "item_number": "A53010-500-000-004-RB",
  "item_description": "REPLACEMENT BELT, DOUBLE THICK HEAVY FOR 5.5 CU YD HOPPER",
  "quantity": 5,
  "unit_cost": 125.75,
  "extended_cost": 628.75,
  "currency": "USD",
  
  // Categorization Results
  "category_path": [
    {"id": "102", "name": "Components & Supplies", "level": 1, "code": "C"},
    {"id": "102648", "name": "Manufacturing Components and Supplies", "level": 2, "code": "31000000"},
    {"id": "102966", "name": "Rope and chain and cable and wire and strap", "level": 3, "code": "31150000"},
    {"id": "102991", "name": "Straps", "level": 4, "code": "31151900"},
    {"id": "102993", "name": "Leather straps", "level": 5, "code": "31151902"}
  ],
  "final_category_id": "102993",
  "final_category_level": 5,
  "categorization_confidence": 0.92,
  "categorization_method": "ML", // "ML", "Rule-based", or "Manual"
  "needs_review": false,
  
  // Enriched Product Attributes
  "product_attributes": {
    "primary_function": "Replacement belt for hopper",
    "specifications": {"Hopper Capacity": "5.5 cubic yards"},
    "attributes": ["Replacement", "Double Thick", "Heavy Duty"]
  },
  
  // Supplier Insights
  "supplier_insights": {
    "industry": "Industrial Manufacturing",
    "market_segment": "Conveyor Systems and Components",
    "company_size": "Small",
    "risk_level": "Medium",
    "sustainability_rating": 4,
    "financial_health": 6
  }
};

// Category Analytics Dataset
export const categoryAnalytics = {
  "category_id": "102993",
  "category_name": "Leather straps",
  "category_level": 5,
  "parent_category_id": "102991",
  "parent_category_name": "Straps",
  "category_path": "Components & Supplies > Manufacturing Components > Rope and chain > Straps > Leather straps",
  "category_code": "31151902",
  
  // Spend Metrics
  "total_spend": 158432.75,
  "spend_year_to_date": 42567.50,
  "spend_previous_year": 145980.25,
  "spend_change_percent": 8.53,
  "transaction_count": 245,
  "average_transaction_value": 646.66,
  
  // Supplier Metrics
  "supplier_count": 12,
  "top_suppliers": [
    {"id": "AKRB", "name": "Akron Belting", "spend": 58432.50, "percent": 36.88},
    {"id": "BELT45", "name": "Belt Industries", "spend": 32451.75, "percent": 20.48},
    {"id": "CONVEX", "name": "Conveyor Express", "spend": 24789.00, "percent": 15.65}
  ],
  "supplier_concentration": 72.01, // Percentage of spend with top 3 suppliers
  
  // Price Analysis
  "price_range": {"min": 87.50, "max": 250.00, "average": 125.75, "median": 115.50},
  "price_volatility": 0.15, // Coefficient of variation in prices
  
  // Categorization Stats
  "categorization_confidence": {
    "average": 0.87,
    "low_confidence_items": 15,
    "manually_reviewed": 23
  }
};

// Supplier Analytics Dataset
export const supplierAnalytics = {
  "supplier_id": "AKRB",
  "supplier_name": "Akron Belting",
  "standard_name": "AKRON BELTING",
  
  // Basic Supplier Info
  "industry": "Industrial Manufacturing",
  "market_segment": "Conveyor Systems and Components",
  "company_size": "Small",
  "location": "Akron, OH, USA",
  "relationship_status": "Preferred", // Preferred, Approved, One-time, etc.
  
  // Spend Metrics
  "total_spend": 452718.25,
  "spend_year_to_date": 125431.50,
  "spend_previous_year": 415980.25,
  "spend_change_percent": 8.83,
  "transaction_count": 587,
  "average_transaction_value": 771.24,
  
  // Category Distribution
  "category_distribution": [
    {"category_id": "102993", "name": "Leather straps", "spend": 58432.50, "percent": 12.91},
    {"category_id": "102994", "name": "Nylon straps", "spend": 45781.25, "percent": 10.11},
    {"category_id": "103245", "name": "Conveyor belts", "spend": 215432.75, "percent": 47.59}
  ],
  "category_concentration": 70.61, // Percentage of spend in top 3 categories
  
  // Risk and Performance
  "risk_level": "Medium",
  "risk_factors": ["Single source for key items", "Regional supplier"],
  "sustainability_rating": 4,
  "digital_presence_score": 3,
  "financial_health": 6,
  "innovation_score": 5,
  
  // Contract Information
  "contract_status": "Active",
  "contract_expiration": "2025-08-15",
  "payment_terms": "Net 45",
  "discounts_available": ["2% 10 Net 30", "Volume discount > $50,000"]
};

// Model Performance Dataset
export const modelPerformance = {
  "model_version": "1.2.5",
  "training_date": "2025-04-01",
  "data_points_trained": 50432,
  
  // Overall Performance
  "overall_accuracy": 0.89,
  "overall_f1_score": 0.87,
  "low_confidence_predictions": 0.08, // Percentage
  
  // Performance by Level
  "level_performance": [
    {"level": 1, "accuracy": 0.96, "f1_score": 0.95, "number_of_classes": 8},
    {"level": 2, "accuracy": 0.93, "f1_score": 0.92, "number_of_classes": 28},
    {"level": 3, "accuracy": 0.90, "f1_score": 0.89, "number_of_classes": 104},
    {"level": 4, "accuracy": 0.87, "f1_score": 0.85, "number_of_classes": 312},
    {"level": 5, "accuracy": 0.81, "f1_score": 0.79, "number_of_classes": 785}
  ],
  
  // Feature Importance
  "feature_importance": [
    {"feature": "product_description_tfidf", "importance": 0.45},
    {"feature": "supplier_industry", "importance": 0.18},
    {"feature": "primary_function", "importance": 0.15},
    {"feature": "attributes", "importance": 0.12},
    {"feature": "specifications", "importance": 0.10}
  ],
  
  // Confusion Areas
  "common_misclassifications": [
    {"from_category": "102993", "to_category": "102994", "count": 45, "confusion_rate": 0.15},
    {"from_category": "103245", "to_category": "103246", "count": 38, "confusion_rate": 0.12}
  ]
};

// Cost Saving Opportunities Dataset
export const costSavingOpportunities = {
  "opportunity_id": "OPP123456",
  "opportunity_type": "Price Standardization",
  "description": "Standardize pricing for leather straps across suppliers",
  "affected_category": {"id": "102993", "name": "Leather straps"},
  "estimated_savings": 18500.00,
  "confidence_level": "High",
  
  // Opportunity Details
  "current_state": {
    "suppliers": [
      {"id": "AKRB", "name": "Akron Belting", "avg_price": 125.75},
      {"id": "BELT45", "name": "Belt Industries", "avg_price": 145.50},
      {"id": "CONVEX", "name": "Conveyor Express", "avg_price": 118.25}
    ],
    "volume": 245,
    "current_spend": 158432.75
  },
  
  "recommended_action": {
    "target_price": 118.25,
    "preferred_supplier": "CONVEX",
    "implementation_difficulty": "Medium",
    "next_steps": [
      "Review contracts with current suppliers",
      "Negotiate price match with preferred suppliers",
      "Consolidate volume with best-price supplier"
    ]
  },
  
  // Supporting Evidence
  "evidence": {
    "price_trend": "Stable",
    "market_conditions": "Competitive",
    "benchmark_data": "Industry average price is $120.50"
  }
};

// Data Quality Metrics
export const dataQuality = {
  "scan_date": "2025-05-01",
  "total_transactions": 152478,
  "total_spend": 25487651.25,
  
  // Categorization Completeness
  "categorized_transactions": 149875,
  "categorization_rate": 0.983, // Percentage of transactions categorized
  "high_confidence_categories": 0.92, // Percentage with high confidence
  
  // Data Completeness
  "completeness_metrics": {
    "supplier_data": 0.995, // Completeness rate
    "product_descriptions": 0.978,
    "pricing_data": 0.999,
    "category_data": 0.983
  },
  
  // Data Quality Issues
  "quality_issues": [
    {"issue_type": "Missing Product Descriptions", "count": 3251, "impact": "Medium"},
    {"issue_type": "Duplicate Transactions", "count": 145, "impact": "Low"},
    {"issue_type": "Inconsistent Supplier Names", "count": 278, "impact": "Medium"},
    {"issue_type": "Uncategorized Items", "count": 2603, "impact": "High"}
  ],
  
  // Recent Improvements
  "recent_improvements": [
    {"metric": "Categorization Rate", "previous": 0.923, "current": 0.983, "improvement": 0.06},
    {"metric": "High Confidence Categories", "previous": 0.85, "current": 0.92, "improvement": 0.07}
  ]
};
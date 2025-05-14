// src/data/dummyData.js

// Dashboard Overview Data
export const overviewData = {
  totalSpend: 158432.75,
  categoriesCount: 245,
  suppliersCount: 12,
  categorizationRate: 90.3,
  spendTrend: 8.5
};

// KPI Data for various charts
export const kpiData = {
  spendByMonth: [
    { month: 'Jan', spend2024: 3800000, spend2025: 4200000 },
    { month: 'Feb', spend2024: 3500000, spend2025: 3800000 },
    { month: 'Mar', spend2024: 4200000, spend2025: 4500000 },
    { month: 'Apr', spend2024: 3900000, spend2025: 4100000 },
    { month: 'May', spend2024: 4500000, spend2025: 4800000 },
    { month: 'Jun', spend2024: 4000000, spend2025: 4300000 }
  ],
  categoryDistribution: [
    { category: 'Components & Supplies', value: 35 },
    { category: 'Manufacturing Components', value: 25 },
    { category: 'Rope and Chain', value: 15 },
    { category: 'Straps', value: 15 },
    { category: 'Other Categories', value: 10 }
  ]
};

/* Spend Analysis Data
   ─────────────────────────────────────────────────────────── */
export const spendAnalysisData = {
  // Hierarchical data for a treemap (unchanged)
  hierarchicalSpend: {
    name: 'All Categories',
    children: [
      {
        name: 'Components & Supplies',
        value: 10_000_000,
        children: [
          {
            name: 'Manufacturing Components',
            value: 7_000_000,
            children: [
              {
                name: 'Rope and chain',
                value: 4_000_000,
                children: [
                  { name: 'Straps',  value: 2_000_000 },
                  { name: 'Chains',  value: 1_200_000 },
                  { name: 'Cables',  value:   800_000 }
                ]
              },
              { name: 'Fasteners', value: 1_500_000 },
              { name: 'Bearings',  value: 1_500_000 }
            ]
          },
          { name: 'Industrial Supplies', value: 3_000_000 }
        ]
      },
      { name: 'Equipment & Tools', value: 7_000_000 },
      { name: 'Services',          value: 5_000_000 }
    ]
  },

  // Heat‑map matrix (unchanged)
  supplierCategoryMatrix: [
    { supplier: 'Akron Belting',    category: 'Leather straps',  spend:  58_432.50 },
    { supplier: 'Akron Belting',    category: 'Nylon straps',    spend:  45_781.25 },
    { supplier: 'Akron Belting',    category: 'Conveyor belts',  spend: 215_432.75 },

    { supplier: 'Belt Industries',  category: 'Leather straps',  spend:  32_451.75 },
    { supplier: 'Belt Industries',  category: 'Nylon straps',    spend:  28_750.00 },
    { supplier: 'Belt Industries',  category: 'Conveyor belts',  spend: 145_670.00 },

    { supplier: 'Conveyor Express', category: 'Leather straps',  spend:  24_789.00 },
    { supplier: 'Conveyor Express', category: 'Nylon straps',    spend:  12_340.00 },
    { supplier: 'Conveyor Express', category: 'Conveyor belts',  spend:  98_540.00 }
  ],

  // Top suppliers (sorted by spend)
  topSuppliers: [
    { id: 'AKRB',   name: 'Akron Belting',      spend: 58_432.50, percent: 36.88 },
    { id: 'BELT45', name: 'Belt Industries',    spend: 32_451.75, percent: 20.48 },
    { id: 'CONVEX', name: 'Conveyor Express',   spend: 24_789.00, percent: 15.65 },
    { id: 'INDP',   name: 'Industrial Parts Inc', spend: 17_854.00, percent: 11.26 },
    { id: 'GSC',    name: 'Global Supply Co',     spend: 15_632.00, percent:  9.73 }
  ]
};

/* Supplier Insights Data
   ─────────────────────────────────────────────────────────── */
export const supplierInsightsData = {
  /* Risk‑Spend pairs (used by risk map & table) */
  supplierRisk: [
    { supplier: 'Akron Belting',      risk: 0.6, spend: 452_718.25 },
    { supplier: 'Belt Industries',    risk: 0.3, spend: 325_430.00 },
    { supplier: 'Conveyor Express',   risk: 0.5, spend: 245_980.00 },
    { supplier: 'Industrial Parts Inc', risk: 0.7, spend: 178_540.00 },
    { supplier: 'Global Supply Co',     risk: 0.4, spend: 156_320.00 }
  ],

  /* Category specialisation (feeds bar‑chart & table) */
  categorySpecialization: [
    {
      supplier: 'Akron Belting',
      categories: [
        { name: 'Leather straps', spend: 58_432.50 },
        { name: 'Nylon straps',   spend: 45_781.25 },
        { name: 'Conveyor belts', spend: 215_432.75 }
      ]
    },
    {
      supplier: 'Belt Industries',
      categories: [
        { name: 'Leather straps', spend: 32_451.75 },
        { name: 'Nylon straps',   spend: 28_750.00 },
        { name: 'Conveyor belts', spend: 145_670.00 }
      ]
    },
    {
      supplier: 'Conveyor Express',
      categories: [
        { name: 'Leather straps', spend: 24_789.00 },
        { name: 'Nylon straps',   spend: 12_340.00 },
        { name: 'Conveyor belts', spend: 98_540.00 }
      ]
    }
  ],

  /* Sustainability / Financial‑health scores (feeds bar‑chart & table) */
  supplierPerformance: [
    { supplier: 'Akron Belting',      sustainability: 4, financialHealth: 6 },
    { supplier: 'Belt Industries',    sustainability: 5, financialHealth: 7 },
    { supplier: 'Conveyor Express',   sustainability: 6, financialHealth: 5 },
    { supplier: 'Industrial Parts Inc', sustainability: 3, financialHealth: 8 },
    { supplier: 'Global Supply Co',     sustainability: 7, financialHealth: 6 }
  ]
};


// Model Performance Data
export const modelPerformanceData = {
  accuracyByLevel: [
    {level: 1, accuracy: 0.96, f1_score: 0.95, number_of_classes: 8},
    {level: 2, accuracy: 0.93, f1_score: 0.92, number_of_classes: 28},
    {level: 3, accuracy: 0.90, f1_score: 0.89, number_of_classes: 104},
    {level: 4, accuracy: 0.87, f1_score: 0.85, number_of_classes: 312},
    {level: 5, accuracy: 0.81, f1_score: 0.79, number_of_classes: 785}
  ],
  
  featureImportance: [
    {feature: "product_description_tfidf", importance: 0.45},
    {feature: "supplier_industry", importance: 0.18},
    {feature: "primary_function", importance: 0.15},
    {feature: "attributes", importance: 0.12},
    {feature: "specifications", importance: 0.10}
  ],
  
  confidenceDistribution: [
    { range: "0-10%", count: 1 },
    { range: "10-20%", count: 2 },
    { range: "20-30%", count: 3 },
    { range: "30-40%", count: 4 },
    { range: "40-50%", count: 5 },
    { range: "50-60%", count: 8 },
    { range: "60-70%", count: 12 },
    { range: "70-80%", count: 18 },
    { range: "80-90%", count: 25 },
    { range: "90-100%", count: 22 }
  ],
  
  commonMisclassifications: [
    {from_category: "102993", to_category: "102994", count: 45, confusion_rate: 0.15, from_name: "Leather straps", to_name: "Nylon straps"},
    {from_category: "103245", to_category: "103246", count: 38, confusion_rate: 0.12, from_name: "Conveyor belts", to_name: "Rubber belts"}
  ]
};

// Cost Savings Data
export const costSavingsData = {
  opportunitiesSummary: {
    totalSavings: 63350.00,
    avgDifficulty: "Medium",
    confidence: "High"
  },
  
  priceVariation: [
    { supplier: "Akron Belting", currentPrice: 125.75, targetPrice: 118.25, industryAvg: 120.50 },
    { supplier: "Belt Industries", currentPrice: 145.50, targetPrice: 118.25, industryAvg: 120.50 },
    { supplier: "Conveyor Express", currentPrice: 118.25, targetPrice: 118.25, industryAvg: 120.50 }
  ],
  
  opportunities: [
    { 
      type: "Price Standardization", 
      category: "Leather straps", 
      savings: 18500.00,
      difficulty: "Medium",
      confidence: "High"
    },
    { 
      type: "Supplier Consolidation", 
      category: "Nylon straps", 
      savings: 12750.00,
      difficulty: "Medium",
      confidence: "Medium"
    },
    { 
      type: "Volume Discount", 
      category: "Conveyor belts", 
      savings: 32100.00,
      difficulty: "Low",
      confidence: "High"
    }
  ]
};

// Data Quality Metrics
export const dataQualityData = {
  summary: {
    categorizedTransactions: 149875,
    totalTransactions: 152478,
    categorization_rate: 0.903,
    high_confidence_categories: 0.92
  },
  
  completeness: {
    supplier_data: 0.995,
    product_descriptions: 0.978,
    pricing_data: 0.999,
    category_data: 0.983
  },
  
  issues: [
    {issue_type: "Missing Product Descriptions", count: 3251, impact: "Medium"},
    {issue_type: "Duplicate Transactions", count: 145, impact: "Low"},
    {issue_type: "Inconsistent Supplier Names", count: 278, impact: "Medium"},
    {issue_type: "Uncategorized Items", count: 2603, impact: "High"}
  ],
  
  improvements: [
    {metric: "Categorization Rate", previous: 0.923, current: 0.983, improvement: 0.06},
    {metric: "High Confidence Categories", previous: 0.85, current: 0.92, improvement: 0.07}
  ]
};
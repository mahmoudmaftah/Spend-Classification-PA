# PAVUS AI Procurement Analytics System

This project implements an AI-driven categorization system to transform raw procurement transaction data into a structured and classified spend cube. The system leverages both Generative AI (using OpenAI's GPT models) and traditional machine learning techniques to automate the categorization process at scale.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Components](#pipeline-components)
- [Configuration](#configuration)
- [Dashboard](#dashboard)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)

## 📊 Overview

The PAVUS AI Procurement Analytics System processes raw procurement transaction data through a multi-phase pipeline:

1. **Data Ingestion & Preprocessing**: Clean and standardize raw procurement data and extract supplier and product features
2. **Data Enrichment**: Enhance supplier and product information using GPT models
3. **Categorization**: Classify products into a hierarchical taxonomy using machine learning models
   - The system supports custom taxonomies and can be trained on user-defined categories
   - It also provides confidence scores for each classification
4. **Spend Cube Generation**: Create structured spend analytics for visualization
5. **Dashboard**: A Vue.js & Tailwind css dashboard with advanced features to visualize spend analytics and insights

The system is designed to be scalable, configurable, and adaptable to different procurement datasets and categorization needs.

## 🗂️ Project Structure

```
pavus-ai-procurement/
├── procurement_data_pipeline.py      # Data preprocessing module
├── procurement_data_enrichment.py    # LLM-based data enrichment
├── procurement_categorization.py     # Categorization algorithms
├── end_to_end_pipeline.py            # Main pipeline orchestration
├── dashboard_app.py                  # Streamlit dashboard
├── requirements.txt                  # Project dependencies
├── ML/                               # Machine learning models and utilities 
├── data/                             # Input data directory
├── pavus_output/                     # Output directory
│   ├── preprocessing/                # Preprocessed data
│   ├── enrichment/                   # Enriched data and taxonomy
│   ├── categorization/               # Trained models and predictions
│   └── spend_cube/                   # Spend cube analyses and visualizations
└── README.md                         # This file
```

## 🛠️ Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-organization/pavus-ai-procurement.git
   cd pavus-ai-procurement
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   
   Create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

## 🚀 Usage

### Running the End-to-End Pipeline

1. Prepare your procurement data in CSV or Excel format with these recommended columns:
   - Transaction date
   - Supplier ID and name
   - Product ID and description  
   - Quantity and cost information

2. Run the pipeline:
   ```bash
   python end_to_end_pipeline.py
   ```

   By default, this will:
   - Look for data in the `data/` directory
   - Process the file named `expanded_procurement_data.csv`
   - Output results to the `pavus_output/` directory

3. Customize the pipeline by modifying the configuration in `end_to_end_pipeline.py`

### Running the Dashboard

Launch the dashboard to visualize the spend cube results:

```bash
npm install
npm run build
npm run serve
```

The dashboard will be available at `http://localhost:8080`. You can enter the path to your output directory (default: `pavus_output`) to visualize the results.

## 🧩 Pipeline Components

### 1. Data Preprocessing (`procurement_data_pipeline.py`)

- Data loading and validation
- Cleaning and standardization
- Supplier name standardization
- Product feature extraction

### 2. Data Enrichment (`procurement_data_enrichment.py`)

- Supplier attribute extraction using OpenAI
- Product attribute extraction using OpenAI
- Taxonomy generation
- Category suggestions

### 3. Categorization (`procurement_categorization.py`)

- Feature engineering
- Model training for hierarchical classification
- Prediction with confidence scoring
- Model evaluation and visualization



## ⚙️ Configuration

The pipeline can be configured by modifying the `config` dictionary in `end_to_end_pipeline.py`:

```python
config = {
    'output_dir': 'pavus_output',        # Output directory
    'openai_model': 'gpt-4-turbo',       # OpenAI model to use
    'use_llm_enrichment': True,          # Enable/disable LLM enrichment
    'sample_size_for_enrichment': 50,    # Maximum items to enrich with LLM
    'openai_api_key': None               # Will use from .env if not provided
}
```

## 🤖 Pre-trained Model Weights

This project includes pre-trained model weights for predicting product taxonomy labels at each level. These models were trained on a specific procurement taxonomy dataset:

| Model Level | Description | Download Link |
|-------------|-------------|--------------|
| Level 1 | Top category classification | [Download Level 1 Weights](https://um6p-my.sharepoint.com/:u:/g/personal/mahmoud_maftah_um6p_ma/EQHLBEpZF0JArWH_VKNykIABQWgepcNiOte1nZ-uXjaWvA?e=Z9QQHs) |
| Level 2 | Sub-category classification | [Download Level 2 Weights](https://um6p-my.sharepoint.com/:u:/g/personal/mahmoud_maftah_um6p_ma/EQVCsQFLbu5Aia8VlQe3Tj4BLCH7Kl8Hpv7CRusz4qpBoA?e=eE97cV) |
| Level 3 | Product family classification | [Download Level 3 Weights](https://um6p-my.sharepoint.com/:u:/g/personal/mahmoud_maftah_um6p_ma/EQ_YmMzIPchGvVyPaDR8rc8Bxbd8sQV2p-yVcPrU0-NXCg?e=WzQicp) |
| Level 4 | Detailed product classification | [Download Level 4 Weights](https://um6p-my.sharepoint.com/:u:/g/personal/mahmoud_maftah_um6p_ma/Ec4rTAILgHVDm-GvhMAYwKwBBU7DXtfqWL_3GxOfoa57aA?e=iTKBxW) |
| Level 5 | Detailed product classification | [Download Level 5 Weights](https://um6p-my.sharepoint.com/:u:/g/personal/mahmoud_maftah_um6p_ma/ETZ_7GdEZV5BlaZSbfeqD_cBDh_wPCxK25s3ko9xrAwIsg?e=AmdrOw) |

To use these pre-trained weights:
```python
from model.taxonomy_classifier import TaxonomyClassifier

# Initialize model with pre-trained weights
classifier = TaxonomyClassifier()
classifier.load_weights('path/to/downloaded/weights.h5')

# Predict taxonomy for a product description
predictions = classifier.predict("Office chair with adjustable height")
```

## 📈 Dashboard

The Streamlit dashboard provides comprehensive procurement analytics and insights through multiple visualization modules:

### 1. Spend Analysis Insights
- Spend by category with time series and year-over-year comparison
- Category concentration risk analysis (e.g., "80% of IT spend goes to 3 vendors")
- Maverick spend identification (purchases outside preferred suppliers)
- Contract compliance rate tracking

### 2. Supplier Performance Insights
- Spend distribution across suppliers within each category
- Price variance analysis for similar products
- Supplier diversification metrics
- Payment term compliance monitoring

### 3. Process Efficiency Insights
- Procurement cycle times by category
- Approval workflow bottleneck identification
- P2P process exception tracking
- Tail spend analysis (small, unmanaged purchases)

### 4. Budget & Cost Control Insights
- Budget vs. actual comparisons by category
- Cost avoidance opportunity identification
- Volume discount opportunity analysis
- Seasonal spending pattern visualization

### 5. Advanced Analytics
- Predictive spend forecasting
- Anomaly detection for unusual spending patterns
- Should-cost modeling
- Carbon footprint tracking by category (for ESG reporting)

### Interactive Visualization Types
- Treemap View: Nested rectangles showing spend hierarchy
- Spend Velocity Chart: Daily/weekly spend trends by category
- Supplier Risk Matrix: Spend concentration vs. financial health
- Category Benchmarking: Your spend % vs. industry standards
- Savings Opportunity Heatmap: Potential savings by category


## 🔧 Customization

### Adding Custom Classification Models

Modify `procurement_categorization.py` to add custom classification models:

```python
# In the _set_default_config method
self.config.setdefault('model_type', 'your_custom_model')

# In the train_models method
if self.config['model_type'] == 'your_custom_model':
    from your_module import YourModel
    model = YourModel(params)
```

### Modifying the Taxonomy

The system can work with custom taxonomies:
1. Create a JSON file with your taxonomy following the format in `output/enrichment/taxonomy.json`
2. Specify this file in the configuration:
   ```python
   categorization_config = {
       'taxonomy_path': 'path/to/your/taxonomy.json'
   }
   ```

## 🔍 Troubleshooting

### Common Issues

**OpenAI API Key Issues**
- Ensure your API key is correctly set in the `.env` file or passed directly to the configuration
- Check for sufficient credits in your OpenAI account

**Memory Errors with Large Datasets**
- Reduce batch sizes by setting smaller `sample_size_for_enrichment` in the configuration
- Process data in chunks if necessary

**Missing Dependencies**
- Run `pip install -r requirements.txt` to ensure all dependencies are installed
- Some visualizations require optional packages like `wordcloud`

### Logging

The pipeline uses Python's logging module to log information and errors:
- Check the log files (`procurement_pipeline.log`, `procurement_enrichment.log`, etc.)
- Set more verbose logging by modifying the logging level in each module

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
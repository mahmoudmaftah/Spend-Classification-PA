# Procurement Data Classification System

This project provides a comprehensive solution for classifying procurement data using various machine learning techniques. It enables automatic categorization of products based on their descriptions, specifications, and attributes.

## Features

- **Advanced Text Processing**: Utilizes NLP techniques for feature extraction from product descriptions
- **Multiple Classification Models**: Trains and compares various ML models (Random Forest, Gradient Boosting, SVM, etc.)
- **Dimensionality Reduction**: Applies PCA, t-SNE, and UMAP for data visualization
- **Clustering Analysis**: Performs K-means and other clustering algorithms to discover patterns
- **Model Comparison**: Visualizes model performance metrics with detailed plots
- **User Feedback Integration**: Supports incremental learning when manual corrections are provided
- **Comprehensive Reporting**: Generates detailed reports of classification results and performance metrics

## Getting Started

### Prerequisites

Ensure you have the following installed:
- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- nltk
- umap-learn (optional, for UMAP dimensionality reduction)

Install dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn nltk
pip install umap-learn  # Optional
```

### Data Format

The system expects a CSV file with the following columns:

- `Item Number`: Unique identifier for each item
- `Item Description`: Text description of the product
- `clean_description`: Processed version of the description (can be the same as Item Description)
- `extracted_keywords`: List of keywords (in JSON format)
- `attributes`: Product attributes (in JSON format)
- `specifications`: Product specifications (in JSON format)
- `primary_function`: Description of the product's function
- `category_l1_name`, `category_l2_name`, etc.: Target category levels for classification

### Basic Usage

1. Place your data CSV file in the project directory.

2. Run the classifier script:

```bash
python procurement_classifier.py --data your_data.csv --output results
```

3. For a demonstration using sample data:

```bash
python procurement_demo.py
```

### Advanced Usage

#### Training Specific Models

To train specific models for certain category levels:

```bash
python procurement_classifier.py --data your_data.csv --output results --targets category_l1_name category_l3_name
```

#### Customizing the Pipeline

You can customize the classification pipeline by modifying the code:

```python
from procurement_classifier import ProcurementClassifier

# Initialize the classifier
classifier = ProcurementClassifier(data_path='your_data.csv', output_dir='results')

# Load and preprocess data
classifier.load_data()

# Engineer features with a different test split
classifier.engineer_features(test_size=0.3)

# Create text vectors with more features
classifier.create_text_vectors(method='tfidf', max_features=1000)

# Train only specific models
models_to_train = ['random_forest', 'gradient_boosting']
classifier.train_models('category_l1_name', models_to_train=models_to_train)

# Evaluate models
classifier.evaluate_models('category_l1_name')

# Visualize clusters
classifier.visualize_clusters('category_l1_name', n_clusters=8)

# Save the trained models
classifier.save_models()
```

#### Using Trained Models for Prediction

Once models are trained, you can use them for prediction:

```python
# Load existing models
classifier.load_models('results/models')

# Make predictions
predictions, probabilities = classifier.predict(new_data, 'category_l1_name')
```

#### Integrating User Feedback

The system supports integrating user feedback to improve models:

```python
# Update a model with user feedback
classifier.update_model_with_feedback(
    'category_l1_name',          # Target column
    'random_forest',             # Model name
    feedback_data,               # New data with features
    correct_labels               # Correct labels provided by users
)
```

## Code Structure

- `procurement_classifier.py`: Main classification system with complete functionality
- `procurement_demo.py`: Demonstration script with sample data generation

## Output

The system generates the following outputs in the specified output directory:

- **Models**: Serialized trained models for each category level
- **Plots**: Visualization of model performance, feature importance, and clusters
- **Report**: Comprehensive markdown report summarizing the classification results

## Extending the System

To extend the system for your specific needs:

1. **Custom Feature Engineering**: Modify the `engineer_features()` method to add domain-specific features
2. **New Models**: Add additional classification models in the `train_models()` method
3. **Custom Preprocessing**: Enhance text preprocessing in the `preprocess_text()` method
4. **Advanced Visualizations**: Add new visualization methods as needed

## License

This project is licensed under the MIT License - see the LICENSE file for details.
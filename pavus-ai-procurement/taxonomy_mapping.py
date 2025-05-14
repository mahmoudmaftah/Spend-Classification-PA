import pandas as pd
import os
from dotenv import load_dotenv
import openai
import logging
from typing import Dict, List, Optional, Tuple, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TaxonomyNode:
    """Represents a node in the taxonomy tree."""
    
    def __init__(self, key: str, code: str, title: str, parent_key: Optional[str] = None):
        self.key = key
        self.code = code
        self.title = title
        self.parent_key = parent_key
        self.children = []
    
    def add_child(self, child: 'TaxonomyNode') -> None:
        """Add a child node to this node."""
        self.children.append(child)
    
    def __repr__(self) -> str:
        return f"<TaxonomyNode key={self.key}, title='{self.title}', children={len(self.children)}>"


class TaxonomyTree:
    """Represents the entire taxonomy tree structure."""
    
    def __init__(self):
        self.nodes_by_key = {}  # Dict mapping key to node
        self.root_nodes = []    # List of top-level nodes
    
    def add_node(self, node: TaxonomyNode) -> None:
        """Add a node to the tree."""
        self.nodes_by_key[node.key] = node
        
        # If this is a root node (no parent), add to root_nodes list
        if node.parent_key is None or pd.isna(node.parent_key):
            self.root_nodes.append(node)
        else:
            # Add as child to parent if parent exists
            parent_key = str(int(float(node.parent_key))) if not pd.isna(node.parent_key) else None
            if parent_key and parent_key in self.nodes_by_key:
                self.nodes_by_key[parent_key].add_child(node)
    
    def get_node(self, key: str) -> Optional[TaxonomyNode]:
        """Get a node by its key."""
        return self.nodes_by_key.get(key)
    
    def max_depth(self) -> int:
        """Calculate the maximum depth of the taxonomy tree."""
        def _max_depth(node: TaxonomyNode) -> int:
            if not node.children:
                return 1
            return 1 + max(_max_depth(child) for child in node.children)
        
        return max(_max_depth(node) for node in self.root_nodes) if self.root_nodes else 0
    
    def max_degree(self) -> int:
        """Calculate the maximum degree (number of children) of any node in the tree."""
        def _max_degree(node: TaxonomyNode) -> int:
            return max(len(node.children), max((_max_degree(child) for child in node.children), default=0)) if node.children else 0
        
        return max(_max_degree(node) for node in self.root_nodes) if self.root_nodes else 0


class GPT4Classifier:
    """Uses GPT-4 to classify items within a taxonomy."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4-turbo"):
        # Load environment variables from .env file if it exists
        load_dotenv()
        
        # Use provided API key or get from environment
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("No OpenAI API key provided. Set OPENAI_API_KEY environment variable or pass key to constructor.")
        
        self.model = model
        self.client = openai.OpenAI(api_key=self.api_key) if self.api_key else None
    
    def classify(self, description: str, options: List[Dict[str, str]]) -> str:
        """
        Use GPT-4 to classify a description into one of the provided options.
        
        Args:
            description: The spend description to classify
            options: A list of dictionaries, each containing 'key' and 'title' of a taxonomy node
            
        Returns:
            The key of the selected option
        """
        if not self.client:
            raise ValueError("OpenAI client not initialized. Please provide a valid API key.")
        
        # Format options for the prompt
        options_text = "\n".join([f"{i+1}. {opt['title']} (ID: {opt['key']})" for i, opt in enumerate(options)])

        # print(f"Options: {options_text}")
        
        # Create the prompt
        prompt = f"""I need to classify this spend description into the most appropriate category.

Description: {description}

Available categories:
{options_text}

Select the most appropriate category by responding with ONLY the ID number of the category. 
Do not include any explanation or additional text in your response, just the ID number."""
        
        try:
            # Call the OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # Low temperature for more deterministic responses
                max_tokens=20     # We only need a very short response
            )
            
            # Extract the category ID from the response
            gpt_response = response.choices[0].message.content.strip()
            
            # Try to find the selected option
            for option in options:
                if option['key'] in gpt_response:
                    return option['key']
            
            # If we couldn't match the response directly to a key,
            # log a warning and return the first option as a fallback
            logger.warning(f"Couldn't match GPT-4 response '{gpt_response}' to any option key. Using first option as fallback.")
            return options[0]['key'] if options else None
            
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            # Return the first option as a fallback in case of error
            return options[0]['key'] if options else None


class TaxonomyClassifier:
    """Main class for classifying items using the taxonomy."""
    
    def __init__(self, taxonomy_file: str, api_key: Optional[str] = None, model: str = "gpt-4"):
        self.taxonomy = self._load_taxonomy(taxonomy_file)
        self.classifier = GPT4Classifier(api_key, model)
    
    def _load_taxonomy(self, file_path: str) -> TaxonomyTree:
        """
        Load the taxonomy from an Excel file.
        
        Args:
            file_path: Path to the Excel file containing the taxonomy
            
        Returns:
            A TaxonomyTree object representing the loaded taxonomy
        """
        logger.info(f"Loading taxonomy from {file_path}")
        
        # Read the Excel file
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}. Please use .xlsx, .xls, or .csv.")
        
        # Create the taxonomy tree
        tree = TaxonomyTree()
        
        # First, create all nodes
        for _, row in df.iterrows():
            key = str(row['Key'])
            parent_key = row['Parent key'] if not pd.isna(row['Parent key']) else None
            code = str(row['Code'])
            title = str(row['Title'])
            
            node = TaxonomyNode(key, code, title, parent_key)
            tree.add_node(node)

        # print the maximum depth and degree of the taxonomy tree
        max_depth = tree.max_depth()
        max_degree = tree.max_degree()
        logger.info(f"Max depth of taxonomy tree: {max_depth}")
        logger.info(f"Max degree of taxonomy tree: {max_degree}")

        
        logger.info(f"Loaded taxonomy with {len(tree.nodes_by_key)} nodes and {len(tree.root_nodes)} root categories")
        return tree
    
    def classify_item(self, description: str) -> Dict[str, Any]:
        """
        Classify an item by navigating through the taxonomy tree using GPT-4.
        
        Args:
            description: The description of the item to classify
            
        Returns:
            A dictionary containing the classification path and final classification
        """
        logger.info(f"Classifying item: {description}")
        
        # Start with the root nodes
        current_nodes = self.taxonomy.root_nodes
        classification_path = []
        
        # Navigate down the tree until we reach a leaf node or GPT-4 can't make a decision
        while current_nodes:
            # Format options for GPT-4
            options = [{"key": node.key, "title": node.title} for node in current_nodes]
            
            # Get the classification from GPT-4
            selected_key = self.classifier.classify(description, options)
            
            if not selected_key:
                break
            
            # Get the selected node
            selected_node = self.taxonomy.get_node(selected_key)
            
            if not selected_node:
                logger.warning(f"Selected key {selected_key} not found in taxonomy")
                break
            
            # Add to classification path
            classification_path.append({
                "key": selected_node.key,
                "code": selected_node.code,
                "title": selected_node.title
            })
            
            # If this node has children, continue down the tree
            if selected_node.children:
                current_nodes = selected_node.children
            else:
                # We've reached a leaf node
                break
        
        # Return the classification result
        result = {
            "input": description,
            "classification_path": classification_path,
            "final_classification": classification_path[-1] if classification_path else None
        }
        
        logger.info(f"Classification result: {result['final_classification']['title'] if result['final_classification'] else 'None'}")
        return result


def main():
    """Example usage of the TaxonomyClassifier."""
    # Initialize the classifier
    classifier = TaxonomyClassifier("taxonomy.xlsx")
    
    # Example items to classify
    items = [
        "Purchase of 2 German Shepherd puppies for the K9 unit",
        "Office supplies including pens, paper, and staplers",
        # "Construction of a new warehouse facility",
        # "IT consulting services for database migration",
        # "Laboratory equipment for testing water samples"
    ]
    
    # Classify each item
    for item in items:
        result = classifier.classify_item(item)
        
        print(f"\nItem: {item}")
        print("Classification path:")
        for i, node in enumerate(result["classification_path"]):
            print(f"  {i+1}. {node['title']} (Code: {node['code']})")
        print(f"Final classification: {result['final_classification']['title']} (Code: {result['final_classification']['code']})")


if __name__ == "__main__":
    main()
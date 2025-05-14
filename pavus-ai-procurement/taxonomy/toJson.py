import pandas as pd
import json
import logging
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def excel_to_json(excel_file, json_file):
    """
    Convert taxonomy Excel file to JSON format.
    
    Args:
        excel_file: Path to the Excel file
        json_file: Path where JSON will be saved
    """
    logger.info(f"Converting {excel_file} to JSON format")
    
    # Read the Excel file
    if excel_file.endswith('.xlsx') or excel_file.endswith('.xls'):
        df = pd.read_excel(excel_file)
    elif excel_file.endswith('.csv'):
        df = pd.read_csv(excel_file)
    else:
        raise ValueError(f"Unsupported file format: {excel_file}. Please use .xlsx, .xls, or .csv.")
    
    # Convert to desired JSON format
    taxonomy = []
    
    # Track levels for each node
    node_levels = {}
    
    # First pass: Create all nodes with basic info
    for _, row in df.iterrows():
        node_id = str(row['Key'])
        parent_id = str(int(float(row['Parent key']))) if not pd.isna(row['Parent key']) else None
        
        # Determine level (1 for root nodes, children have parent's level + 1)
        if parent_id is None:
            level = 1
        else:
            # If parent level is not determined yet, set a placeholder
            level = None
        
        node_levels[node_id] = level
        
        # Create node with name and description from title
        node = {
            "id": node_id,
            "name": str(row['Title']),
            "level": level,
            "parent_id": parent_id,
            "description": f"Category code: {row['Code']}",
            "code": str(row['Code'])  # Adding code as an additional field
        }
        
        taxonomy.append(node)
    
    # Second pass: Determine levels for nodes where parent's level wasn't known
    need_update = True
    while need_update:
        need_update = False
        for node in taxonomy:
            if node["level"] is None:
                parent_id = node["parent_id"]
                if parent_id in node_levels and node_levels[parent_id] is not None:
                    node["level"] = node_levels[parent_id] + 1
                    node_levels[node["id"]] = node["level"]
                    need_update = True
    
    # Remove any temporary fields
    to_drop = []
    for node in taxonomy:
        # If we still couldn't determine the level, default to level 1
        if node["level"] is None:
            # print its id for debugging
            logger.warning(f"Node {node['id']} has no parent, setting level to 1")
            
            # drop the nodes with no parent
            node["level"] = 1
            to_drop.append(node["id"])

    
    # Remove nodes with no parent
    taxonomy = [node for node in taxonomy if node["id"] not in to_drop]
    

        
    
    # Write to JSON file
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(taxonomy, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Taxonomy converted and saved to {json_file}")
    logger.info(f"Total nodes: {len(taxonomy)}")
    
    # Print number of nodes per level
    level_counts = {}
    for node in taxonomy:
        level = node["level"]
        level_counts[level] = level_counts.get(level, 0) + 1
    
    for level, count in sorted(level_counts.items()):
        logger.info(f"Level {level}: {count} nodes")
    
    return taxonomy

def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(description='Convert taxonomy Excel file to JSON format')
    parser.add_argument('--input', '-i', help='Path to input Excel file', default='taxonomy.xlsx')
    parser.add_argument('--output', '-o', help='Path to output JSON file', default='taxonomy.json')
    
    args = parser.parse_args()
    excel_to_json(args.input, args.output)

if __name__ == "__main__":
    main()
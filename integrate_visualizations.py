"""
Visualization Integration Script for Cricket Commentary Analysis

This script integrates the visualization module with the existing cricket
commentary analysis system. It adds visualization capabilities to the final.py script.

Usage:
1. Place this file in the project root directory
2. Run: python integrate_visualizations.py
"""

import os
import sys
import importlib.util
import shutil


def create_backup(file_path):
    """Create a backup of the original file"""
    backup_path = file_path + '.bak'
    if not os.path.exists(backup_path):
        shutil.copy2(file_path, backup_path)
        print(f"Created backup of {file_path} at {backup_path}")
    return backup_path


def modify_final_py():
    """Modify the final.py file to include visualization capabilities"""
    final_py_path = "final.py"

    # Create backup
    create_backup(final_py_path)

    # Read the original file
    with open(final_py_path, 'r') as file:
        content = file.read()

    # Add import statement for the visualization module
    if "from commentary_visualizer import create_visualizations" not in content:
        import_statement = "import sys\nsys.path.append(\".\")  # Add current directory to path\nfrom Code.vAnds import v_and_s\n"
        replacement = "import sys\nsys.path.append(\".\")  # Add current directory to path\nfrom Code.vAnds import v_and_s\nfrom commentary_visualizer import create_visualizations\n"
        content = content.replace(import_statement, replacement)

    # Add code to call the visualization module
    visualization_code = """
# Add these lines after the v_and_s call
if isinstance(analysis_results, dict) and 'classifications' in analysis_results:
    print("\\n" + "=" * 80)
    print("\\nGENERATING VISUALIZATIONS...")
    print("=" * 80)
    try:
        create_visualizations(text_lines, analysis_results['classifications'])
        print("\\nVisualizations have been created in Data/Visualizations/")
    except Exception as e:
        print(f"Error creating visualizations: {e}")
"""

    # Find the position to insert the visualization code
    target_pattern = "if isinstance(analysis_results, dict) and 'classifications' in analysis_results:"
    position = content.find(target_pattern)

    # If the target pattern is found and the visualization code is not already added
    if position != -1 and "GENERATING VISUALIZATIONS" not in content:
        # Find the end of the existing code block
        code_block_end = content.find("print(\"\\n\" + \"=\"*80)", position)
        if code_block_end != -1:
            # Insert our visualization code before the next code block
            new_content = content[:code_block_end] + visualization_code + content[code_block_end:]

            # Write the modified content back to the file
            with open(final_py_path, 'w') as file:
                file.write(new_content)

            print(f"Successfully modified {final_py_path} to include visualization capabilities")
            return True

    # If the file was already modified
    if "GENERATING VISUALIZATIONS" in content:
        print(f"{final_py_path} already includes visualization capabilities")
        return True

    print(f"Could not modify {final_py_path}. Manual integration required.")
    return False


def create_example_visualization():
    """Create a demonstration visualization script for users to test"""
    demo_file = "visualize_demo.py"

    content = """#!/usr/bin/env python3
\"\"\"
Cricket Commentary Visualization Demo

This script demonstrates the visualization capabilities without running the full analysis.
It uses sample data to generate all visualizations.
\"\"\"

import os
import sys
import csv
import random
from collections import Counter

# Ensure the current directory is in the path
sys.path.append(".")

# Import the visualization module
from commentary_visualizer import create_visualizations

# Check if input data exists
input_file = "Data/Input/input.csv"
if not os.path.exists(input_file):
    print(f"Error: Input data not found at {input_file}")
    print("Please run input.py first to prepare the input data.")
    exit(1)

# Read the commentary data
text_lines = []
current_over = []

print(f"Reading commentary data from {input_file}")
with open(input_file, 'r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  # Skip the header row

    for row in csv_reader:
        commentary = row[0]
        over_complete = row[1].lower() == 'true'

        current_over.append(commentary)

        if over_complete:
            text_lines.append(current_over)
            current_over = []

# If there are any remaining balls in the current over, add it to the text_lines list
if current_over:
    text_lines.append(current_over)

# Create sample classification data if actual classifications aren't available
def create_sample_classifications(text_lines):
    categories = [
        'technical analysis',
        'player praise',
        'player criticism', 
        'strategic observation',
        'audience reaction'
    ]

    # Assign a random category to each over
    classifications = []
    for i, over in enumerate(text_lines):
        # Weight the categories to create more realistic distribution
        weights = [0.4, 0.25, 0.15, 0.15, 0.05]
        category = random.choices(categories, weights=weights)[0]

        classifications.append({
            'over_number': i + 1,
            'dominant_category': category,
            'avg_confidence': random.uniform(0.7, 0.95)
        })

    return classifications

# Create sample classifications
print(f"Creating sample classifications for {len(text_lines)} overs")
sample_classifications = create_sample_classifications(text_lines)

# Generate visualizations
print("Generating visualizations...")
create_visualizations(text_lines, sample_classifications)

print("\\nVisualization demo completed!")
print("Check the Data/Visualizations/ directory for the generated visualizations.")
"""

    with open(demo_file, 'w') as file:
        file.write(content)

    # Make the file executable
    os.chmod(demo_file, 0o755)

    print(f"Created demo visualization script at {demo_file}")
    return True


def main():
    """Main function to integrate visualizations"""
    print("=" * 80)
    print("Cricket Commentary Visualization Integration")
    print("=" * 80)

    # Check if the visualization module exists
    if not os.path.exists("commentary_visualizer.py"):
        print("Error: commentary_visualizer.py not found!")
        print("Please make sure the visualization module is in the project root directory.")
        return False

    # Create necessary directories
    os.makedirs("Data/Visualizations", exist_ok=True)

    # Try to modify final.py
    modified = modify_final_py()

    # Create the demo visualization script
    created_demo = create_example_visualization()

    if modified and created_demo:
        print("\nIntegration successful!")
        print("\nYou can now use the visualization capabilities in the following ways:")
        print("1. Run the regular analysis with visualizations:")
        print("   python final.py")
        print("2. Run just the visualization demo on existing data:")
        print("   python visualize_demo.py")
        print("\nAll visualizations will be saved in the Data/Visualizations/ directory.")
        return True
    else:
        print("\nIntegration was partially successful.")
        print("Please check the output above for specific issues.")
        return False


if __name__ == "__main__":
    main()
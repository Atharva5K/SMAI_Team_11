#!/usr/bin/env python3
"""
Cricket Commentary Visualization Demo

This script demonstrates the visualization capabilities without running the full analysis.
It uses sample data to generate all visualizations.
"""

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

print("\nVisualization demo completed!")
print("Check the Data/Visualizations/ directory for the generated visualizations.")

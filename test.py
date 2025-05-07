import os
import csv
import sys
sys.path.append(".")  # Add current directory to path
from Code.vAnds import v_and_s

print("Starting test script...")

# Check if the input file exists
input_file = 'Data/Input/input.csv'
if not os.path.exists(input_file):
    print(f"Error: Input file not found at {input_file}")
    print("Please run input.py first to prepare the input data.")
    exit(1)

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

# Print the text_lines list
print(f"Found {len(text_lines)} complete overs")
print("Analyzing commentary for interesting moments...")
v_and_s(text_lines, 10)

print("Test completed successfully!")
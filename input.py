# import os
# import pandas as pd
#
# # Create directories if they don't exist
# os.makedirs("Data", exist_ok=True)
# os.makedirs("Data/COMMENTARY_INTL_MATCH", exist_ok=True)
# os.makedirs("Data/Input", exist_ok=True)
#
# # Set the paths
# data_dir = 'Data/COMMENTARY_INTL_MATCH/'
# input_dir = 'Data/Input/'
#
# # Create the input directory if it doesn't exist
# os.makedirs(input_dir, exist_ok=True)
#
# print(f"Looking for CSV files in {data_dir}")
# csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
#
# if not csv_files:
#     print(f"No CSV files found in {data_dir}. Please make sure you've downloaded the data.")
#     exit(1)
#
# # Use the first CSV file if 1244025_COMMENTARY.csv doesn't exist
# target_file = '1244025_COMMENTARY.csv'
# if target_file not in csv_files:
#     target_file = csv_files[0]
#     print(f"Using {target_file} as input file")
# else:
#     print(f"Found target file: {target_file}")
#
# # Read the CSV file
# csv_file = os.path.join(data_dir, target_file)
# print(f"Reading file: {csv_file}")
# df = pd.read_csv(csv_file)
#
# # Check if required columns exist
# required_columns = ['Commentary', 'Over_complete']
# missing_columns = [col for col in required_columns if col not in df.columns]
#
# if missing_columns:
#     print(f"Error: Missing required columns: {missing_columns}")
#     print(f"Available columns: {df.columns.tolist()}")
#     exit(1)
#
# # Extract the desired columns
# columns_to_extract = ['Commentary', 'Over_complete']
# extracted_df = df[columns_to_extract]
#
# # Write the extracted columns to a new CSV file
# output_file = os.path.join(input_dir, 'input.csv')
# extracted_df.to_csv(output_file, index=False)
#
# print(f"Extracted columns written to: {output_file}")

import os
import sys
import pandas as pd
import argparse


def list_available_files(data_dir):
    """List all CSV files in the data directory"""
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not files:
        print(f"No CSV files found in {data_dir}")
        return []
    print("\nAvailable match files:")
    for idx, file in enumerate(files):
        print(f"  {idx + 1}. {file}")
    return files


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process cricket match commentary.')
    parser.add_argument('-f', '--file', help='Specific match commentary file to analyze')
    parser.add_argument('-l', '--list', action='store_true', help='List available match files')
    args = parser.parse_args()

    # Set the relative paths - FIX: Removed leading slash and adjusted for running from Code directory
    data_dir = "../Data/COMMENTARY_INTL_MATCH"
    input_dir = "../Data/Input"

    # Create the input directory if it doesn't exist
    os.makedirs(input_dir, exist_ok=True)

    # Check if directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} not found.")
        sys.exit(1)

    # Get list of available files
    available_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

    # If --list flag is used, show available files and exit
    if args.list:
        list_available_files(data_dir)
        sys.exit(0)

    # Determine which file to use
    target_file = None

    if args.file:
        # User specified a file
        if args.file in available_files:
            target_file = args.file
            print(f"Using specified file: {target_file}")
        else:
            print(f"Error: Specified file '{args.file}' not found.")
            files = list_available_files(data_dir)
            if files:
                print("\nPlease choose from the available files or provide a full path.")
            sys.exit(1)
    else:
        # No file specified, prompt user to select one
        files = list_available_files(data_dir)
        if not files:
            print("No CSV files found. Please add match files to the directory.")
            sys.exit(1)

        while True:
            try:
                choice = input("\nEnter the number of the match to analyze or 'q' to quit: ")
                if choice.lower() == 'q':
                    sys.exit(0)

                idx = int(choice) - 1
                if 0 <= idx < len(files):
                    target_file = files[idx]
                    break
                else:
                    print(f"Please enter a number between 1 and {len(files)}")
            except ValueError:
                print("Please enter a valid number")

    # Read the CSV file
    csv_file = os.path.join(data_dir, target_file)
    print(f"\nReading match commentary from: {csv_file}")

    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    # Verify required columns exist
    required_columns = ['Commentary', 'Over_complete']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        print(f"Error: File is missing required columns: {missing_columns}")
        print(f"Available columns: {df.columns.tolist()}")
        sys.exit(1)

    # Extract the desired columns
    columns_to_extract = ['Commentary', 'Over_complete']
    extracted_df = df[columns_to_extract]

    # Write the extracted columns to a new CSV file
    output_file = os.path.join(input_dir, 'input.csv')
    extracted_df.to_csv(output_file, index=False)

    print(f"Success! Extracted {len(extracted_df)} commentary lines to: {output_file}")
    print("You can now run final.py to analyze this match.")


if __name__ == "__main__":
    main()
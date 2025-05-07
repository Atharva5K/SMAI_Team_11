# import os
# import csv
# import pickle
# import pandas as pd
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.preprocessing.text import Tokenizer
# from cricket_terminal import Colors
# import sys
# sys.path.append(".")
# from cricket_terminal import *  # Import cricket-themed terminal styling  # Add current directory to path
# from Code.vAnds import v_and_s
# from commentary_visualizer import create_visualizations
# from cricket_analysis.team_extractor import extract_teams, get_team_names, get_match_summary
# from cricket_analysis.scorecard_generator import generate_scorecard, get_full_scorecard
#
# print("Cricket Commentary Analysis - Starting...")
#
# # Check if the trained model exists
# model_path = "Code/Pickles/trained_model.pkl"
# if not os.path.exists(model_path):
#     print(f"Error: Trained model not found at {model_path}")
#     print("Please run rnn.py first to train the model.")
#     exit(1)
#
# # Load the trained model
# print(f"Loading trained model from {model_path}")
# with open(model_path, "rb") as file:
#     model = pickle.load(file)
#
# # Check if the input data exists
# input_file = "Data/Input/input.csv"
# if not os.path.exists(input_file):
#     print(f"Error: Input data not found at {input_file}")
#     print("Please run input.py first to prepare the input data.")
#     exit(1)
#
# # Define path to original commentary file
# commentary_file = "Data/COMMENTARY_INTL_MATCH/936153_COMMENTARY.csv"
#
# # Extract team information
# print("\n===== TEAM INFORMATION =====")
# print("Extracting team information...")
# if os.path.exists(commentary_file):
#     team_info = extract_teams(commentary_file)
#     if team_info:
#         team1, team2 = get_team_names(team_info)
#         match_summary = get_match_summary(team_info)
#         print(f"\nMatch: {team1} vs {team2}")
#     else:
#         print("Warning: Could not extract team information")
#         team1, team2 = "Team 1", "Team 2"
#         match_summary = "Match information not available"
# else:
#     print(f"Warning: Commentary file {commentary_file} not found")
#     team1, team2 = "Team 1", "Team 2"
#     match_summary = "Match information not available"
#
# # Generate scorecard
# print("\n===== SCORECARD GENERATION =====")
# print("Generating scorecard from commentary data...")
# if os.path.exists(commentary_file):
#     scorecard = generate_scorecard(commentary_file, team_info)
#     if not scorecard:
#         print("Warning: Could not generate scorecard")
# else:
#     print(f"Warning: Commentary file {commentary_file} not found")
#     scorecard = None
#
# # Load the input data
# print("\n===== COMMENTARY ANALYSIS =====")
# print(f"Loading input data from {input_file}")
# input_data = pd.read_csv(input_file)
#
# # Get the commentary and over completion data
# commentary = input_data["Commentary"].astype(str).values
# over_complete = input_data["Over_complete"].values
#
# print(f"Processing {len(commentary)} commentary lines")
#
# # Tokenize the commentary
# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(commentary)
# sequences = tokenizer.texts_to_sequences(commentary)
#
# # Pad the sequences
# max_length = max(len(seq) for seq in sequences)
# padded_sequences = pad_sequences(sequences, maxlen=max_length)
#
# # Initialize variables
# over_num = 1
# total_runs = 0
# total_wickets = 0
# over_stats = {}
#
# print("Predicting runs and wickets for each over...")
#
# # Process each commentary line
# for padded_sequence, complete in zip(padded_sequences, over_complete):
#     # Reshape the padded sequence to match the model's input shape
#     padded_sequence = padded_sequence.reshape(1, max_length)
#
#     # Predict runs and wickets for the commentary line
#     prediction = model.predict(padded_sequence)
#     predicted_runs = int(round(prediction[0][0]))
#     predicted_wickets = int(round(prediction[0][1]))
#
#     # Update total runs and wickets
#     total_runs += predicted_runs
#     total_wickets += predicted_wickets
#
#     # Check if the over is complete
#     if complete:
#         over_stats[over_num] = [total_runs, total_wickets]
#         over_num += 1
#         total_runs = 0
#         total_wickets = 0
#
# # Print match information
# print("\n" + "="*80)
# print_header(f"CRICKET MATCH ANALYSIS: {team1} vs {team2}")
# print(f"{Colors.SKY_BLUE}{"=" * 80}{Colors.RESET}")
# print_match_summary(match_summary)
# print(f"{Colors.SKY_BLUE}{"=" * 80}{Colors.RESET}")
#
# # Display scorecard if available
# if scorecard:
#     print("\n" + "="*80)
#     print_header("MATCH SCORECARD")
#     print(f"{Colors.SKY_BLUE}{"=" * 80}{Colors.RESET}")
#     print(get_full_scorecard(scorecard))
#     print(f"{Colors.SKY_BLUE}{"=" * 80}{Colors.RESET}")
#
# # Sort overs by runs
# sorted_overs_runs = sorted(over_stats.items(), key=lambda x: x[1][0], reverse=True)
#
# # Print the top 10 overs with highest runs
# print("\n" + "="*80)
# print_top_overs_header("TOP 10 OVERS WITH HIGHEST RUNS")
# print(f"{Colors.SKY_BLUE}{"=" * 80}{Colors.RESET}")
# for i, (over, stats) in enumerate(sorted_overs_runs[:10], start=1):
#     print_top_over(i, over, stats[0], stats[1])
#
# # Sort overs by wickets
# sorted_overs_wickets = sorted(over_stats.items(), key=lambda x: x[1][1], reverse=True)
#
# # Print the top 10 overs with highest wickets
# print("\n" + "="*80)
# print_top_overs_header("TOP 10 OVERS WITH HIGHEST WICKETS")
# print(f"{Colors.SKY_BLUE}{"=" * 80}{Colors.RESET}")
# for i, (over, stats) in enumerate(sorted_overs_wickets[:10], start=1):
#     print_top_over(i, over, stats[0], stats[1])
#
# print("\n" + "="*80)
# print_header("ANALYZING COMMENTARY FOR INTERESTING MOMENTS...")
# print(f"{Colors.SKY_BLUE}{"=" * 80}{Colors.RESET}")
#
# text_lines = []
# current_over = []
#
# with open(input_file, 'r') as file:
#     csv_reader = csv.reader(file)
#     next(csv_reader)  # Skip the header row
#
#     for row in csv_reader:
#         commentary = row[0]
#         over_complete = row[1].lower() == 'true'
#
#         current_over.append(commentary)
#
#         if over_complete:
#             text_lines.append(current_over)
#             current_over = []
#
# # If there are any remaining balls in the current over, add it to the text_lines list
# if current_over:
#     text_lines.append(current_over)
#
# print(f"Analyzing {len(text_lines)} overs for highlights...")
# # v_and_s(text_lines, 10)
# analysis_results = v_and_s(text_lines, 10, include_classification=True)
#
# # Add these lines after the v_and_s call
# if isinstance(analysis_results, dict) and 'classifications' in analysis_results:
#     print("\n" + "=" * 80)
#     print_header("COMMENTARY TYPE DISTRIBUTION BY MATCH PHASE")
#     print(f"{Colors.SKY_BLUE}{"=" * 80}{Colors.RESET}")
#
#     # Split match into beginning, middle, and end phases
#     num_overs = len(text_lines)
#     early_phase = int(num_overs * 0.33)
#     middle_phase = int(num_overs * 0.67)
#
#     early_overs = analysis_results['classifications'][:early_phase]
#     middle_overs = analysis_results['classifications'][early_phase:middle_phase]
#     late_overs = analysis_results['classifications'][middle_phase:]
#
#
#     # Count dominant categories in each phase
#     def count_categories(overs):
#         counts = {}
#         for over in overs:
#             category = over['dominant_category']
#             counts[category] = counts.get(category, 0) + 1
#         return counts
#
#
#     early_counts = count_categories(early_overs)
#     middle_counts = count_categories(middle_overs)
#     late_counts = count_categories(late_overs)
#
#     # Print phase analysis
#     print_category_distribution("Early", sorted(early_counts.items(), key=lambda x: x[1], reverse=True))
#     for cat, count in sorted(early_counts.items(), key=lambda x: x[1], reverse=True):
#         if cat != "undetermined":
#             # Original category print - replaced by print_category_distribution
#
#     print_category_distribution("Middle", sorted(middle_counts.items(), key=lambda x: x[1], reverse=True))
#     for cat, count in sorted(middle_counts.items(), key=lambda x: x[1], reverse=True):
#         if cat != "undetermined":
#             # Original category print - replaced by print_category_distribution
#
#     print_category_distribution("Late", sorted(late_counts.items(), key=lambda x: x[1], reverse=True))
#     for cat, count in sorted(late_counts.items(), key=lambda x: x[1], reverse=True):
#         if cat != "undetermined":
#             # Original category print - replaced by print_category_distribution
#
#
# # Add these lines after the v_and_s call
# if isinstance(analysis_results, dict) and 'classifications' in analysis_results:
#     print("\n" + "=" * 80)
#     print("\nGENERATING VISUALIZATIONS...")
#     print(f"{Colors.SKY_BLUE}{"=" * 80}{Colors.RESET}")
#     try:
#         create_visualizations(text_lines, analysis_results['classifications'])
#         print("\nVisualizations have been created in Data/Visualizations/")
#     except Exception as e:
#         print(f"Error creating visualizations: {e}")
# print("\n" + "="*80)
# print(f"\n{Colors.PITCH_GREEN}{Colors.BOLD}Cricket Analysis Complete for {team1} vs {team2}!{Colors.RESET}")
# print(f"{Colors.SKY_BLUE}{"=" * 80}{Colors.RESET}")

import os
import csv
import pickle
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import sys

sys.path.append(".")  # Add current directory to path
from cricket_terminal import Colors, print_header, print_match_summary, print_top_overs_header, print_top_over, \
    print_category_distribution
from Code.vAnds import v_and_s
from commentary_visualizer import create_visualizations
from cricket_analysis.team_extractor import extract_teams, get_team_names, get_match_summary
from cricket_analysis.scorecard_generator import generate_scorecard, get_full_scorecard

print("Cricket Commentary Analysis - Starting...")

# Check if the trained model exists
model_path = "Code/Pickles/trained_model.pkl"
if not os.path.exists(model_path):
    print(f"Error: Trained model not found at {model_path}")
    print("Please run rnn.py first to train the model.")
    exit(1)

# Load the trained model
print(f"Loading trained model from {model_path}")
with open(model_path, "rb") as file:
    model = pickle.load(file)

# Check if the input data exists
input_file = "Data/Input/input.csv"
if not os.path.exists(input_file):
    print(f"Error: Input data not found at {input_file}")
    print("Please run input.py first to prepare the input data.")
    exit(1)

# Define path to original commentary file
commentary_file = "Data/COMMENTARY_INTL_MATCH/936153_COMMENTARY.csv"

# Extract team information
print("\n===== TEAM INFORMATION =====")
print("Extracting team information...")
if os.path.exists(commentary_file):
    team_info = extract_teams(commentary_file)
    if team_info:
        team1, team2 = get_team_names(team_info)
        match_summary = get_match_summary(team_info)
        print(f"\nMatch: {team1} vs {team2}")
    else:
        print("Warning: Could not extract team information")
        team1, team2 = "Team 1", "Team 2"
        match_summary = "Match information not available"
else:
    print(f"Warning: Commentary file {commentary_file} not found")
    team1, team2 = "Team 1", "Team 2"
    match_summary = "Match information not available"

# Generate scorecard
print("\n===== SCORECARD GENERATION =====")
print("Generating scorecard from commentary data...")
if os.path.exists(commentary_file):
    scorecard = generate_scorecard(commentary_file, team_info)
    if not scorecard:
        print("Warning: Could not generate scorecard")
else:
    print(f"Warning: Commentary file {commentary_file} not found")
    scorecard = None

# Load the input data
print("\n===== COMMENTARY ANALYSIS =====")
print(f"Loading input data from {input_file}")
input_data = pd.read_csv(input_file)

# Get the commentary and over completion data
commentary = input_data["Commentary"].astype(str).values
over_complete = input_data["Over_complete"].values

print(f"Processing {len(commentary)} commentary lines")

# Tokenize the commentary
tokenizer = Tokenizer()
tokenizer.fit_on_texts(commentary)
sequences = tokenizer.texts_to_sequences(commentary)

# Pad the sequences
max_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_length)

# Initialize variables
over_num = 1
total_runs = 0
total_wickets = 0
over_stats = {}

print("Predicting runs and wickets for each over...")

# Process each commentary line
for padded_sequence, complete in zip(padded_sequences, over_complete):
    # Reshape the padded sequence to match the model's input shape
    padded_sequence = padded_sequence.reshape(1, max_length)

    # Predict runs and wickets for the commentary line
    prediction = model.predict(padded_sequence)
    predicted_runs = int(round(prediction[0][0]))
    predicted_wickets = int(round(prediction[0][1]))

    # Update total runs and wickets
    total_runs += predicted_runs
    total_wickets += predicted_wickets

    # Check if the over is complete
    if complete:
        over_stats[over_num] = [total_runs, total_wickets]
        over_num += 1
        total_runs = 0
        total_wickets = 0

# Print match information
print("\n" + "=" * 80)
print_header(f"CRICKET MATCH ANALYSIS: {team1} vs {team2}")
print(f"{Colors.SKY_BLUE}{'=' * 80}{Colors.RESET}")
print_match_summary(match_summary)
print(f"{Colors.SKY_BLUE}{'=' * 80}{Colors.RESET}")

# Display scorecard if available
if scorecard:
    print("\n" + "=" * 80)
    print_header("MATCH SCORECARD")
    print(f"{Colors.SKY_BLUE}{'=' * 80}{Colors.RESET}")
    print(get_full_scorecard(scorecard))
    print(f"{Colors.SKY_BLUE}{'=' * 80}{Colors.RESET}")

# Sort overs by runs
sorted_overs_runs = sorted(over_stats.items(), key=lambda x: x[1][0], reverse=True)

# Print the top 10 overs with highest runs
print("\n" + "=" * 80)
print_top_overs_header("TOP 10 OVERS WITH HIGHEST RUNS")
print(f"{Colors.SKY_BLUE}{'=' * 80}{Colors.RESET}")
for i, (over, stats) in enumerate(sorted_overs_runs[:10], start=1):
    print_top_over(i, over, stats[0], stats[1])

# Sort overs by wickets
sorted_overs_wickets = sorted(over_stats.items(), key=lambda x: x[1][1], reverse=True)

# Print the top 10 overs with highest wickets
print("\n" + "=" * 80)
print_top_overs_header("TOP 10 OVERS WITH HIGHEST WICKETS")
print(f"{Colors.SKY_BLUE}{'=' * 80}{Colors.RESET}")
for i, (over, stats) in enumerate(sorted_overs_wickets[:10], start=1):
    print_top_over(i, over, stats[0], stats[1])

print("\n" + "=" * 80)
print_header("ANALYZING COMMENTARY FOR INTERESTING MOMENTS...")
print(f"{Colors.SKY_BLUE}{'=' * 80}{Colors.RESET}")

text_lines = []
current_over = []

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

print(f"Analyzing {len(text_lines)} overs for highlights...")
# v_and_s(text_lines, 10)
analysis_results = v_and_s(text_lines, 10, include_classification=True)

# Add these lines after the v_and_s call
if isinstance(analysis_results, dict) and 'classifications' in analysis_results:
    print("\n" + "=" * 80)
    print_header("COMMENTARY TYPE DISTRIBUTION BY MATCH PHASE")
    print(f"{Colors.SKY_BLUE}{'=' * 80}{Colors.RESET}")

    # Split match into beginning, middle, and end phases
    num_overs = len(text_lines)
    early_phase = int(num_overs * 0.33)
    middle_phase = int(num_overs * 0.67)

    early_overs = analysis_results['classifications'][:early_phase]
    middle_overs = analysis_results['classifications'][early_phase:middle_phase]
    late_overs = analysis_results['classifications'][middle_phase:]


    # Count dominant categories in each phase
    def count_categories(overs):
        counts = {}
        for over in overs:
            category = over['dominant_category']
            counts[category] = counts.get(category, 0) + 1
        return counts


    early_counts = count_categories(early_overs)
    middle_counts = count_categories(middle_overs)
    late_counts = count_categories(late_overs)

    # Prepare data for category distribution (including percentages)
    early_with_pct = [(cat, (count, count / len(early_overs) * 100))
                      for cat, count in sorted(early_counts.items(), key=lambda x: x[1], reverse=True)]
    middle_with_pct = [(cat, (count, count / len(middle_overs) * 100))
                       for cat, count in sorted(middle_counts.items(), key=lambda x: x[1], reverse=True)]
    late_with_pct = [(cat, (count, count / len(late_overs) * 100))
                     for cat, count in sorted(late_counts.items(), key=lambda x: x[1], reverse=True)]

    # Print phase analysis
    print_category_distribution("Early", early_with_pct)
    print_category_distribution("Middle", middle_with_pct)
    print_category_distribution("Late", late_with_pct)

# Add these lines after the phase analysis
if isinstance(analysis_results, dict) and 'classifications' in analysis_results:
    print("\n" + "=" * 80)
    print_header("GENERATING VISUALIZATIONS")
    print(f"{Colors.SKY_BLUE}{'=' * 80}{Colors.RESET}")
    try:
        create_visualizations(text_lines, analysis_results['classifications'])
        print(f"\n{Colors.PITCH_GREEN}Visualizations have been created in Data/Visualizations/{Colors.RESET}")
    except Exception as e:
        print(f"Error creating visualizations: {e}")

print("\n" + "=" * 80)
print(f"\n{Colors.PITCH_GREEN}{Colors.BOLD}Cricket Analysis Complete for {team1} vs {team2}!{Colors.RESET}")
print(f"{Colors.SKY_BLUE}{'=' * 80}{Colors.RESET}")
import pandas as pd
import os


def extract_teams(csv_file_path):
    """
    Extract team information from cricket commentary data

    Args:
        csv_file_path (str): Path to the commentary CSV file

    Returns:
        dict: Dictionary containing team information
    """
    # Check if file exists
    if not os.path.exists(csv_file_path):
        print(f"Error: File {csv_file_path} not found")
        return None

    # Read the CSV file
    try:
        df = pd.read_csv(csv_file_path)

        # Check if required columns exist
        required_columns = ['Batting_Team_name', 'Bowling_Team_name', 'Innings']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return None

        # Extract unique team names
        team1 = df['Batting_Team_name'].iloc[0]
        team2 = df['Bowling_Team_name'].iloc[0]

        # Initialize team stats dictionary
        team_info = {
            'team1': {
                'name': team1,
                'id': df['Batting_Team_id'].iloc[0],
                'innings': []
            },
            'team2': {
                'name': team2,
                'id': df['Bowling_Team_id'].iloc[0],
                'innings': []
            }
        }

        # Find unique innings in the match
        innings_list = df['Innings'].unique()

        # Process each innings
        for innings in innings_list:
            innings_data = df[df['Innings'] == innings].iloc[0]
            batting_team = innings_data['Batting_Team_name']

            # Determine which team is batting in this innings
            if batting_team == team1:
                team_info['team1']['innings'].append(innings)
            else:
                team_info['team2']['innings'].append(innings)

        # Add match summary
        team_info['match_summary'] = {
            'total_innings': len(innings_list),
            'teams': [team1, team2]
        }

        # Calculate total runs per team
        team1_runs = 0
        team2_runs = 0

        for innings in innings_list:
            innings_df = df[df['Innings'] == innings]
            last_row = innings_df.iloc[-1]
            innings_runs = last_row['Innings_runs']

            if last_row['Batting_Team_name'] == team1:
                team1_runs += innings_runs
            else:
                team2_runs += innings_runs

        team_info['team1']['total_runs'] = team1_runs
        team_info['team2']['total_runs'] = team2_runs

        print(f"Successfully extracted team information for match between {team1} and {team2}")
        return team_info

    except Exception as e:
        print(f"Error processing file: {e}")
        return None


def get_team_names(team_info):
    """
    Get team names from team information

    Args:
        team_info (dict): Team information dictionary

    Returns:
        tuple: Tuple containing team names (team1, team2)
    """
    if not team_info:
        return None, None

    return team_info['team1']['name'], team_info['team2']['name']


def get_match_summary(team_info):
    """
    Generate a match summary from team information

    Args:
        team_info (dict): Team information dictionary

    Returns:
        str: Match summary text
    """
    if not team_info:
        return "No match information available"

    team1 = team_info['team1']['name']
    team2 = team_info['team2']['name']
    team1_runs = team_info['team1']['total_runs']
    team2_runs = team_info['team2']['total_runs']

    summary = f"Match Summary: {team1} vs {team2}\n"
    summary += f"{team1}: {team1_runs} runs\n"
    summary += f"{team2}: {team2_runs} runs\n"

    # Determine match result
    if team1_runs > team2_runs:
        summary += f"{team1} wins by {team1_runs - team2_runs} runs"
    elif team2_runs > team1_runs:
        summary += f"{team2} wins by {team2_runs - team1_runs} runs"
    else:
        summary += "Match ended in a tie"

    return summary


# Example usage of these functions
if __name__ == "__main__":
    # Example usage
    csv_file = "Data/COMMENTARY_INTL_MATCH/936153_COMMENTARY.csv"
    team_info = extract_teams(csv_file)

    if team_info:
        team1, team2 = get_team_names(team_info)
        print(f"Teams: {team1} vs {team2}")
        print(get_match_summary(team_info))
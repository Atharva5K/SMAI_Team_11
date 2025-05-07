import os
import numpy as np
import pandas as pd
from cricket_terminal import Colors  # Add cricket-themed colors


def generate_scorecard(csv_file_path, team_info=None):
    """
    Generate detailed scorecards for both teams from cricket commentary data

    Args:
        csv_file_path (str): Path to the commentary CSV file
        team_info (dict, optional): Team information dictionary

    Returns:
        dict: Dictionary containing scorecard information
    """
    # Check if file exists
    if not os.path.exists(csv_file_path):
        print(f"Error: File {csv_file_path} not found")
        return None

    # Read the CSV file
    try:
        df = pd.read_csv(csv_file_path)

        # Initialize scorecard dictionary
        scorecard = {
            'batting': {},
            'bowling': {},
            'match_stats': {},
            'top_performers': {}
        }

        # Extract team names if not provided
        if team_info is None:
            team1 = df['Batting_Team_name'].iloc[0]
            team2 = df['Bowling_Team_name'].iloc[0]
        else:
            team1 = team_info['team1']['name']
            team2 = team_info['team2']['name']

        scorecard['teams'] = {'team1': team1, 'team2': team2}

        # Initialize team statistics
        for team in [team1, team2]:
            scorecard['batting'][team] = {}
            scorecard['bowling'][team] = {}

        # Unique innings in the match
        innings_list = df['Innings'].unique()

        # Process each innings
        for innings in innings_list:
            innings_df = df[df['Innings'] == innings]

            # Get team names for this innings
            batting_team = innings_df['Batting_Team_name'].iloc[0]
            bowling_team = innings_df['Bowling_Team_name'].iloc[0]

            # Get unique batsmen in this innings
            batsmen = set()
            for _, row in innings_df.iterrows():
                if pd.notna(row['Batsman_name']):
                    batsmen.add(row['Batsman_name'])
                if pd.notna(row['Other_Batsman_name']):
                    batsmen.add(row['Other_Batsman_name'])

            # Calculate batsmen statistics
            for batsman in batsmen:
                # Find the last row where this batsman appears to get final stats
                batsman_rows = innings_df[(innings_df['Batsman_name'] == batsman) |
                                          (innings_df['Other_Batsman_name'] == batsman)]

                if batsman_rows.empty:
                    continue

                last_row = batsman_rows.iloc[-1]

                # Determine which batsman column to use
                if last_row['Batsman_name'] == batsman:
                    runs = last_row['Batsman_runs']
                    balls = last_row['Batsman_balls_faced']
                    fours = last_row['Batsman_four']
                    sixes = last_row['Batsman_sixes']
                else:
                    runs = last_row['Other_Batsman_runs']
                    balls = last_row['Other_Batsman_balls_faced']
                    fours = last_row['Other_Batsman_four']
                    sixes = last_row['Other_Batsman_sixes']

                # Calculate strike rate
                strike_rate = (runs / balls * 100) if balls > 0 else 0

                # Store batsman statistics
                if batsman not in scorecard['batting'][batting_team]:
                    scorecard['batting'][batting_team][batsman] = {
                        'runs': int(runs),
                        'balls': int(balls),
                        'fours': int(fours),
                        'sixes': int(sixes),
                        'strike_rate': round(strike_rate, 2),
                        'status': 'not out'  # Default status
                    }

                # Check for dismissal
                dismissal_rows = innings_df[innings_df['Dismissal_is_true'] == True]
                for _, row in dismissal_rows.iterrows():
                    if pd.notna(row['Dismissal_text']) and batsman in str(row['Dismissal_text']):
                        scorecard['batting'][batting_team][batsman]['status'] = row['Dismissal_type']

            # Get unique bowlers in this innings
            bowlers = set()
            for _, row in innings_df.iterrows():
                if pd.notna(row['Bowler_name']):
                    bowlers.add(row['Bowler_name'])

            # Calculate bowler statistics
            for bowler in bowlers:
                # Find the last row where this bowler appears to get final stats
                bowler_rows = innings_df[innings_df['Bowler_name'] == bowler]

                if bowler_rows.empty:
                    continue

                last_row = bowler_rows.iloc[-1]

                overs = last_row['Bowler_over']
                maidens = last_row['Bowler_maiden']
                runs = last_row['Bowler_conceded']
                wickets = last_row['Bowler_wickets']
                balls = last_row['Bowler_balls']

                # Calculate economy rate
                economy_rate = (runs / (balls / 6)) if balls > 0 else 0

                # Store bowler statistics
                if bowler not in scorecard['bowling'][bowling_team]:
                    scorecard['bowling'][bowling_team][bowler] = {
                        'overs': float(overs),
                        'maidens': int(maidens),
                        'runs': int(runs),
                        'wickets': int(wickets),
                        'economy': round(economy_rate, 2)
                    }

        # Calculate team totals and extras
        for innings in innings_list:
            innings_df = df[df['Innings'] == innings]
            last_row = innings_df.iloc[-1]

            batting_team = last_row['Batting_Team_name']

            total_runs = last_row['Innings_runs']
            total_wickets = last_row['Innings_wickets']
            extras = (last_row['Innings_wides'] +
                      last_row['Innings_no_balls'] +
                      last_row['Innings_byes'] +
                      last_row['Innings_leg_byes'])

            # Add team total to scorecard
            if 'totals' not in scorecard['batting'][batting_team]:
                scorecard['batting'][batting_team]['totals'] = {
                    'runs': int(total_runs),
                    'wickets': int(total_wickets),
                    'extras': int(extras)
                }
            else:
                # Add to existing totals for multiple innings
                scorecard['batting'][batting_team]['totals']['runs'] += int(total_runs)
                scorecard['batting'][batting_team]['totals']['wickets'] += int(total_wickets)
                scorecard['batting'][batting_team]['totals']['extras'] += int(extras)

        # Identify top performers
        # Top batsman (most runs)
        top_batsman = {'name': None, 'team': None, 'runs': 0}
        for team, batsmen in scorecard['batting'].items():
            for batsman, stats in batsmen.items():
                if batsman != 'totals' and stats['runs'] > top_batsman['runs']:
                    top_batsman['name'] = batsman
                    top_batsman['team'] = team
                    top_batsman['runs'] = stats['runs']

        # Top bowler (most wickets, then best economy)
        top_bowler = {'name': None, 'team': None, 'wickets': 0, 'economy': float('inf')}
        for team, bowlers in scorecard['bowling'].items():
            for bowler, stats in bowlers.items():
                if stats['wickets'] > top_bowler['wickets'] or (
                        stats['wickets'] == top_bowler['wickets'] and
                        stats['economy'] < top_bowler['economy']):
                    top_bowler['name'] = bowler
                    top_bowler['team'] = team
                    top_bowler['wickets'] = stats['wickets']
                    top_bowler['economy'] = stats['economy']

        # Most economic bowler (at least 2 overs)
        eco_bowler = {'name': None, 'team': None, 'economy': float('inf'), 'overs': 0}
        for team, bowlers in scorecard['bowling'].items():
            for bowler, stats in bowlers.items():
                if stats['overs'] >= 2 and stats['economy'] < eco_bowler['economy']:
                    eco_bowler['name'] = bowler
                    eco_bowler['team'] = team
                    eco_bowler['economy'] = stats['economy']
                    eco_bowler['overs'] = stats['overs']

        # Store top performers
        scorecard['top_performers']['top_batsman'] = top_batsman
        scorecard['top_performers']['top_bowler'] = top_bowler
        scorecard['top_performers']['most_economic_bowler'] = eco_bowler

        print(f"Successfully generated scorecard for match between {team1} and {team2}")
        return scorecard

    except Exception as e:
        print(f"Error processing scorecard: {e}")
        return None


def format_batting_scorecard(scorecard, team):
    """
    Format batting scorecard for display with cricket-themed colors

    Args:
        scorecard (dict): Scorecard dictionary
        team (str): Team name

    Returns:
        str: Formatted batting scorecard
    """
    if not scorecard or 'batting' not in scorecard or team not in scorecard['batting']:
        return f"No batting data available for {team}"

    batting_data = scorecard['batting'][team]

    # Skip if only totals are present
    if len(batting_data) <= 1:
        return f"No batting data available for {team}"

    formatted = f"\n{Colors.TEAM_BLUE}{Colors.BOLD}{team} - BATTING SCORECARD{Colors.RESET}\n"
    formatted += f"{Colors.SKY_BLUE}{'=' * 50}{Colors.RESET}\n"
    formatted += f"{Colors.STATS_CYAN}{'BATSMAN':<30} {'RUNS':<5} {'BALLS':<5} {'4s':<3} {'6s':<3} {'SR':<7}{Colors.RESET}\n"
    formatted += f"{Colors.SKY_BLUE}{'-' * 50}{Colors.RESET}\n"

    for batsman, stats in batting_data.items():
        if batsman == 'totals':
            continue

        formatted += f"{batsman:<30} {Colors.SCORE_GOLD}{stats['runs']:<5}{Colors.RESET} {stats['balls']:<5} "
        formatted += f"{Colors.PITCH_GREEN}{stats['fours']:<3}{Colors.RESET} {Colors.SUN_YELLOW}{stats['sixes']:<3}{Colors.RESET} {stats['strike_rate']:<7.2f}\n"

    # Add totals
    if 'totals' in batting_data:
        totals = batting_data['totals']
        formatted += f"{Colors.SKY_BLUE}{'-' * 50}{Colors.RESET}\n"
        formatted += f"TOTAL: {Colors.SCORE_GOLD}{Colors.BOLD}{totals['runs']}{Colors.RESET}/{Colors.BALL_RED}{totals['wickets']}{Colors.RESET} (Extras: {totals['extras']})\n"

    return formatted


def format_bowling_scorecard(scorecard, team):
    """
    Format bowling scorecard for display with cricket-themed colors

    Args:
        scorecard (dict): Scorecard dictionary
        team (str): Team name

    Returns:
        str: Formatted bowling scorecard
    """
    if not scorecard or 'bowling' not in scorecard or team not in scorecard['bowling']:
        return f"No bowling data available for {team}"

    bowling_data = scorecard['bowling'][team]

    # Skip if no bowlers
    if len(bowling_data) == 0:
        return f"No bowling data available for {team}"

    formatted = f"\n{Colors.TEAM_RED}{Colors.BOLD}{team} - BOWLING SCORECARD{Colors.RESET}\n"
    formatted += f"{Colors.SKY_BLUE}{'=' * 50}{Colors.RESET}\n"
    formatted += f"{Colors.STATS_CYAN}{'BOWLER':<30} {'OVERS':<6} {'MAIDENS':<8} {'RUNS':<5} {'WICKETS':<8} {'ECONOMY':<8}{Colors.RESET}\n"
    formatted += f"{Colors.SKY_BLUE}{'-' * 50}{Colors.RESET}\n"

    for bowler, stats in bowling_data.items():
        formatted += f"{bowler:<30} {stats['overs']:<6.1f} {Colors.PITCH_GREEN}{stats['maidens']:<8}{Colors.RESET} "
        formatted += f"{stats['runs']:<5} {Colors.BALL_RED}{stats['wickets']:<8}{Colors.RESET} {stats['economy']:<8.2f}\n"

    return formatted


def format_top_performers(scorecard):
    """
    Format top performers for display with cricket-themed colors

    Args:
        scorecard (dict): Scorecard dictionary

    Returns:
        str: Formatted top performers
    """
    if not scorecard or 'top_performers' not in scorecard:
        return "No top performer data available"

    top_performers = scorecard['top_performers']

    formatted = f"\n{Colors.SKY_BLUE}{Colors.BOLD}TOP PERFORMERS{Colors.RESET}\n"
    formatted += f"{Colors.SKY_BLUE}{'=' * 50}{Colors.RESET}\n"

    # Top batsman
    if 'top_batsman' in top_performers and top_performers['top_batsman']['name']:
        top_bat = top_performers['top_batsman']
        formatted += f"{Colors.STATS_CYAN}Top Batsman:{Colors.RESET} {Colors.TEAM_BLUE}{top_bat['name']}{Colors.RESET} ({top_bat['team']}) - {Colors.SCORE_GOLD}{top_bat['runs']}{Colors.RESET} runs\n"

    # Top bowler
    if 'top_bowler' in top_performers and top_performers['top_bowler']['name']:
        top_bowl = top_performers['top_bowler']
        formatted += f"{Colors.STATS_CYAN}Top Bowler:{Colors.RESET} {Colors.TEAM_RED}{top_bowl['name']}{Colors.RESET} ({top_bowl['team']}) - {Colors.BALL_RED}{top_bowl['wickets']}{Colors.RESET} wickets, economy {top_bowl['economy']:.2f}\n"

    # Most economic bowler
    if 'most_economic_bowler' in top_performers and top_performers['most_economic_bowler']['name']:
        eco_bowl = top_performers['most_economic_bowler']
        formatted += f"{Colors.STATS_CYAN}Most Economic Bowler:{Colors.RESET} {Colors.TEAM_RED}{eco_bowl['name']}{Colors.RESET} ({eco_bowl['team']}) - {eco_bowl['economy']:.2f} (min. 2 overs)\n"

    return formatted


def get_full_scorecard(scorecard):
    """
    Get complete formatted scorecard with cricket-themed colors

    Args:
        scorecard (dict): Scorecard dictionary

    Returns:
        str: Complete formatted scorecard
    """
    if not scorecard:
        return "No scorecard data available"

    team1 = scorecard['teams']['team1']
    team2 = scorecard['teams']['team2']

    formatted = ""
    formatted += f"{Colors.SKY_BLUE}{'=' * 50}{Colors.RESET}\n"

    # Add batting scorecards
    formatted += format_batting_scorecard(scorecard, team1)
    formatted += "\n"
    formatted += format_batting_scorecard(scorecard, team2)

    # Add bowling scorecards
    formatted += "\n"
    formatted += format_bowling_scorecard(scorecard, team1)
    formatted += "\n"
    formatted += format_bowling_scorecard(scorecard, team2)

    # Add top performers
    formatted += "\n"
    formatted += format_top_performers(scorecard)

    return formatted


# Example usage of these functions
if __name__ == "__main__":
    # Example usage
    csv_file = "Data/COMMENTARY_INTL_MATCH/936153_COMMENTARY.csv"
    scorecard = generate_scorecard(csv_file)

    if scorecard:
        print(get_full_scorecard(scorecard))
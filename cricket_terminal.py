"""
Cricket Terminal Styling Module

This module enhances terminal output with cricket-themed colors and styling
to make the analysis results more visually appealing.
"""

# ANSI color codes for terminal styling
class Colors:
    # Cricket theme colors
    PITCH_GREEN = '\033[38;5;34m'  # Bright green for pitch-related info
    BALL_RED = '\033[38;5;160m'    # Cricket ball red for wickets
    SKY_BLUE = '\033[38;5;39m'     # Sky blue for headers
    SUN_YELLOW = '\033[38;5;220m'  # Sunny yellow for highlights
    TEAM_BLUE = '\033[38;5;27m'    # Team blue for team info
    TEAM_RED = '\033[38;5;196m'    # Team red for opposing team
    STATS_CYAN = '\033[38;5;51m'   # Cyan for statistics
    SCORE_GOLD = '\033[38;5;214m'  # Gold for scores

    # Styling options
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    # Reset code (always use at the end of colored text)
    RESET = '\033[0m'

def print_header(text):
    """Print a section header with cricket-styled formatting"""
    border = "=" * 80
    print(f"\n{Colors.SKY_BLUE}{Colors.BOLD}{border}{Colors.RESET}")
    print(f"{Colors.SKY_BLUE}{Colors.BOLD}{text.center(80)}{Colors.RESET}")
    print(f"{Colors.SKY_BLUE}{Colors.BOLD}{border}{Colors.RESET}")

def print_subheader(text):
    """Print a subsection header with cricket-styled formatting"""
    print(f"\n{Colors.TEAM_BLUE}{Colors.BOLD}{text}{Colors.RESET}")
    print(f"{Colors.TEAM_BLUE}{'-' * 50}{Colors.RESET}")

def print_team_info(team1, team2):
    """Print team information with cricket-styled formatting"""
    vs_text = f"{team1} vs {team2}"
    print(f"\n{Colors.TEAM_BLUE}{Colors.BOLD}{team1}{Colors.RESET} vs {Colors.TEAM_RED}{Colors.BOLD}{team2}{Colors.RESET}")

def print_match_summary(summary_text):
    """Print match summary with cricket-styled formatting"""
    lines = summary_text.split('\n')

    for line in lines:
        if "wins" in line:
            # Highlight the winning team
            parts = line.split("wins")
            print(f"{Colors.SCORE_GOLD}{Colors.BOLD}{parts[0]}{Colors.RESET}wins{parts[1]}")
        elif ":" in line:
            # Color team names and scores differently
            parts = line.split(":")
            print(f"{Colors.TEAM_BLUE}{parts[0]}:{Colors.RESET}{Colors.SCORE_GOLD}{parts[1]}{Colors.RESET}")
        else:
            print(line)

def print_scorecard_header(team_name):
    """Print scorecard header with cricket-styled formatting"""
    print(f"\n{Colors.TEAM_BLUE}{Colors.BOLD}{team_name} - BATTING SCORECARD{Colors.RESET}")
    print(f"{Colors.SKY_BLUE}{'=' * 50}{Colors.RESET}")
    print(f"{Colors.STATS_CYAN}{'BATSMAN':<30} {'RUNS':<5} {'BALLS':<5} {'4s':<3} {'6s':<3} {'SR':<7}{Colors.RESET}")
    print(f"{Colors.SKY_BLUE}{'-' * 50}{Colors.RESET}")

def print_batsman_stats(batsman, stats):
    """Print batsman statistics with cricket-styled formatting"""
    print(f"{batsman:<30} {Colors.SCORE_GOLD}{stats['runs']:<5}{Colors.RESET} "
          f"{stats['balls']:<5} {Colors.PITCH_GREEN}{stats['fours']:<3}{Colors.RESET} "
          f"{Colors.SUN_YELLOW}{stats['sixes']:<3}{Colors.RESET} {stats['strike_rate']:<7.2f}")

def print_bowling_header(team_name):
    """Print bowling header with cricket-styled formatting"""
    print(f"\n{Colors.TEAM_RED}{Colors.BOLD}{team_name} - BOWLING SCORECARD{Colors.RESET}")
    print(f"{Colors.SKY_BLUE}{'=' * 50}{Colors.RESET}")
    print(f"{Colors.STATS_CYAN}{'BOWLER':<30} {'OVERS':<6} {'MAIDENS':<8} {'RUNS':<5} {'WICKETS':<8} {'ECONOMY':<8}{Colors.RESET}")
    print(f"{Colors.SKY_BLUE}{'-' * 50}{Colors.RESET}")

def print_bowler_stats(bowler, stats):
    """Print bowler statistics with cricket-styled formatting"""
    print(f"{bowler:<30} {stats['overs']:<6.1f} {Colors.PITCH_GREEN}{stats['maidens']:<8}{Colors.RESET} "
          f"{stats['runs']:<5} {Colors.BALL_RED}{stats['wickets']:<8}{Colors.RESET} {stats['economy']:<8.2f}")

def print_total_score(runs, wickets, extras):
    """Print total score with cricket-styled formatting"""
    print(f"{Colors.SKY_BLUE}{'-' * 50}{Colors.RESET}")
    print(f"TOTAL: {Colors.SCORE_GOLD}{Colors.BOLD}{runs}{Colors.RESET}/{Colors.BALL_RED}{wickets}{Colors.RESET} "
          f"(Extras: {extras})")

def print_top_overs_header(title):
    """Print top overs header with cricket-styled formatting"""
    print_header(title)

def print_top_over(rank, over, runs, wickets):
    """Print top over information with cricket-styled formatting"""
    print(f"{Colors.STATS_CYAN}{rank}. Over {over}:{Colors.RESET} "
          f"Runs = {Colors.SCORE_GOLD}{runs}{Colors.RESET}, "
          f"Wickets = {Colors.BALL_RED}{wickets}{Colors.RESET}")

def print_highlight(over, summary):
    """Print highlight information with cricket-styled formatting"""
    print(f"\n{Colors.SUN_YELLOW}{Colors.BOLD}Over {over}:{Colors.RESET} {summary}")

def print_completion():
    """Print completion message with cricket-styled formatting"""
    print(f"\n{Colors.SKY_BLUE}{Colors.BOLD}{'=' * 80}{Colors.RESET}")
    print(f"{Colors.PITCH_GREEN}{Colors.BOLD}Cricket Analysis Complete!{Colors.RESET}")
    print(f"{Colors.SKY_BLUE}{Colors.BOLD}{'=' * 80}{Colors.RESET}")

def print_phase_header(phase_name):
    """Print phase header with cricket-styled formatting"""
    print(f"\n{Colors.TEAM_BLUE}{Colors.BOLD}{phase_name} Match Commentary Types:{Colors.RESET}")

def print_category_distribution(phase, categories_with_counts):
    """Print category distribution with cricket-styled formatting"""
    print_phase_header(phase)

    for cat, count_data in categories_with_counts:
        if cat != "undetermined":
            # Choose color based on category
            color = Colors.STATS_CYAN
            if "praise" in cat.lower():
                color = Colors.PITCH_GREEN
            elif "criticism" in cat.lower():
                color = Colors.BALL_RED
            elif "technical" in cat.lower():
                color = Colors.SKY_BLUE
            elif "strategic" in cat.lower():
                color = Colors.TEAM_BLUE
            elif "audience" in cat.lower():
                color = Colors.SUN_YELLOW

            # Unpack count and percentage if provided as tuple, otherwise calculate percentage
            if isinstance(count_data, tuple):
                count, percentage = count_data
            else:
                count = count_data
                percentage = 0  # Default if we don't have total to calculate percentage

            print(f"  {color}{cat.upper()}{Colors.RESET}: {count} overs ({percentage:.1f}%)")

# Here's a helper function to modify print statements in final.py to use our styled versions
def handle_print_statement(line):
    """
    Transform a regular print statement into a cricket-styled one
    This function is for manual modification of final.py if needed
    """
    # Example transformations - add more as needed
    if line.strip().startswith('print("=" * 80'):
        return '    print_header("YOUR TITLE HERE")'

    return line

# Example of how to use this module in final.py
if __name__ == "__main__":
    # Simple demo of the formatting
    print_header("CRICKET MATCH ANALYSIS: India vs Australia")

    print_match_summary("Match Summary: India vs Australia\nIndia: 287 runs\nAustralia: 269 runs\nIndia wins by 18 runs")

    print_scorecard_header("India")
    print_batsman_stats("Virat Kohli", {"runs": 82, "balls": 68, "fours": 7, "sixes": 3, "strike_rate": 120.59})
    print_batsman_stats("Rohit Sharma", {"runs": 56, "balls": 42, "fours": 4, "sixes": 2, "strike_rate": 133.33})
    print_total_score(287, 5, 12)

    print_top_overs_header("TOP 10 OVERS WITH HIGHEST RUNS")
    print_top_over(1, 16, 22, 0)
    print_top_over(2, 18, 18, 1)

    print_completion()
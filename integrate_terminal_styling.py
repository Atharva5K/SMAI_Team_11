"""
Cricket Terminal Styling Integration

This script modifies final.py to use the cricket terminal styling module
for more appealing colored output.
"""

import os
import sys
import importlib.util
import shutil


def create_backup(file_path):
    """Create a backup of the original file"""
    backup_path = file_path + '.terminal.bak'
    if not os.path.exists(backup_path):
        shutil.copy2(file_path, backup_path)
        print(f"Created backup of {file_path} at {backup_path}")
    return backup_path


def modify_final_py():
    """Modify the final.py file to include terminal styling"""
    final_py_path = "final.py"

    # Create backup
    create_backup(final_py_path)

    # Read the original file
    with open(final_py_path, 'r') as file:
        content = file.read()

    # Add import statement for the cricket terminal module
    if "from cricket_terminal import" not in content:
        import_pattern = "import sys\nsys.path.append(\".\")"
        replacement = "import sys\nsys.path.append(\".\")\nfrom cricket_terminal import *  # Import cricket-themed terminal styling"
        content = content.replace(import_pattern, replacement)

    # Replace print statements with styled versions
    replacements = [
        # Main headers
        (r'print\("\\n" \+ "="*80\)', r'print_header'),
        (r'print\(f"\\nCRICKET MATCH ANALYSIS: {team1} vs {team2}"\)',
         r'print_header(f"CRICKET MATCH ANALYSIS: {team1} vs {team2}")'),
        (r'print\("=" \* 80\)', r'print(f"{Colors.SKY_BLUE}{"=" * 80}{Colors.RESET}")'),

        # Match summary
        (r'print\(match_summary\)', r'print_match_summary(match_summary)'),

        # Scorecard
        (r'print\("\\nMATCH SCORECARD"\)', r'print_header("MATCH SCORECARD")'),
        (r'print\(get_full_scorecard\(scorecard\)\)', r'print(get_full_scorecard(scorecard))'),

        # Top overs sections
        (r'print\("\\nTOP 10 OVERS WITH HIGHEST RUNS:"\)', r'print_top_overs_header("TOP 10 OVERS WITH HIGHEST RUNS")'),
        (r'print\("\\nTOP 10 OVERS WITH HIGHEST WICKETS:"\)',
         r'print_top_overs_header("TOP 10 OVERS WITH HIGHEST WICKETS")'),

        # Top over lines
        (r'print\(f"{i}\. Over {over}: Runs = {stats\[0\]}, Wickets = {stats\[1\]}"\)',
         r'print_top_over(i, over, stats[0], stats[1])'),

        # Analysis sections
        (r'print\("\\nANALYZING COMMENTARY FOR INTERESTING MOMENTS\.\.\."\)',
         r'print_header("ANALYZING COMMENTARY FOR INTERESTING MOMENTS...")'),
        (r'print\("\\nCOMMENTARY TYPE DISTRIBUTION BY MATCH PHASE"\)',
         r'print_header("COMMENTARY TYPE DISTRIBUTION BY MATCH PHASE")'),

        # Phase analysis
        (r'print\("\\nEarly Match Commentary Types:"\)',
         r'print_category_distribution("Early", sorted(early_counts.items(), key=lambda x: x[1], reverse=True))'),
        (r'print\("\\nMiddle Match Commentary Types:"\)',
         r'print_category_distribution("Middle", sorted(middle_counts.items(), key=lambda x: x[1], reverse=True))'),
        (r'print\("\\nLate Match Commentary Types:"\)',
         r'print_category_distribution("Late", sorted(late_counts.items(), key=lambda x: x[1], reverse=True))'),

        # Category prints
        (r'print\(f"  {cat.upper\(\)}: {count} overs \({count / len\(early_overs\) \* 100:.1f}%\)"\)',
         r'# Original category print - replaced by print_category_distribution'),
        (r'print\(f"  {cat.upper\(\)}: {count} overs \({count / len\(middle_overs\) \* 100:.1f}%\)"\)',
         r'# Original category print - replaced by print_category_distribution'),
        (r'print\(f"  {cat.upper\(\)}: {count} overs \({count / len\(late_overs\) \* 100:.1f}%\)"\)',
         r'# Original category print - replaced by print_category_distribution'),

        # Completion
        (r'print\(f"\\nCricket Analysis Complete for {team1} vs {team2}!"\)',
         r'print(f"\\n{Colors.PITCH_GREEN}{Colors.BOLD}Cricket Analysis Complete for {team1} vs {team2}!{Colors.RESET}")'),
    ]

    # Apply replacements
    for old, new in replacements:
        import re
        content = re.sub(old, new, content)

    # Create the category_distribution function modifications
    category_print_sections = [
        "for cat, count in sorted(early_counts.items(), key=lambda x: x[1], reverse=True):",
        "for cat, count in sorted(middle_counts.items(), key=lambda x: x[1], reverse=True):",
        "for cat, count in sorted(late_counts.items(), key=lambda x: x[1], reverse=True):"
    ]

    for section in category_print_sections:
        section_with_print = section + "\n        if cat != \"undetermined\":\n            print(f\"  {cat.upper()}: {count} overs"
        section_with_comment = section + "\n        if cat != \"undetermined\":\n            # Print handled by print_category_distribution"
        content = content.replace(section_with_print, section_with_comment)

    # Write the modified content back to the file
    with open(final_py_path, 'w') as file:
        file.write(content)

    print(f"Successfully modified {final_py_path} to include cricket-themed terminal styling")
    return True


def main():
    """Main function to integrate terminal styling"""
    print("=" * 80)
    print("Cricket Terminal Styling Integration")
    print("=" * 80)

    # Check if the cricket_terminal.py module exists
    if not os.path.exists("cricket_terminal.py"):
        print("Error: cricket_terminal.py not found!")
        print("Please make sure the cricket terminal styling module is in the project root directory.")
        return False

    # Try to modify final.py
    modified = modify_final_py()

    if modified:
        print("\nIntegration successful!")
        print("\nYour terminal output will now have cricket-themed colors and styling.")
        print("Run your analysis as usual with:")
        print("   python final.py")
        return True
    else:
        print("\nIntegration was not successful.")
        print("Please check the output above for specific issues.")
        return False


if __name__ == "__main__":
    main()
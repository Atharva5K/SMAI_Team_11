"""
Cricket Commentary Visualization Module

This module generates visualizations for cricket commentary analysis:
- Sentiment Heatmap: Color-coded representation of emotional intensity in commentary by over
- Word Cloud: Generate word clouds from commentary text, sized by frequency or excitement level
- Commentary Type Distribution: Pie charts showing the distribution of commentary types
- Phase Analysis Chart: Visualization of commentary types across match phases

Dependencies:
- matplotlib
- seaborn
- pandas
- numpy
- wordcloud
- nltk
- PIL (pillow)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import re
import matplotlib.patches as mpatches
from collections import Counter

# Create output directory if it doesn't exist
os.makedirs("Data/Visualizations", exist_ok=True)


# Ensure NLTK resources are available
def setup_nltk():
    """Download required NLTK resources if not already available"""
    import nltk
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('sentiment/vader_lexicon')
    except LookupError:
        print("Downloading required NLTK data...")
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('vader_lexicon')


def load_cricket_shape():
    """
    Load cricket bat or ball shape for word cloud
    If image not found, create a circular mask
    """
    try:
        # Try to load cricket bat image (user should place this in the Data directory)
        cricket_mask = np.array(Image.open("Data/cricket_bat.png"))
        return cricket_mask
    except FileNotFoundError:
        # If image not found, create a circular mask
        print("Cricket shape image not found, creating circular mask...")
        x, y = np.ogrid[:300, :300]
        mask = (x - 150) ** 2 + (y - 150) ** 2 > 130 ** 2
        mask = 255 * np.logical_not(mask)
        return mask


def extract_sentiment_scores(text_lines):
    """
    Extract sentiment scores for each over in the match

    Args:
        text_lines (list): List of lists where each inner list contains commentary for an over

    Returns:
        list: List of sentiment scores for each over
    """
    setup_nltk()
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = []

    for over_idx, over in enumerate(text_lines):
        # Combine all commentary lines for this over
        over_text = ' '.join(over)

        # Calculate sentiment scores
        sentiment = sia.polarity_scores(over_text)

        # Store compound score (ranges from -1 to 1)
        sentiment_scores.append({
            'over': over_idx + 1,
            'compound': sentiment['compound'],
            'positive': sentiment['pos'],
            'negative': sentiment['neg'],
            'neutral': sentiment['neu']
        })

    return sentiment_scores


def create_sentiment_heatmap(text_lines, output_file="Data/Visualizations/sentiment_heatmap.png"):
    """
    Create a heatmap visualization of emotional intensity in commentary by over

    Args:
        text_lines (list): List of lists where each inner list contains commentary for an over
        output_file (str): Path to save the visualization
    """
    print("Generating sentiment heatmap...")

    # Get sentiment scores
    sentiment_scores = extract_sentiment_scores(text_lines)

    # Create DataFrame for visualization
    df = pd.DataFrame(sentiment_scores)

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 4]})

    # Plot compound sentiment as line plot in first subplot
    ax1.plot(df['over'], df['compound'], marker='o', color='black', linewidth=2)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax1.set_ylabel('Overall Sentiment\n(Compound)')
    ax1.set_xlim(0.5, len(df) + 0.5)
    ax1.set_xticklabels([])
    ax1.grid(True, alpha=0.3)

    # Create detailed heatmap data for positive, neutral, negative sentiment
    heatmap_data = df[['positive', 'neutral', 'negative']].T
    heatmap_data.columns = df['over']

    # Custom colormap from red (negative) to white (neutral) to green (positive)
    cmap = LinearSegmentedColormap.from_list(
        'sentiment_cmap', ['#d9534f', '#f9f9f9', '#5cb85c'], N=100)

    # Create heatmap in second subplot
    sns.heatmap(heatmap_data, cmap=cmap, linewidths=0.5, ax=ax2,
                cbar_kws={'label': 'Sentiment Intensity'})

    # Add over numbers to x-axis
    ax2.set_xticklabels([f"{over}" for over in df['over']], rotation=0)
    ax2.set_xlabel('Over Number')

    # Set y-axis labels
    ax2.set_yticklabels(['Positive', 'Neutral', 'Negative'], rotation=0)

    # Add title and adjust layout
    fig.suptitle('Cricket Commentary Sentiment Analysis by Over', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Add annotation for high excitement moments
    threshold = 0.5
    high_excitement = df[df['compound'] > threshold]
    if not high_excitement.empty:
        for _, row in high_excitement.iterrows():
            ax1.annotate('!', (row['over'], row['compound']),
                         xytext=(0, 10), textcoords='offset points',
                         ha='center', va='bottom', fontsize=12, weight='bold')

    # Save the figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Sentiment heatmap saved to {output_file}")
    plt.close()


def generate_word_cloud(text_lines, output_file="Data/Visualizations/commentary_wordcloud.png"):
    """
    Generate an attractive rectangular word cloud from cricket commentary

    Args:
        text_lines (list): List of lists where each inner list contains commentary for an over
        output_file (str): Path to save the visualization
    """
    print("Generating enhanced word cloud...")
    setup_nltk()

    # Flatten all commentary into one text
    all_text = ' '.join([' '.join(over) for over in text_lines])

    # Clean text
    all_text = all_text.lower()
    all_text = re.sub(r'[^\w\s]', '', all_text)

    # Remove stopwords and cricket-specific common words that aren't insightful
    stop_words = set(stopwords.words('english'))
    cricket_common_words = {
        'ball', 'over', 'runs', 'batting', 'bowling', 'batsman',
        'bowler', 'fielder', 'match', 'delivery', 'cricket', 'innings',
        'the', 'and', 'this', 'that', 'with', 'from', 'just', 'now'
    }
    stop_words.update(cricket_common_words)

    # Tokenize and filter
    word_tokens = word_tokenize(all_text)
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words and len(word) > 2]

    # Get word frequencies
    word_freq = Counter(filtered_text)

    # Define a custom colormap that's visually appealing for cricket
    cricket_colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6", "#1abc9c", "#d35400"]

    # Custom color function that cycles through cricket colors based on word position
    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        return cricket_colors[hash(word) % len(cricket_colors)]

    # Generate the word cloud with enhanced settings
    wordcloud = WordCloud(
        width=1000,
        height=600,
        background_color='white',
        colormap=None,
        color_func=color_func,
        max_words=150,
        collocations=False,
        random_state=42,
        prefer_horizontal=0.9,
        min_font_size=10,
        max_font_size=100,
        font_path=None,  # Use default font
        relative_scaling=0.6,  # Balance frequency with visibility
        margin=10,
        mode="RGB"
    ).generate_from_frequencies(word_freq)

    # Create a figure with a specific size and DPI for high quality
    plt.figure(figsize=(14, 8), dpi=200, facecolor='white')
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')

    # Add a title with a custom style
    plt.title('Cricket Commentary Word Cloud',
              fontsize=24,
              weight='bold',
              color='#34495e',
              pad=20,
              fontname='sans-serif')

    # Add a subtle border
    plt.gca().spines['top'].set_visible(True)
    plt.gca().spines['right'].set_visible(True)
    plt.gca().spines['bottom'].set_visible(True)
    plt.gca().spines['left'].set_visible(True)
    plt.gca().spines['top'].set_color('#dddddd')
    plt.gca().spines['right'].set_color('#dddddd')
    plt.gca().spines['bottom'].set_color('#dddddd')
    plt.gca().spines['left'].set_color('#dddddd')

    # Add a caption
    plt.figtext(0.5, 0.01,
                'Word size indicates frequency in commentary',
                ha='center',
                fontsize=12,
                color='#7f8c8d',
                style='italic')

    # Tight layout for better spacing
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    # Save with high quality
    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.2, format='png')
    print(f"Enhanced word cloud saved to {output_file}")
    plt.close()

def create_commentary_distribution_plot(classifications, output_file="Data/Visualizations/commentary_distribution.png"):
    """
    Create pie charts showing the distribution of commentary types

    Args:
        classifications (list): List of classification dicts from CommentaryClassifier
        output_file (str): Path to save the visualization
    """
    print("Generating commentary type distribution charts...")

    # Extract dominant categories from classifications
    categories = [c['dominant_category'] for c in classifications if c['dominant_category'] != 'undetermined']
    category_counts = Counter(categories)

    # Remove 'undetermined' if present for cleaner visualization
    if 'undetermined' in category_counts:
        del category_counts['undetermined']

    # Prepare data for pie chart
    labels = [category.replace('_', ' ').title() for category in category_counts.keys()]
    sizes = list(category_counts.values())
    total = sum(sizes)
    percentages = [count / total * 100 for count in sizes]

    # Define custom colors for each category
    colors = {
        'technical analysis': '#3498db',
        'player praise': '#2ecc71',
        'player criticism': '#e74c3c',
        'strategic observation': '#9b59b6',
        'audience reaction': '#f39c12'
    }

    pie_colors = [colors.get(cat.lower(), '#95a5a6') for cat in category_counts.keys()]

    # Create the figure with subplots
    fig = plt.figure(figsize=(18, 9))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])

    # Pie chart
    ax1 = plt.subplot(gs[0])
    wedges, texts, autotexts = ax1.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        colors=pie_colors,
        wedgeprops={'width': 0.5, 'edgecolor': 'w', 'linewidth': 1.5},
        textprops={'fontsize': 12},
        pctdistance=0.85
    )

    # Equal aspect ratio ensures that pie is drawn as a circle
    ax1.axis('equal')
    plt.setp(autotexts, size=10, weight="bold")
    ax1.set_title('Commentary Type Distribution', fontsize=16, pad=20)

    # Add center circle to make it a donut chart
    centre_circle = plt.Circle((0, 0), 0.4, fc='white')
    ax1.add_patch(centre_circle)

    # Add text in center
    ax1.text(0, 0, f'Total\n{total} Overs',
             horizontalalignment='center',
             verticalalignment='center',
             fontsize=14,
             fontweight='bold')

    # Bar chart for confidence by category
    ax2 = plt.subplot(gs[1])

    # Calculate average confidence per category
    confidence_by_category = {}
    for c in classifications:
        cat = c['dominant_category']
        if cat != 'undetermined':
            if cat not in confidence_by_category:
                confidence_by_category[cat] = []
            confidence_by_category[cat].append(c['avg_confidence'])

    avg_confidence = {cat: sum(conf) / len(conf) for cat, conf in confidence_by_category.items()}

    # Sort categories by count for the bar chart
    sorted_categories = [cat for cat, _ in sorted(category_counts.items(),
                                                  key=lambda x: x[1],
                                                  reverse=True)]

    # Filter matching colors and prepare data for bar chart
    bar_colors = [colors.get(cat.lower(), '#95a5a6') for cat in sorted_categories]
    bar_labels = [cat.replace('_', ' ').title() for cat in sorted_categories]
    confidence_values = [avg_confidence.get(cat, 0) for cat in sorted_categories]

    # Create bar chart
    bars = ax2.barh(bar_labels, confidence_values, color=bar_colors, alpha=0.8, height=0.6)

    # Add value labels to the bars
    for bar in bars:
        width = bar.get_width()
        ax2.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                 f'{width:.2f}',
                 ha='left', va='center',
                 fontsize=10, fontweight='bold')

    ax2.set_title('Average Confidence by Category', fontsize=16, pad=20)
    ax2.set_xlim(0, 1.0)
    ax2.set_xlabel('Confidence Score', fontsize=12)
    ax2.grid(True, axis='x', alpha=0.3)

    # Add overall title
    fig.suptitle('Cricket Commentary Classification Analysis', fontsize=20, y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save the figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Commentary distribution chart saved to {output_file}")
    plt.close()


def create_phase_analysis_chart(classifications, output_file="Data/Visualizations/commentary_phase_analysis.png"):
    """
    Create a visualization of how commentary types change across match phases

    Args:
        classifications (list): List of classification dicts from CommentaryClassifier
        output_file (str): Path to save the visualization
    """
    print("Generating commentary phase analysis chart...")

    # Skip undetermined categories for cleaner visualization
    filtered_classifications = [c for c in classifications if c['dominant_category'] != 'undetermined']

    if not filtered_classifications:
        print("Warning: No valid classifications found for phase analysis")
        return

    # Split match into beginning, middle, and end phases
    num_overs = len(filtered_classifications)
    early_phase = int(num_overs * 0.33)
    middle_phase = int(num_overs * 0.67)

    early_overs = filtered_classifications[:early_phase]
    middle_overs = filtered_classifications[early_phase:middle_phase]
    late_overs = filtered_classifications[middle_phase:]

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

    # Get all unique categories
    all_categories = set()
    for counts in [early_counts, middle_counts, late_counts]:
        all_categories.update(counts.keys())

    # Create data structure for plotting
    categories = list(all_categories)
    categories.sort()  # Sort for consistency

    # Create table data for the stacked bar chart
    data = []
    for category in categories:
        row = [
            early_counts.get(category, 0) / len(early_overs) * 100 if early_overs else 0,
            middle_counts.get(category, 0) / len(middle_overs) * 100 if middle_overs else 0,
            late_counts.get(category, 0) / len(late_overs) * 100 if late_overs else 0
        ]
        data.append(row)

    # Define colors for each category
    colors = {
        'technical analysis': '#3498db',
        'player praise': '#2ecc71',
        'player criticism': '#e74c3c',
        'strategic observation': '#9b59b6',
        'audience reaction': '#f39c12'
    }

    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9),
                                   gridspec_kw={'width_ratios': [2, 1]})

    # Stacked bar chart for percentages
    labels = ['Early Phase', 'Middle Phase', 'Late Phase']
    width = 0.6

    # Create the stacked bars
    bottom = np.zeros(3)
    for i, category in enumerate(categories):
        category_data = data[i]
        color = colors.get(category, '#95a5a6')
        ax1.bar(labels, category_data, width, bottom=bottom,
                label=category.replace('_', ' ').title(), color=color)
        bottom += category_data

    ax1.set_title('Commentary Type Distribution by Match Phase', fontsize=16, pad=20)
    ax1.set_ylabel('Percentage of Overs (%)', fontsize=12)
    ax1.set_ylim(0, 100)
    ax1.grid(True, axis='y', alpha=0.3)

    # Create legend
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=12)

    # Heatmap showing category distribution across phases
    heatmap_data = np.zeros((len(categories), 3))
    for i, category in enumerate(categories):
        heatmap_data[i, 0] = early_counts.get(category, 0)
        heatmap_data[i, 1] = middle_counts.get(category, 0)
        heatmap_data[i, 2] = late_counts.get(category, 0)

    # Create DataFrame for the heatmap
    df_heatmap = pd.DataFrame(heatmap_data,
                              index=[cat.replace('_', ' ').title() for cat in categories],
                              columns=['Early', 'Middle', 'Late'])

    # Create the heatmap
    sns.heatmap(df_heatmap, annot=True, fmt='g', cmap='YlGnBu', ax=ax2,
                linewidths=0.5, cbar_kws={'label': 'Number of Overs'})

    ax2.set_title('Commentary Type Count by Match Phase', fontsize=16, pad=20)

    # Add overall title
    fig.suptitle('Evolution of Cricket Commentary Throughout the Match', fontsize=20, y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.9])

    # Save the figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Phase analysis chart saved to {output_file}")
    plt.close()


def create_visualizations(text_lines, classifications):
    """
    Create all four visualizations using the provided data

    Args:
        text_lines (list): List of lists where each inner list contains commentary for an over
        classifications (list): List of classification dicts from CommentaryClassifier
    """
    try:
        # Create visualizations
        create_sentiment_heatmap(text_lines)
        generate_word_cloud(text_lines)
        create_commentary_distribution_plot(classifications)
        create_phase_analysis_chart(classifications)

        print("\nAll visualizations created successfully!")
        print("Visualizations saved to the Data/Visualizations directory.")
    except Exception as e:
        print(f"Error creating visualizations: {e}")


if __name__ == "__main__":
    print("Cricket Commentary Visualization Module")
    print("This module should be imported and used with the cricket analysis system.")
    print("Example usage:")
    print("  from commentary_visualizer import create_visualizations")
    print("  create_visualizations(text_lines, classifications)")
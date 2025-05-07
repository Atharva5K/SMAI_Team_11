# -*- coding: utf-8 -*-
"""
Commentary Classification using Zero-Shot Learning
"""

from transformers import pipeline
import pandas as pd
import os
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize


# Ensure necessary NLTK data is available
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading required NLTK data...")
        nltk.download('punkt')


# Define the zero-shot classifier
class CommentaryClassifier:
    def __init__(self):
        print("Initializing zero-shot classification model...")
        # Use BART for English commentary - can be replaced with XLM-RoBERTa for multilingual support
        self.classifier = pipeline("zero-shot-classification",
                                   model="facebook/bart-large-mnli",
                                   device=-1)  # Use CPU by default, set to specific GPU index if available

        # Define classification categories
        self.categories = [
            "technical analysis",
            "player praise",
            "player criticism",
            "strategic observation",
            "audience reaction"
        ]

    def classify_commentary(self, text):
        """
        Classify a single commentary text

        Args:
            text (str): Commentary text to classify

        Returns:
            tuple: (top_category, confidence_score)
        """
        if not text or len(text.strip()) < 10:
            return "undetermined", 0.0

        result = self.classifier(text, self.categories)
        top_category = result['labels'][0]
        confidence = result['scores'][0]

        return top_category, confidence

    def classify_over(self, over_text_list):
        """
        Classify all commentary lines in an over and provide summary

        Args:
            over_text_list (list): List of commentary texts for an over

        Returns:
            dict: Classification summary for the over
        """
        classifications = []
        for text in over_text_list:
            category, confidence = self.classify_commentary(text)
            classifications.append({
                'text': text,
                'category': category,
                'confidence': confidence
            })

        # Get the dominant category for this over
        categories = [c['category'] for c in classifications if c['category'] != "undetermined"]
        if not categories:
            dominant_category = "undetermined"
        else:
            # Find most common category
            unique_categories = set(categories)
            dominant_category = max(unique_categories,
                                    key=lambda cat: sum(1 for c in classifications
                                                        if c['category'] == cat))

        # Calculate average confidence for the dominant category
        confidence_scores = [c['confidence'] for c in classifications
                             if c['category'] == dominant_category]
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0

        return {
            'dominant_category': dominant_category,
            'avg_confidence': avg_confidence,
            'classifications': classifications
        }

    def batch_classify_overs(self, text_lines):
        """
        Classify all overs in the match

        Args:
            text_lines (list): List of lists, where each inner list contains commentary for an over

        Returns:
            list: Classification results for each over
        """
        print(f"Classifying {len(text_lines)} overs using zero-shot classification...")
        over_classifications = []

        for i, over in enumerate(text_lines):
            print(f"Classifying over {i + 1}/{len(text_lines)}...", end="\r")
            result = self.classify_over(over)
            result['over_number'] = i + 1
            over_classifications.append(result)

        print("\nClassification complete!")
        return over_classifications

    def analyze_category_distribution(self, classifications):
        """
        Analyze the distribution of commentary categories throughout the match

        Args:
            classifications (list): Classification results from batch_classify_overs

        Returns:
            dict: Analysis of category distribution
        """
        total_overs = len(classifications)
        category_counts = {category: 0 for category in self.categories}
        category_counts["undetermined"] = 0

        # Count dominant categories
        for over_class in classifications:
            category_counts[over_class['dominant_category']] += 1

        # Calculate percentages
        category_percentages = {
            cat: (count / total_overs) * 100
            for cat, count in category_counts.items()
        }

        # Find overs with highest confidence for each category
        best_overs = {}
        for category in self.categories:
            category_overs = [
                over for over in classifications
                if over['dominant_category'] == category
            ]

            if category_overs:
                best_over = max(category_overs, key=lambda x: x['avg_confidence'])
                best_overs[category] = {
                    'over_number': best_over['over_number'],
                    'confidence': best_over['avg_confidence']
                }

        return {
            'category_counts': category_counts,
            'category_percentages': category_percentages,
            'best_overs': best_overs
        }

    def print_classification_report(self, analysis):
        """
        Print a formatted report of the classification analysis

        Args:
            analysis (dict): Analysis from analyze_category_distribution
        """
        print("\n" + "=" * 80)
        print("COMMENTARY CLASSIFICATION ANALYSIS")
        print("=" * 80)

        # Print category distribution
        print("\nDistribution of Commentary Types:")
        print("-" * 50)
        for category, percentage in analysis['category_percentages'].items():
            if category != "undetermined":
                print(f"{category.upper()}: {analysis['category_counts'][category]} overs ({percentage:.1f}%)")

        # Print best examples of each category
        print("\nBest Examples of Each Commentary Type:")
        print("-" * 50)
        for category, data in analysis['best_overs'].items():
            print(f"{category.upper()}: Over {data['over_number']} (Confidence: {data['confidence']:.2f})")
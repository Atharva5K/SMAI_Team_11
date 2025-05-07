# Cricket Commentary Analysis System

An AI-driven sentiment analysis system for cricket commentary that identifies key moments and classifies commentary types using natural language processing techniques.

## Project Overview

This project implements a comprehensive framework for analyzing cricket commentary using various NLP techniques:

* **RNN Model**: Predicts runs and wickets from commentary text
* **Sentiment Analysis**: Identifies emotionally charged moments in commentary
* **Zero-Shot Classification**: Categorizes commentary into types (player praise, player criticism, etc.)
* **Visualization System**: Creates heatmaps, word clouds, and distribution charts
* **Cricket-Themed Terminal Output**: Provides styled presentation of results

## Dataset

The project uses the Cricket Commentary Dataset containing ball-by-ball commentary for international cricket matches.

* **Dataset Link**: [Cricket Commentary Dataset](https://drive.google.com/drive/folders/1npQBMMqnF-QEcKks29c8SkeB4LC62Wb2?usp=sharing)
* Includes:

  * Ball-by-ball commentary text
  * Match metadata
  * Team information
  * Over and delivery details
  * Runs, wickets, and extras information

## Requirements

### System Requirements

* Python 3.8+
* CUDA-compatible GPU (recommended for model training)
* 8GB+ RAM

### Python Dependencies

Install all dependencies via:

```bash
pip install -r requirements.txt
```

Key dependencies:

* TensorFlow 2.8+
* NLTK 3.7+
* Matplotlib 3.5+
* Pandas 1.4+
* Seaborn 0.11+
* Transformers 4.17+
* Wordcloud 1.8+
* scikit-learn 1.0+

## Installation

````
1. Install dependencies:
   ```bash
pip install -r requirements.txt
````

2. Download the dataset and model weights as described below.

## Model Weights

Pre-trained model weights can be downloaded from:

* [Model Weights (Google Drive)](https://drive.google.com/drive/folders/1Hbp8axzA06o6_-96sSIk_DWEoj_4j7eJ?usp=drive_link)

Place the downloaded files in `Code/Pickles/`.

## Directory Structure
Before running the code make sure your directory structure looks as follows
```
cricket-commentary-analysis/
├── Code/                        
│   ├── Pickles/                 
│   ├── commentary_classifier.py 
│   └── vAnds.py                 
├── Data/                      
│   ├── COMMENTARY_INTL_MATCH/   
│   ├── Input/                   
│   └── Visualizations/         
├── final.py                     
├── commentary_visualizer.py    
├── cricket_terminal.py         
├── data_ext.py                  
├── input.py                    
├── rnn.py                    
├── scorecard_generator.py       
├── team_extractor.py            
├── requirements.txt            
└── README.md                    
```

## Execution Steps

Below are detailed instructions for running the complete analysis pipeline:

1. **Data Extraction**
   First, run the data extraction script to prepare the initial dataset:

   ```bash
   python data_ext.py
   ```

   This script:

   * Processes raw cricket commentary files from `Data/COMMENTARY_INTL_MATCH/`
   * Extracts commentary text, runs, and wickets information
   * Creates the `Data/Commentary_Entire.csv` file for model training

2. **Input Preparation**
   Next, run the input preparation script:

   ```bash
   python input.py [options]
   ```

   Options for `input.py`:

   * **List available matches:**

     ```bash
     python input.py --list
     ```

     Displays all available match commentary files in the `Data/COMMENTARY_INTL_MATCH/` directory.
   * **Process a specific match file:**

     ```bash
     python input.py --file 936153_COMMENTARY.csv
     ```

     Extracts commentary from the specified match file.
   * **Interactive mode (default):**

     ```bash
     python input.py
     ```

     Presents a numbered list of available matches and prompts you to select one for analysis.
     The script extracts the necessary columns (`Commentary`, `Over_complete`) and saves them to `Data/Input/input.csv`.

3. **Train the RNN Model (Optional - Skip if using provided weights)**

   ```bash
   python rnn.py
   ```

   Creates a trained model at `Code/Pickles/trained_model.pkl`.

4. **Run Full Analysis Pipeline**

   ```bash
   python final.py
   ```

   This script:

   * Loads the trained model
   * Extracts team information and generates scorecards
   * Predicts runs and wickets for each over
   * Identifies interesting moments in commentary
   * Performs zero-shot classification of commentary types
   * Analyzes commentary distribution across match phases
   * Generates visualizations saved to `Data/Visualizations/`
   * Presents results with cricket-themed terminal styling

5. **Execution Directory Structure**

`

## Running the Code

### Training the Model from Scratch

If you want to train the model from scratch, you need to execute the following scripts in order:

```bash
# Step 1: Extract and prepare the dataset
python data_ext.py

# Step 2: Train the RNN model
python rnn.py
```

This process will:
- Process the raw commentary data
- Train an RNN model to predict runs and wickets
- Save the trained model to `Code/Pickles/trained_model.pkl`

### Running Inference

For analyzing cricket commentary data:

```bash
# Step 1: Prepare the input data from a specific commentary file
python input.py --file <path_to_commentary_csv_file>
# For example: python input.py --file 936153_COMMENTARY.csv

# Step 2: Run the complete analysis pipeline
python final.py
```

This will:
- Load the trained model
- Process the selected match commentary
- Generate visualizations and statistics
- Output analysis with cricket-themed styling


All outputs are saved under `Data/Visualizations/`.

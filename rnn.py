import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Create directories if they don't exist
os.makedirs("Code/Pickles", exist_ok=True)

# Load the dataset
data_file = "Data/Commentary_Entire.csv"
if not os.path.exists(data_file):
    print(f"Error: {data_file} not found. Please run data_ext.py first.")
    exit(1)

print(f"Loading data from {data_file}")
data = pd.read_csv(data_file)

# Preprocess the data
commentary = data["Commentary"].astype(str).values  # Convert to string type
runs = data["Runs"].values
wickets = data["Wickets"].values

print(f"Preprocessing {len(commentary)} commentary lines")

# Tokenize the commentary
tokenizer = Tokenizer()
tokenizer.fit_on_texts(commentary)
sequences = tokenizer.texts_to_sequences(commentary)

# Pad the sequences
max_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_length)

print(f"Split data into training and testing sets")
# Split the data into training and testing sets
X_train, X_test, y_runs_train, y_runs_test, y_wickets_train, y_wickets_test = (
    train_test_split(padded_sequences, runs, wickets, test_size=0.2, random_state=42)
)

# Create the RNN model
print("Creating and training the RNN model (this may take a while)...")
model = Sequential(
    [
        Embedding(len(tokenizer.word_index) + 1, 128, input_length=max_length),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(2, activation="linear"),
    ]
)

# Compile the model
model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])

# Prepare the target variables
y_train = np.column_stack((y_runs_train, y_wickets_train))
y_test = np.column_stack((y_runs_test, y_wickets_test))

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32)

# Evaluate the model
y_pred = model.predict(X_test)
runs_pred = np.round(y_pred[:, 0]).astype(int)
wickets_pred = np.round(y_pred[:, 1]).astype(int)

accuracy_runs = accuracy_score(y_runs_test, runs_pred)
precision_runs = precision_score(y_runs_test, runs_pred, average="weighted")
accuracy_wickets = accuracy_score(y_wickets_test, wickets_pred)
precision_wickets = precision_score(y_wickets_test, wickets_pred, average="weighted")

print("Runs Prediction:")
print("Accuracy:", accuracy_runs)
print("Precision:", precision_runs)
print("Wickets Prediction:")
print("Accuracy:", accuracy_wickets)
print("Precision:", precision_wickets)

output_dir = "Code/Pickles"
os.makedirs(output_dir, exist_ok=True)

# Save the trained model
model_path = os.path.join(output_dir, "trained_model.pkl")
print(f"Saving trained model to {model_path}")
with open(model_path, "wb") as file:
    pickle.dump(model, file)

print("Testing model with sample inputs...")
# Hard-coded commentary input
commentary_input = "short and wide, poor ball. Warner goes back and slaps the ball through cover for four. Easy as ... Not much swing, just timed"
commentary_sequence = tokenizer.texts_to_sequences([commentary_input])
commentary_padded = pad_sequences(commentary_sequence, maxlen=max_length)

prediction = model.predict(commentary_padded)
predicted_runs = np.round(prediction[:, 0]).astype(int)[0]
predicted_wickets = np.round(prediction[:, 1]).astype(int)[0]

print("Predicted Runs:", predicted_runs)
print("Predicted Wickets:", predicted_wickets)

# Hard-coded commentary input
commentary_input = "taken! Yasir Shah strikes. Flatter trajectory outside off, gets it to turn a touch but more importantly, gets some extra bounce. Smith sees something short and goes back to cut. The extra bounce is responsible for a thick outside edge. Sarfraz hangs onto a quality reflex catch, had his gloves in the right spot."
commentary_sequence = tokenizer.texts_to_sequences([commentary_input])
commentary_padded = pad_sequences(commentary_sequence, maxlen=max_length)

prediction = model.predict(commentary_padded)
predicted_runs = np.round(prediction[:, 0]).astype(int)[0]
predicted_wickets = np.round(prediction[:, 1]).astype(int)[0]

print("Predicted Runs:", predicted_runs)
print("Predicted Wickets:", predicted_wickets)

print("RNN training and model creation completed successfully!")
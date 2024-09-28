import os
import csv
import re
import pandas as pd
from transformers import AutoTokenizer

# Set up paths
RAW_DATA_PATH = 'data/raw/messages.csv'
PROCESSED_DATA_PATH = 'data/processed/cleaned_messages.csv'

# Ensure the processed data directory exists
os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)

# Load a pre-trained tokenizer for Amharic (e.g., XLM-Roberta)
def load_pretrained_tokenizer():
    # You can use any multilingual tokenizer here. XLM-Roberta is an example.
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    return tokenizer

# Amharic-specific text normalization function (keeping prices)
def normalize_amharic_text(text):
    # Check if text is a string; if not, replace with an empty string
    if not isinstance(text, str):
        return ""
    
    # Remove unwanted characters but keep numbers (prices) intact
    text = re.sub(r'[^\w\s\d]', '', text)  # Remove punctuation but keep digits
    text = text.replace('\n', ' ')         # Replace newlines with spaces
    return text

# Function to remove leading underscores from tokens (e.g., "▁እንኳን" -> "እንኳን")
def remove_underscores(tokens):
    return [token.lstrip('▁') for token in tokens]

# Tokenize using Hugging Face's pre-trained tokenizer
def preprocess_text_hf(text, max_length=512):
    # Normalize the text
    normalized_text = normalize_amharic_text(text)
    
    # Load the pre-trained tokenizer
    tokenizer = load_pretrained_tokenizer()
    
    # Tokenize the text using the pre-trained tokenizer
    tokens = tokenizer.tokenize(normalized_text)
    
    # Truncate tokens if they exceed the maximum sequence length
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    
    # Remove underscores from tokens
    tokens = remove_underscores(tokens)
    
    return tokens

# Read raw data (messages.csv)
def load_raw_data(filepath=RAW_DATA_PATH):
    data = pd.read_csv(filepath)
    return data

# Process each message and save cleaned data
def process_and_save_data():
    # Load the raw messages
    data = load_raw_data()

    processed_data = []
    for index, row in data.iterrows():
        message_text = row['message_text']
        sender_id = row['sender_id']
        channel = row['channel']
        date = row['date']  # Read the date column from the CSV

        # Preprocess the message using the pre-trained tokenizer
        tokens = preprocess_text_hf(message_text)

        # Store the cleaned data in a list of dictionaries
        processed_data.append({
            'sender_id': sender_id,
            'channel': channel,
            'date': date,  # Include date in the processed data
            'cleaned_message': ' '.join(tokens).strip()  # Save tokens as a string and remove extra spaces
        })

    # Save the processed data to a CSV
    df = pd.DataFrame(processed_data)
    df.to_csv(PROCESSED_DATA_PATH, index=False, encoding='utf-8')
    print(f"Processed data saved to {PROCESSED_DATA_PATH}")

if __name__ == '__main__':
    process_and_save_data()

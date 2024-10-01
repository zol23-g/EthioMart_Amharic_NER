import os
import re
import pandas as pd

# Set up paths
RAW_DATA_PATH = 'data/raw/messages.csv'
PROCESSED_DATA_PATH = 'data/processed/cleaned_messages.csv'

# Ensure the processed data directory exists
os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)

# Amharic-specific text normalization function (keeping prices)
def normalize_amharic_text(text):
    # Check if text is a string; if not, replace with an empty string
    if not isinstance(text, str):
        return ""
    
    # Remove unwanted characters but keep numbers (prices) intact
    # Remove punctuation, but keep spaces and digits
    text = re.sub(r'[^\w\s\d]', '', text)  # Remove punctuation but keep digits
    text = text.replace('\n', ' ')  # Replace newlines with spaces
    return text

# Function to check if a token is English
def is_english_word(token):
    # Match only tokens that contain English letters (A-Z, a-z)
    return re.match(r'^[A-Za-z]+$', token) is not None

# Custom tokenizer function to split text into tokens by spaces and underscores, and exclude English words
def custom_tokenizer(text):
    # Normalize the text first
    normalized_text = normalize_amharic_text(text)
    
    # Tokenize by splitting the normalized text by spaces
    tokens = normalized_text.split()  # Split based on whitespace
    
    # Further split tokens that contain underscores and filter out English words
    amharic_tokens = []
    for token in tokens:
        # Split by underscore, but also keep words intact
        subtokens = token.split('_')  # Split by underscore
        for subtoken in subtokens:
            if not is_english_word(subtoken):  # Filter out English words
                amharic_tokens.append(subtoken)

    return amharic_tokens

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

        # Preprocess the message using the custom tokenizer
        tokens = custom_tokenizer(message_text)

        # Store the cleaned data in a list of dictionaries
        processed_data.append({
            'sender_id': sender_id,
            'channel': channel,
            'date': date,  # Include date in the processed data
            'cleaned_message': ' '.join(tokens).strip()  # Join tokens as a string and remove extra spaces
        })

    # Save the processed data to a CSV
    df = pd.DataFrame(processed_data)
    df.to_csv(PROCESSED_DATA_PATH, index=False, encoding='utf-8')
    print(f"Processed data saved to {PROCESSED_DATA_PATH}")

if __name__ == '__main__':
    process_and_save_data()

import os
import pandas as pd
import re

# Paths for the raw data and where to save the labeled data
RAW_DATA_PATH = 'data/processed/cleaned_messages.csv'
LABELED_DATA_PATH = 'data/labeled/labeled_data.conll'

# Ensure the labeled data directory exists
os.makedirs(os.path.dirname(LABELED_DATA_PATH), exist_ok=True)

# Function to label tokens manually
def label_entity(token):
    print(f"\nToken: {token}")
    entity = input("Enter entity (B-Product, I-Product, B-LOC, I-LOC, B-PRICE, I-PRICE, O for other): ").strip()
    if entity not in ["B-Product", "I-Product", "B-LOC", "I-LOC", "B-PRICE", "I-PRICE", "O"]:
        print("Invalid label. Please try again.")
        return label_entity(token)
    return entity

# Function to preprocess and tokenize the messages
def preprocess_and_tokenize(text):
    # Split the text into words/tokens
    tokens = re.split(r'\s+', text)
    return tokens

# Function to label each token of each message and save it in CoNLL format
def label_and_save_data():
    # Load the preprocessed raw messages
    data = pd.read_csv(RAW_DATA_PATH)

    with open(LABELED_DATA_PATH, 'w', encoding='utf-8') as file:
        for index, row in data.iterrows():
            message = row['cleaned_message']
            sender_id = row['sender_id']
            channel = row['channel']
            
            print(f"\nLabeling message {index+1}/{len(data)} from sender {sender_id} in channel {channel}")
            print(f"Message: {message}")
            
            tokens = preprocess_and_tokenize(message)

            # Label each token manually
            for token in tokens:
                if token:  # Ignore empty tokens
                    label = label_entity(token)
                    file.write(f"{token}\t{label}\n")
            file.write("\n")  # Blank line to separate sentences/messages
            
    print(f"\nLabeled data saved to {LABELED_DATA_PATH}")

if __name__ == '__main__':
    label_and_save_data()

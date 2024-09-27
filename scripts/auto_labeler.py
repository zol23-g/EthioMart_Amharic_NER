import os
import re
import pandas as pd

# Set up paths
RAW_DATA_PATH = 'data/processed/cleaned_messages.csv'
LABELED_DATA_PATH = 'data/labeled/auto_labeled_data.conll'

# Ensure the labeled data directory exists
os.makedirs(os.path.dirname(LABELED_DATA_PATH), exist_ok=True)

# Define patterns and keywords for labeling
location_keywords = ['አድራሻ']  # Keywords that start locations
location_sequences = [['መ', 'ገና', 'ኛ'], ['ሜ', 'ክሲ', 'ኮ'], ['ቦ', 'ሌ']]  # I-LOC sequences
price_keywords = ['ዋጋ']  # Price indicators

# Function to automatically label a message based on predefined patterns
def label_message(tokens):
    labels = []
    is_b_product = True  # We assume the first English tokens are B-Product
    is_product = False
    is_b_loc = False
    is_price = False
    
    i = 0
    while i < len(tokens):
        token = tokens[i]
        
        # Stop labeling products and other English tokens after we encounter B-PRICE
        if token in price_keywords:
            labels.append((token, 'B-PRICE'))
            is_price = True  # Set the flag to indicate we've encountered a price
            break  # Exit the loop since we don't process further tokens
        
        # Label the first English words as B-Product or I-Product until we hit B-PRICE
        if re.match(r'[A-Za-z]', token):
            if is_b_product:  # First English token
                labels.append((token, 'B-Product'))
                is_b_product = False  # Switch to I-Product for subsequent tokens
            else:
                labels.append((token, 'I-Product'))  # All following English tokens as I-Product
        elif re.match(r'[A-Za-z]', token):  # In case of more English tokens after price
            labels.append((token, 'O'))  # Other English tokens are labeled O after B-PRICE
            
        # Label specific keywords for location
        elif token in location_keywords:
            labels.append((token, 'B-LOC'))
            is_b_loc = True
        
        # Handle location sequences: 'መ ገና ኛ', 'ሜ ክሲ ኮ', 'ቦ ሌ'
        elif is_b_loc or token in [seq[0] for seq in location_sequences]:
            for seq in location_sequences:
                # Check if the current token matches the first token in any location sequence
                if tokens[i:i+len(seq)] == seq:
                    for j in range(len(seq)):
                        labels.append((tokens[i+j], 'I-LOC'))
                    i += len(seq) - 1  # Skip the entire sequence after labeling
                    break
            else:
                # If no sequence match, label as O
                labels.append((token, 'O'))
            is_b_loc = False
        
        # Default label as O for other tokens
        else:
            labels.append((token, 'O'))
        
        i += 1
    
    return labels

# Tokenize the message into words
def tokenize_message(message):
    # Check if the message is a string; if not, return an empty list
    if not isinstance(message, str):
        return []
    
    # Tokenize by splitting on whitespace
    return re.split(r'\s+', message.strip())

# Read the cleaned messages and automatically label them
def process_and_label_data():
    # Load the cleaned messages
    data = pd.read_csv(RAW_DATA_PATH)

    with open(LABELED_DATA_PATH, 'w', encoding='utf-8') as file:
        for index, row in data.iterrows():
            message_text = row['cleaned_message']
            
            # Tokenize the message
            tokens = tokenize_message(message_text)

            # Automatically label the message
            labeled_tokens = label_message(tokens)

            # Write the labeled tokens to the CoNLL file
            for token, label in labeled_tokens:
                file.write(f"{token}\t{label}\n")
            file.write("\n")  # Blank line to separate messages

    print(f"Labeled data saved to {LABELED_DATA_PATH}")

if __name__ == '__main__':
    process_and_label_data()

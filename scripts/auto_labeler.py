import os
import re
import pandas as pd

# Set up paths
RAW_DATA_PATH = 'data/processed/cleaned_messages.csv'
LABELED_DATA_PATH = 'data/labeled/auto_labeled_data.conll'

# Ensure the labeled data directory exists
os.makedirs(os.path.dirname(LABELED_DATA_PATH), exist_ok=True)

# Define patterns and keywords for labeling
location_keywords = ['ቁጥር', '44', '210']  # Keywords that start locations
location_sequences = [['መ', 'ገና', 'ኛ'], ['ሜ', 'ክሲ', 'ኮ'], ['22', 'ማ', 'ዞ', 'ሪያ'], ['ቦ', 'ሌ']]  # I-LOC sequences
price_keywords = ['ዋጋ']  # Price indicators
product_keywords = [['ኦ', 'ሪ', 'ጅ', 'ናል'], ['የ', 'ል', 'ጆች']]  # B-Product sequences
product_sequences = [['ማ', 'ሊያ'], ['ቁም', 'ጣ']]  # I-Product sequences

# Function to automatically label a message based on predefined patterns
def label_message(tokens):
    labels = []
    is_b_loc = False
    is_price = False
    is_b_product = False
    
    i = 0
    while i < len(tokens):
        token = tokens[i]
        
        # Skip English tokens entirely
        if re.match(r'[A-Za-z]', token):
            i += 1
            continue
        
        # Label 'ዋጋ' as B-PRICE
        elif token in price_keywords:
            labels.append((token, 'B-PRICE'))
            is_price = True
        
        # Label the number before 'ብር' as I-PRICE
        elif is_price and token == 'ብር':
            labels.append((token, 'I-PRICE'))
            is_price = False  # Reset after labeling 'ብር'
        
        # Label digits as I-PRICE if they come before 'ብር'
        elif is_price and token.isdigit():
            labels.append((token, 'I-PRICE'))
        
        # Check for product sequences
        elif is_b_product or token in [seq[0] for seq in product_keywords]:
            for seq in product_keywords:
                if tokens[i:i+len(seq)] == seq:
                    labels.append((tokens[i], 'B-Product'))
                    is_b_product = True
                    i += len(seq) - 1  # Skip the entire sequence after labeling
                    break
            else:
                labels.append((token, 'O'))
                is_b_product = False
        
        # Check for I-Product sequences
        elif is_b_product or token in [seq[0] for seq in product_sequences]:
            for seq in product_sequences:
                if tokens[i:i+len(seq)] == seq:
                    for j in range(len(seq)):
                        labels.append((tokens[i+j], 'I-Product'))
                    i += len(seq) - 1  # Skip the entire sequence after labeling
                    break
            else:
                labels.append((token, 'O'))
                is_b_product = False
        
        elif token in location_keywords:
            labels.append((token, 'B-LOC'))
            is_b_loc = True
        
        # Handle location sequences: 'መ ገና ኛ', 'ሜ ክሲ ኮ', 'ቦ ሌ'
        elif is_b_loc or token in [seq[0] for seq in location_sequences]:
            for seq in location_sequences:
                if tokens[i:i+len(seq)] == seq:
                    for j in range(len(seq)):
                        labels.append((tokens[i+j], 'I-LOC'))
                    i += len(seq) - 1  # Skip the entire sequence after labeling
                    break
            else:
                labels.append((token, 'O'))
            is_b_loc = False
        
        else:
            labels.append((token, 'O'))
        
        i += 1
    
    return labels

# Tokenize the message into words
def tokenize_message(message):
    if not isinstance(message, str):
        return []
    return re.split(r'\s+', message.strip())

# Read the cleaned messages and automatically label them
def process_and_label_data():
    data = pd.read_csv(RAW_DATA_PATH)

    with open(LABELED_DATA_PATH, 'w', encoding='utf-8') as file:
        for index, row in data.iterrows():
            message_text = row['cleaned_message']
            tokens = tokenize_message(message_text)
            labeled_tokens = label_message(tokens)

            for token, label in labeled_tokens:
                file.write(f"{token}\t{label}\n")
            file.write("\n")  # Blank line to separate messages

    print(f"Labeled data saved to {LABELED_DATA_PATH}")

if __name__ == '__main__':
    process_and_label_data()

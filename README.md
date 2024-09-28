
# EthioMart Amharic NER Labeling

## Project Overview

This project is part of the EthioMart initiative to build a Named Entity Recognition (NER) system that automatically labels relevant entities (such as product names, prices, and locations) in both English and Amharic text. The system processes product-related messages, tokenizes the text, and labels entities using predefined patterns.

## Features

- **Data Preprocessing**: Cleans and tokenizes raw messages into individual tokens.
- **Entity Labeling**: Automatically labels tokens with relevant entities like `B-Product`, `I-Product`, `B-LOC`, `I-LOC`, `B-PRICE`, `I-PRICE`, and `O`.
- **Support for Multilingual Text**: Handles both English and Amharic text, recognizing patterns specific to each language.

## Entity Types

The project focuses on the following entity types:
- **B-Product**: The beginning of a product entity.
- **I-Product**: Inside a product entity.
- **B-LOC**: The beginning of a location entity.
- **I-LOC**: Inside a location entity.
- **B-PRICE**: The beginning of a price entity.
- **I-PRICE**: The numeric value associated with a price.
- **O**: Tokens that are outside any entities.

## Setup

### Prerequisites

- Python 3.x
- Required Python packages: `pandas`, `re`, and `unittest` (for testing).

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ethiomart-ner-labeling.git
   cd ethiomart-ner-labeling
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the required directories are present:
   - `data/processed/`: For storing cleaned message data.
   - `data/labeled/`: For saving labeled output in CoNLL format.

## Usage

1. **Process and Label Data**:
   Run the `process_and_label_data` function to preprocess and label the messages. This function reads raw message data, tokenizes it, and applies entity labeling.
   ```bash
   python scripts/auto_labeler.py
   ```

2. **Check Output**:
   The labeled data will be saved in `data/labeled/auto_labeled_data.conll`.

## Testing

This project includes unit tests for key functions. To run the tests:

```bash
python -m unittest discover tests/
```

The tests ensure that tokenization and labeling are functioning as expected.

import unittest
import os
import pandas as pd
from scripts.preprocess_data import load_raw_data, normalize_amharic_text, preprocess_text_hf, process_and_save_data

class TestAmharicProcessing(unittest.TestCase):

    def setUp(self):
        # Create a temporary CSV file for testing
        self.test_raw_data_path = 'data/raw/test_messages.csv'
        self.test_processed_data_path = 'data/processed/test_cleaned_messages.csv'
        
        # Sample data for testing
        sample_data = {
            'message_text': ['እንኳን ወደ ዚህ ገጽ በደህና መጡ!', 'ዋጋ 100 ብር ነው', 'ምን እንደሚል ነው?'],
            'sender_id': [1, 2, 3],
            'channel': ['chat', 'chat', 'chat'],
            'date': ['2024-01-01', '2024-01-02', '2024-01-03']
        }
        
        # Save sample data to CSV
        df = pd.DataFrame(sample_data)
        df.to_csv(self.test_raw_data_path, index=False)

    def test_load_raw_data(self):
        # Test if the raw data loads correctly
        data = load_raw_data(self.test_raw_data_path)
        self.assertEqual(len(data), 3)  # Check if 3 rows are loaded
        self.assertIn('message_text', data.columns)  # Check if the column exists

    def test_normalize_amharic_text(self):
        # Test normalization function
        raw_text = 'እንኳን! ወደ ዚህ ገጽ በደህና መጡ!'
        normalized_text = normalize_amharic_text(raw_text)
        self.assertEqual(normalized_text, 'እንኳን ወደ ዚህ ገጽ በደህና መጡ')  # Check if punctuation is removed

    def test_preprocess_text_hf(self):
        # Test the preprocessing function
        text = 'ዋጋ 100 ብር ነው'
        tokens = preprocess_text_hf(text)
        self.assertGreater(len(tokens), 0)  # Ensure tokens are generated
        self.assertIn('ዋጋ', tokens)  # Check if specific token is present

    def tearDown(self):
        # Remove test files after tests
        if os.path.exists(self.test_raw_data_path):
            os.remove(self.test_raw_data_path)
        if os.path.exists(self.test_processed_data_path):
            os.remove(self.test_processed_data_path)

if __name__ == '__main__':
    unittest.main()

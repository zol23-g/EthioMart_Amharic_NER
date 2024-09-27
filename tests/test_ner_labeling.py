import unittest
from scripts.auto_labeler import tokenize_message
class TestNERLabeling(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.sample_messages = [
            "Super Ab sor bent Bath room Mat ዋጋ 600 ብር",
            "አድራሻ መ ገና ኛ ቦ ሌ ሜ ክሲ ኮ",
            "Travel S que eze Bo ttle 4 pack Cap a city ዋጋ 1000 ብር"
        ]

        self.price_keywords = ['ዋጋ']
        self.location_keywords = ['አድራሻ']
        self.location_sequences = [['መ', 'ገና', 'ኛ'], ['ሜ', 'ክሲ', 'ኮ'], ['ቦ', 'ሌ']]

    def test_tokenize_message(self):
        # Testing tokenization
        from scripts.auto_labeler import tokenize_message  # Replace with the actual script name

        for message in self.sample_messages:
            tokens = tokenize_message(message)
            self.assertIsInstance(tokens, list)
            self.assertGreater(len(tokens), 0)

def test_label_message(self):
    from scripts.auto_labeler import label_message  # Replace with the actual script name

    # Expected output labels for the first sample message
    expected_labels = [
        ('Super', 'B-Product'),
        ('Ab', 'I-Product'),
        ('sor', 'I-Product'),
        ('bent', 'I-Product'),
        ('Bath', 'I-Product'),
        ('room', 'I-Product'),
        ('Mat', 'I-Product'),
        ('ዋጋ', 'B-PRICE'),
        ('600', 'I-PRICE'),
        ('ብር', 'O')  # If there are any English tokens after B-PRICE, they should be labeled as O
    ]

    tokens = tokenize_message(self.sample_messages[0])  # Get tokens from the first sample
    labels = label_message(tokens)  # Get labels from the label_message function

    self.assertEqual(len(labels), len(expected_labels))

    for (token, label), (expected_token, expected_label) in zip(labels, expected_labels):
        self.assertEqual(token, expected_token)
        self.assertEqual(label, expected_label)

    def test_location_sequence_labeling(self):
        from scripts.auto_labeler import label_message  # Replace with the actual script name

        # Test the location labeling
        tokens = tokenize_message("አድራሻ መ ገና ኛ ቦ ሌ ሜ ክሲ ኮ")
        expected_labels = [
            ('አድራሻ', 'B-LOC'),
            ('መ', 'I-LOC'),
            ('ገና', 'I-LOC'),
            ('ኛ', 'I-LOC'),
            ('ቦ', 'I-LOC'),
            ('ሌ', 'I-LOC'),
            ('ሜ', 'I-LOC'),
            ('ክሲ', 'I-LOC'),
            ('ኮ', 'I-LOC')
        ]

        labels = label_message(tokens)
        self.assertEqual(len(labels), len(expected_labels))

        for (token, label), (expected_token, expected_label) in zip(labels, expected_labels):
            self.assertEqual(token, expected_token)
            self.assertEqual(label, expected_label)

if __name__ == '__main__':
    unittest.main()

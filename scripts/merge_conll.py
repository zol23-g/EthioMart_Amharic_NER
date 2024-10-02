from tqdm import tqdm
def load_conll_files(conll_files):
    merged_data = []
    for file_path in tqdm(conll_files):
        with open(file_path, 'r', encoding='utf-8') as file:
            sentence, labels = [], []
            for line in file:
                # Check for non-empty lines
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) == 2:  # Ensure there are exactly 2 parts (word and label)
                        word, label = parts
                        sentence.append(word)
                        labels.append(label)
                    else:
                        print(f"Skipping malformed line in {file_path}: {line.strip()}")
                else:
                    # If it's an empty line, store the current sentence and labels, then reset
                    if sentence:
                        merged_data.append((sentence, labels))
                        sentence, labels = [], []
    # Append the last sentence if file doesn't end with a newline
    if sentence:
        merged_data.append((sentence, labels))
    return merged_data
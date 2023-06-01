# Port data into Python
chords_raw = []

# Open the txt file and read its contents
with open("master_chords.txt", 'r') as file:
    lines = file.readlines()

    # Iterate over each line in the text file
    for line in lines:
        line = line.strip()  # Remove leading/trailing whitespaces and newlines
        # Remove faulty data
        length = len(line.split(","))
        if 3 < length < 200:
            chords_raw.append(line.split(","))

# Turn data into numbers
replacements = {}
num = 1
numeric_data = []

for i in range(len(chords_raw)):
  to_add = []
  for j in range(len(chords_raw[i])):
    if chords_raw[i][j] in replacements:
      to_add.append(replacements[chords_raw[i][j]])
    else:
      replacements[chords_raw[i][j]] = num
      to_add.append(num)
      num += 1
  numeric_data.append(to_add)

# Augment data

# Define data augmentation transformations
def augment_data(data):
  rng = np.random.RandomState()
  augmented_data = []
  for sample in data:
      # Apply transformations to the sample
      augmented_sample = sample + rng.uniform(-0.1, 0.1, size=sample.shape)
      augmented_data.append(augmented_sample)
  return np.array(augmented_data)

for i in range(1000):
  # Apply data augmentation to the dataset
  augmented_data = augment_data(numeric_data)
  # Concatenate the original and augmented data
  numeric_data = np.concatenate((numeric_data, augmented_data))

in_model = []
out_model = []

for i in numeric_data:
  in_model.append([])
  out_model.append(i[0])
  for j in range(len(i) - 1):
    in_model.append(i[:j + 1])
    out_model.append(i[j + 1])

# One hot encode outputs
encoded = to_categorical(out_model) # Use list comprehension as this function expects integers

# Find maximum length of input sequence for padding
max_val = max([len(i) for i in in_model])

in_model = np.array(in_model)
train_labels = np.array(encoded)

# Pad the input chord sequences
padding_token_index = 0
train_data = tf.keras.preprocessing.sequence.pad_sequences(in_model, maxlen=max_val, padding="post", value=padding_token_index)

# Normalize the data
train_data = [j / max_val for j in train_data]

# Train test split
train, test, train_labels, test_labels = train_test_split(train_data, train_labels, test_size=0.05, shuffle = False) # Augmented data set large enough for small testing subset
train, test, train_labels, test_labels = np.array(train), np.array(test), np.array(train_labels), np.array(test_labels)
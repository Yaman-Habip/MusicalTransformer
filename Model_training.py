# Define hyperparameters and create the model
num_layers = 128
d_model = 1024
num_heads = 128
dff = 1024
input_vocab_size = len(train_labels[0])
target_vocab_size = len(train_labels[0])
dropout_rate = 0.1

model = TransformerModel(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=input_vocab_size,
    target_vocab_size=target_vocab_size,
    dropout_rate=dropout_rate,
)

# Compile the model
model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.CategoricalCrossentropy(from_logits=True), metrics = ["accuracy"])

# Train the model
model.fit(train, train_labels, batch_size=64, epochs=20, verbose = 1, shuffle = False)

# Test the model
test_result = model.evaluate(test, test_labels, verbose = 0)
print("Test accuracy:", test_result[1])
# Write functions for users to access the generative abilities of the model

# Make a prediction on empty where the most desired result of the softmax is always picked
def predict_max_prob(iters, to_fill_param=[]):  # iters must be smaller than max_val, to_fill_param defaults to empty

    # Quantize user input
    to_fill = to_fill_param.copy()
    if len(to_fill) > 0:
        for element, index in replacements.items():
            np.array(to_fill)[np.array(to_fill) == element] = index

    # Loop through and feed the transformer revised output, always taking the most prefered chord
    for i in range(iters):
        to_predict = np.array([to_fill, ])
        to_predict = np.array(
            tf.keras.utils.pad_sequences(to_predict, maxlen = max_val, padding = "post", value = padding_token_index))
        prediction = model.predict(to_predict, verbose = 0)[0]

        # Translate the softmax to a list of numbers, take the most likely outcome
        numerical_prediction = np.where(prediction == max(prediction))[0][0]
        to_fill.append(numerical_prediction)

    # Translate the results to chords
    actual_chords = []
    for i in to_fill:
        actual_chords.append(list(replacements.keys())[list(replacements.values()).index(i)])

    return actual_chords


# Make a prediction on empty where the most desired softmax is sampled
def predict_sampled_prob(iters,
                         to_fill_param=[]):  # iters must be smaller than max_val, to_fill_param defaults to empty

    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def select_element_with_probability(vector):
        selected_element = np.random.choice(len(vector), p = vector)
        return selected_element

    # Quantize user input
    to_fill = to_fill_param.copy()
    if len(to_fill) > 0:
        for element, index in replacements.items():
            np.array(to_fill)[np.array(to_fill) == element] = index

    # Loop through and feed the transformer revised output in a Markov process
    for i in range(iters):
        to_predict = np.array([to_fill, ])
        to_predict = np.array(
            tf.keras.utils.pad_sequences(to_predict, maxlen = max_val, padding = "post", value = padding_token_index))
        prediction = model.predict(to_predict, verbose = 0)[0]

        # Translate the softmax to a list of numbers, sample the resulting vector
        numerical_prediction = softmax(prediction)
        to_fill.append(select_element_with_probability(numerical_prediction))

    # Translate the results to chords
    actual_chords = []
    for i in to_fill:
        actual_chords.append(list(replacements.keys())[list(replacements.values()).index(i)])

    return actual_chords
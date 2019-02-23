import keras
import numpy as np

HIDDEN_SIZE = 16 # size of recurrent state. number of parameters grows quadratically with this
EMBED_SIZE = 16 # size of embedding. number of parameters grows linearly with this
NUM_CHARS = 71 # only use the NUM_CHARS most common characters. reduces the number of parameters needed in the embedding, and probably improves accuracy

filter_chars = 10
chars = ''' etaonsihrldumcwgf,ypbvk.-\n;I"'ATS!HBWEqNCPx?OLjRFMDGzYQJU():KV1028573*4Z69X_$][&'''  # characters sorted from most to least frequent
char_to_idx = dict(zip(chars[:-filter_chars], range(len(chars))))
idx_to_char = dict(zip(range(len(chars)), chars[:-filter_chars]))

model = keras.Sequential()
model.add(keras.layers.Embedding(NUM_CHARS, EMBED_SIZE, batch_input_shape=(1, None)))
for i in range(1):
    model.add(keras.layers.GRU(
        HIDDEN_SIZE, return_sequences=True, stateful=True))
model.add(keras.layers.TimeDistributed(keras.layers.Dense(len(chars) - filter_chars, activation='softmax')))
print(model.summary())

# started training at 12:06
model.load_weights('moby-dick.h5')


def infer(ch):
    '''predict the next character given the previous one'''
    test_input = np.array([char_to_idx.get(ch, 0)], dtype=np.int32)
    test_input = np.expand_dims(test_input, axis=0)
    result = model.predict(test_input)
    result_text = np.argmax(result, axis=2)
    return idx_to_char[result_text.flatten()[0]]


# scoring script. not part of submission
print('starting scoring')
with open('whale2.txt') as mobydick:
    full_text = mobydick.read()

    correct = 0
    incorrect = 0

    for i, ch in enumerate(full_text[:10000]):
        predict = infer(ch)
        if predict == full_text[i+1]:
            correct += 1
        else:
            incorrect += 1

        if i % 1000 == 0:
            print('correct: {}, incorrect: {}'.format(correct, incorrect))


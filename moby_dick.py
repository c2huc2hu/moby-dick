#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
import numpy as np


# In[ ]:


# grab descriptive statistics about the corpus
file = open('whale2.txt')
full_text = file.read()
import collections

character_set = set(full_text)
character_counts = collections.Counter(full_text)
print(''.join(([k for k, v in character_counts.most_common()])))
print(sorted(list(character_set)), len(character_set)) # 81 unique characters


# In[6]:


HIDDEN_SIZE = 16 # size of recurrent state. number of parameters grows quadratically with this
EMBED_SIZE = 16 # size of embedding. number of parameters grows linearly with this
NUM_CHARS = 71 # only use the NUM_CHARS most common characters. reduces the number of parameters needed in the embedding, and probably improves accuracy

# character level language model
# text = "abc abcd abcdef abcdefghijk abc ab abc abc abcd ab" # sample input for testing
mobydick = open('whale2.txt')
text = mobydick.read()

# one-hot code the text
chars = ''' etaonsihrldumcwgf,ypbvk.-\n;I"'ATS!HBWEqNCPx?OLjRFMDGzYQJU():KV1028573*4Z69X_$][&'''
char_to_idx = dict(zip(chars[:NUM_CHARS], range(len(chars))))
idx_to_char = dict(zip(range(len(chars)), chars[:NUM_CHARS]))

# create input and output tensors to learn from. we never output the first character or input the last char
input_ = np.array([char_to_idx.get(ch, 0) for ch in text], dtype=np.int32)
output = keras.utils.to_categorical(input_[1:], NUM_CHARS)
output = np.expand_dims(output, axis=1)
input_ = input_[:-1]
print(input_.shape)
print(output.shape)

# splice the full sequence into shorter sequences for training
training_seq_len = 16
if len(input_) % training_seq_len == 0:
    np.append(input_, 0)
training_input = input_[:len(input_) // training_seq_len * training_seq_len].reshape(
    (-1, training_seq_len))
training_output = output[:len(input_) // training_seq_len * training_seq_len,:,:].reshape(
    (-1, training_seq_len, NUM_CHARS))

print(training_input.shape, training_output.shape)


# In[11]:


model = keras.Sequential()
model.add(keras.layers.Embedding(NUM_CHARS, EMBED_SIZE, batch_input_shape=(1, training_seq_len)))
for i in range(1):
    model.add(keras.layers.GRU(
        HIDDEN_SIZE, return_sequences=True, stateful=True, unroll=True))
model.add(keras.layers.TimeDistributed(keras.layers.Dense(len(chars) - filter_chars, activation='softmax')))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

print(model.summary())


# In[12]:


model.fit(training_input, training_output, batch_size=1,
          epochs=1, shuffle=False) # , validation_data=(input_, output), validation_freq=10)  # supported on master, not on latest stable
model.save(
    './moby-dick.h5')


# In[ ]:


# test inference
model.reset_states()
test_input = np.array([char_to_idx.get(ch, 0) for ch in 'abc'], dtype=np.int32)
test_input = np.expand_dims(test_input, axis=0)
result = model.predict(test_input)
result_text = np.argmax(result, axis=2)
print([idx_to_char[ch] for ch in result_text.flatten().tolist()])


# In[ ]:


def infer(ch):
    '''predict the next character given the previous one'''
    test_input = np.array([char_to_idx.get(ch, 0)])
    test_input = np.expand_dims(test_input, axis=0)
    result = model.predict(test_input)
    result_text = np.argmax(result, axis=2)
    return idx_to_char[result_text.flatten()[0]]

infer('b')


# In[ ]:


import os
print(os.listdir())
keras.__version__


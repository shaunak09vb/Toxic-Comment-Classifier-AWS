# -------------------------------------------------------------------------
#                           Import Libraries
# -------------------------------------------------------------------------
import numpy as np
import pandas as pd
import pickle
import tensorflow
from tensorflow import keras
from keras.layers import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from config import *
from data_cleaning import clean_text_column

# -------------------------------------------------------------------------
#                           Utility Functions
# -------------------------------------------------------------------------
def sum_of_columns(dataframe, columns):
    temp = 0
    for col in columns:
        temp += dataframe[col]
    return temp

class DataPreprocess:

    def __init__(self, data, do_load_existing_tokenizer=True):
        
        self.data = data
        self.doLoadExistingTokenizer = do_load_existing_tokenizer
        # -------------------------------------------------------------------------
        embeddings_index_fasttext = {}
        with open(EMBEDDING_FILE_LOC,encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                embeddings_index_fasttext[word] = np.asarray(values[1:], dtype='float32')
                
        print(f'Total of {len(embeddings_index_fasttext)} word vectors are found.')
        # -------------------------------------------------------------------------
        cols = DETECTION_CLASSES.copy()
        cols.remove('neutral')
        data['neutral'] = np.where(sum_of_columns(data, cols) > 0, 0, 1)
        # -------------------------------------------------------------------------
        data['comment_text'] = clean_text_column(data['comment_text'])
        print("Data Cleaned")
        # -------------------------------------------------------------------------
        processed_train_data = data['comment_text'].values
        self.target_classes = data[DETECTION_CLASSES].values
        print("Data Assigned to Variable X and y!")
        # -------------------------------------------------------------------------
        if not do_load_existing_tokenizer:
            # Convert the comments (strings) into integers
            tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
            tokenizer.fit_on_texts(processed_train_data)
        else:
            with open(TOKENIZER_LOC, 'rb') as handle:
                tokenizer = pickle.load(handle)
        print('Data Tokenized-1')
        # -------------------------------------------------------------------------
        list_tokenized_train = tokenizer.texts_to_sequences(processed_train_data)
        print('Data Tokenized-2')
        # -------------------------------------------------------------------------
        word_index = tokenizer.word_index
        print(f'Found {len(word_index)} unique tokens')
        # -------------------------------------------------------------------------
        if not do_load_existing_tokenizer:
            # Save tokenizer
            print('Saving tokens ...')
            with open(TOKENIZER_LOC, 'wb') as handle:
                pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # -------------------------------------------------------------------------
        self.X_t=pad_sequences(list_tokenized_train, maxlen=MAX_SEQUENCE_LENGTH, padding = 'post')
        print('Shape of Data Tensor:', self.X_t)
        # -------------------------------------------------------------------------
        embedding_matrix_fasttext = np.random.random((len(word_index) + 1, EMBEDDING_DIMENSION))
        for word, i in word_index.items():
            embedding_vector = embeddings_index_fasttext.get(word)
            if embedding_vector is not None:
                embedding_matrix_fasttext[i] = embedding_vector
        print(" Completed! Embedding Matrix")
        # -------------------------------------------------------------------------
        self.embedding_layer = Embedding(len(word_index) + 1,
                                         EMBEDDING_DIMENSION,
                                         weights = [embedding_matrix_fasttext],
                                         input_length = MAX_SEQUENCE_LENGTH,
                                         trainable=False,
                                         name = 'embeddings')
        print("Embedding Layer Created!")
        # -------------------------------------------------------------------------
                
#training_data = pd.read_csv(TRAINING_DATA_LOC)
#print(DataPreprocess(training_data,False))
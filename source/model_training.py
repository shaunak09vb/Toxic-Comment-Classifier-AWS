import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from keras.layers import Dense, Input, LSTM, Dropout
from keras.layers import GlobalMaxPool1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import roc_auc_score

from config import *
from data_preprocessing import DataPreprocess
# -------------------------------------------------------------------------

def build_lstm_model(data, target_classes, embedding_layer):
    
    inp=Input(shape=(MAX_SEQUENCE_LENGTH, ),dtype='int32')
    embedded_sequences = embedding_layer(inp)
    x = LSTM(40, return_sequences=True,name='lstm_layer')(embedded_sequences)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(30, activation="relu", kernel_initializer='he_uniform')(x)
    x = Dropout(0.1)(x)
    preds = Dense(7, activation="sigmoid", kernel_initializer='glorot_uniform')(x)
    # -------------------------------------------------------------------------
    model_1 = Model(inputs=inp, outputs=preds)
    model_1.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    # -------------------------------------------------------------------------
    model_1.summary()
    # -------------------------------------------------------------------------
    checkpoint = ModelCheckpoint(filepath=MODEL_LOCATION,  # saves the 'best' model
                                  monitor='val_loss',
                                  save_best_only=True,
                                  mode='min',
                                  save_weights_only=True)
    # -------------------------------------------------------------------------
    history = model_1.fit(data,
                        target_classes,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_split=VALIDATION_SPLIT,
                        verbose=1)
    # -------------------------------------------------------------------------
    model_1.save(MODEL_LOCATION)
    # -------------------------------------------------------------------------
    return model_1, history
# -------------------------------------------------------------------------
def plot_training_history(lstm_model, history, data, target_classes):
    # "Accuracy"
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('../accuracy.jpeg')
    plt.show()

    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('../loss.jpeg')
    plt.show()

    # Print Average ROC_AUC_Score
    p = lstm_model.predict(data)
    aucs = []
    for j in range(len(DETECTION_CLASSES)):
        auc = roc_auc_score(target_classes[:, j], p[:, j])
        aucs.append(auc)
    print(f'Average ROC_AUC Score: {np.mean(aucs)}')


@click.command()
@click.option('--data', default=TRAINING_DATA_LOCATION, help="Training Data Location")
def execute(data):
    training_data = pd.read_csv(data)

    preprocessing = DataPreprocess(training_data)
    lstm_model, history = build_lstm_model(preprocessing.X_t,
                                         preprocessing.target_classes,
                                         preprocessing.embedding_layer)
    plot_training_history(lstm_model, 
                          history,
                          preprocessing.X_t,
                          preprocessing.target_classes)
# -------------------------------------------------------------------------
if __name__ == '__main__':
    execute()
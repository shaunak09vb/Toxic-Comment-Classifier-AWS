EMBEDDING_DIMENSION = 300
EMBEDDING_FILE_LOCATION = '../model/wiki-news-300d-1M.vec'
TRAINING_DATA_LOCATION = '../data/train.csv'
TEST_DATA_LABEL = '../data/test_labels.csv'
TEST_DATA_COMMENTS = '../data/test.csv'
MAX_VOCAB_SIZE = 100000
MAX_SEQUENCE_LENGTH = 200
BATCH_SIZE = 32
EPOCHS = 2
VALIDATION_SPLIT = 0.2
DETECTION_CLASSES = [
    'toxic',
    'severe_toxic',
    'obscene',
    'threat',
    'insult',
    'identity_hate',
    'neutral']

MODEL_LOCATION = '../model/toxicity_classifier.h5'
TOKENIZER_LOCATION = '../model/tokenizer.pickle'

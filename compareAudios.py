import random

import numpy as np

from funs.audio import read_mfcc
from funs.batcher import sample_from_mfcc
from funs.constants import SAMPLE_RATE, NUM_FRAMES
from funs.conv_models import DeepSpeakerModel
from funs.test import batch_cosine_similarity

# Reproducible results.
np.random.seed(13)
random.seed(13)

# Define the model here.
model = DeepSpeakerModel()

# Load the checkpoint.
model.m.load_weights('ResCNN_triplet_training_checkpoint_265.h5', by_name=True)

# Sample some inputs for WAV/FLAC files for the same speaker.
# To have reproducible results every time you call this function, set the seed every time before calling it.
# np.random.seed(123)
# random.seed(123)
mfcc_001 = sample_from_mfcc(read_mfcc('samples/yo1.wav', SAMPLE_RATE), NUM_FRAMES)
mfcc_002 = sample_from_mfcc(read_mfcc('samples/yo2.wav', SAMPLE_RATE), NUM_FRAMES)

# Call the model to get the embeddings of shape (1, 512) for each file.
predict_001 = model.m.predict(np.expand_dims(mfcc_001, axis=0))
predict_002 = model.m.predict(np.expand_dims(mfcc_002, axis=0))

# Do it again with a different speaker.
mfcc_003 = sample_from_mfcc(read_mfcc('samples/mama1.wav', SAMPLE_RATE), NUM_FRAMES)
predict_003 = model.m.predict(np.expand_dims(mfcc_003, axis=0))

# Compute the cosine similarity and check that it is higher for the same speaker.
print('SAME SPEAKER', batch_cosine_similarity(predict_001, predict_002)) 
print('DIFF SPEAKER', batch_cosine_similarity(predict_001, predict_003)) 
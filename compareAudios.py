import random

import numpy as np

from funs.audio import read_mfcc
from funs.batcher import sample_from_mfcc
from funs.constants import SAMPLE_RATE, NUM_FRAMES
from funs.conv_models import DeepSpeakerModel
from funs.test import batch_cosine_similarity
from os.path import join

# Define the model here.
model = DeepSpeakerModel()
# Load the checkpoint.
model.m.load_weights('ResCNN_triplet_training_checkpoint_265.h5', by_name=True)
def compare(aud1, aud2, verbose=False):
    # Sample some inputs for WAV/FLAC files for the same speaker.
    # To have reproducible results every time you call this function, set the seed every time before calling it.
    mfcc_001 = sample_from_mfcc(read_mfcc(aud1, SAMPLE_RATE), NUM_FRAMES)
    mfcc_002 = sample_from_mfcc(read_mfcc(aud2, SAMPLE_RATE), NUM_FRAMES)

    # Call the model to get the embeddings of shape (1, 512) for each file.
    predict_001 = model.m.predict(np.expand_dims(mfcc_001, axis=0))
    predict_002 = model.m.predict(np.expand_dims(mfcc_002, axis=0))

    # Compute the cosine similarity and check that it is higher for the same speaker.
    res = batch_cosine_similarity(predict_001, predict_002)[0]
    if verbose:
        if res<0.7:
            print('Result: {:.3f}, they are different persons'.format(res)) 
        else:
            print('Result: {:.3f}, they are the same person'.format(res)) 

    return res

persons = ['yo', 'mama', 'erick', 'juan', 'camilo', 'isa']
results = []
isPositive = []
for person in persons:
    print(person)
    nam1 = person+'1.wav'
    nam2 = person+'2.wav'
    path1 = join('samples',nam1)
    path2 = join('samples',nam2)
    print("Positive Examples:")
    res = compare(path1, path2, verbose=True)
    results.append(res)
    isPositive.append(1)

    newPersons = persons.copy()
    newPersons.remove(person)
    print("Negative Examples:")
    for otherPerson in newPersons:
        print(otherPerson)
        nam3 = otherPerson+'1.wav'
        nam4 = otherPerson+'2.wav'
        path3 = join('samples',nam3)
        path4 = join('samples',nam4)

        res1 = compare(path1, path3, verbose=True)
        results.append(res1)
        isPositive.append(0)

        res2 = compare(path1, path4, verbose=True)
        results.append(res2)
        isPositive.append(0)

        res3 = compare(path2, path3, verbose=True)
        results.append(res3)
        isPositive.append(0)

        res4 = compare(path2, path4, verbose=True)
        results.append(res4)
        isPositive.append(0)
    print()
    

    
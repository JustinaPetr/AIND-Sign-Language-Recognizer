import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    Xlengths = test_set.get_all_Xlengths()
    for X, l in Xlengths.values():
        log_l = {}
        max_score = float("-inf")
        best_guess = None
        for word, model in models.items():
            try:
                score = model.score(X, l)
                log_l[word] = score

                if score > max_score:
                    max_score = score
                    best_guess = word
            except:
                log_l[word] = float("-inf")

        guesses.append(best_guess)
        probabilities.append(log_l)

    return probabilities, guesses
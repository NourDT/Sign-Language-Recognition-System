import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None

class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)

class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        best_score = float("inf")
        best_model = None
        num_features = self.X.shape[1]

        #L is the likelihood of the fitted model, params is the number of parameters, and N is the number of data points
        for i_components in range(self.min_n_components, self.max_n_components + 1):
            try:
                new_model = self.base_model(i_components)
                logL = new_model.score(self.X, self.lengths)
                logN = np.log(len(self.X))

                # details about the logic behind this was found here:
                #https://discussions.udacity.com/t/number-of-parameters-bic-calculation/233235/4

                # Initial state occupation probabilities = numStates
                initial_state_occupation_probabilities = i_components

                # Transition probabilities = numStates*(numStates - 1)
                transition_probabilities = i_components * (i_components -1)

                # Emission probabilities = numStates*numFeatures*2 = numMeans+numCovars
                emission_probabilities = i_components * num_features * 2

                # params is the number of model parameters in the test.
                # Parameters = Initial state occupation probabilities + Transition probabilities + Emission probabilities
                params = initial_state_occupation_probabilities + transition_probabilities + emission_probabilities

                # Calculate the score
                new_score = -2 * logL + params * logN
                # Only keep the min value
                if new_score < best_score:
                    best_score, best_model = new_score,new_model

            except:
                pass

        return best_model

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    Need to generate, do, and/or extract the following:
    - log likelihood for the ith worth in array X
        - sum of the log likelihoods for all words in array X except for the ith word
        - divide the sum from the step above by columns - 1
    - subtract this from the value in the first step
    '''


    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # details about the logic behind this was found here:
        # https://discussions.udacity.com/t/discriminative-information-criterion-formula-words-vs-parameters/250880

        # initialize essential objects
        best_model = None
        # initialize at lowest possible number
        best_score= float("-inf")

        # outer loop iterating over components
        for component_num in range(self.min_n_components, self.max_n_components + 1):
            words_left_scores= list()

            try:
                model= self.base_model(component_num)

                # log(P(X(i)) for this word
                score= model.score(self.X, self.lengths)

                # get all the words as dict with words as keys
                words= self.words

                # generate a dict of all words except ith word
                # copy the dict so we don't alter words dict
                words_left = words.copy()

                # remove this word
                words_left.pop(self.this_word)

                # iterate over all the other words and sum up P (logL)
                for word in words_left:
                    # hwords is a dict with values of X and length for each key (word)
                    X, lengths= self.hwords[word]
                    try:
                        # log(P(X(i)) for this word
                        words_left_scores.append(model.score(X, lengths))
                    except:
                        pass

                # put it all together
                M = len(words_left)
                words_left_score = np.sum(words_left_scores)

                # update best score
                DIC = score - words_left_score
                if DIC > best_score:
                    best_score= DIC
                    best_model= model
            except:
                pass
        return best_model

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    # Logic behind this
    # https://www.cs.cmu.edu/~schneide/tut5/node42.html
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        n_splits = 3
        if len(self.sequences) < 2:
            return None
        elif len(self.sequences) == 2:
            n_splits = 2
        split_method = KFold(n_splits=n_splits)

        best_score = float("-inf")
        best_model = None
        for i_components in range(self.min_n_components, self.max_n_components + 1):
            partial_score = 0
            i = 0
            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                new_model = None
                try:
                    # combine sequences with the train split
                    train_X, train_lengths = combine_sequences(cv_train_idx, self.sequences)

                    # I create the HMM model
                    new_model = GaussianHMM(n_components=i_components, covariance_type="diag", n_iter=1000,
                                            random_state=self.random_state, verbose=False).fit(train_X, train_lengths)
                    # Now I add the test data
                    test_X, test_lengths = combine_sequences(cv_test_idx, self.sequences)
                    # I calculate score with the added test data
                    partial_score += hmm_model.score(test_X, test_lengths)
                    i += 1
                except:
                    pass
            if i > 0:
                # Calculate average
                new_score = partial_score / i
            else:
                new_score = 0
            if new_score > best_score:
                best_score = new_score
                best_model = new_model

        return best_model

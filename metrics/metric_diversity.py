import string

import torch
from nltk import ngrams
from sentence_transformers import SentenceTransformer

from utils import contractions_dict, expand_contractions


class PINCscore:
    def __init__(self, max_n_gram):
        self.max_n_gram = max_n_gram

    def ngram(self, document, max_n_gram):
        ngrams_list = []
        for i in range(1, max_n_gram + 1):
            splitted = ngrams(document.split(), i)
            ngrams_list.append(set(splitted))
        return ngrams_list

    def preprocess(self, text):
        # helper funfction for preprocessing text
        pre_processed_text = []
        for i in range(len(text)):
            expanded_text = (expand_contractions(text[i], contractions_dict)).lower()
            no_punc_text = expanded_text.translate(str.maketrans("", "", string.punctuation))
            pre_processed_text.append(no_punc_text)
        return pre_processed_text

    def score(self, contexts, questions, answers, lengths=None, extra=None):
        """
        The score function returns the PINC score for two documents.
        With a maximum_lengths constraint, the function tokenizes the two
        document and measure the level of similarity  between them.
        The original implementation can be found here:
        https://www.cs.utexas.edu/~ml/papers/chen.acl11.pdf
        """
        pre_processed_contexts = self.preprocess(contexts)
        pre_processed_questions = self.preprocess(questions)

        PINC_score_list = []
        for i in range(len(pre_processed_questions)):
            # the N in the N-gram tokenization cannot exceed the number of words in the document
            max_n_gram = min(
                len(pre_processed_questions[i].split()), len(pre_processed_contexts[i].split()), self.max_n_gram
            )

            # if question is blank, then score is 0
            if max_n_gram == 0:
                PINC_score_list.append(0)
                continue

            context_ngram_list = self.ngram(pre_processed_contexts[i], max_n_gram)
            question_ngram_list = self.ngram(pre_processed_questions[i], max_n_gram)
            # we tokenize the groundtruth document and the prediction sentences
            # and create a 1-D array which contains all the n grams, where n ranges
            # from 1 to N
            PINC_score = 0
            for j in range(max_n_gram):
                overlap_count = 0
                for elem in question_ngram_list[j]:
                    if elem in context_ngram_list[j]:
                        overlap_count += 1
                PINC_score += 1 - overlap_count / len(question_ngram_list[j])
            PINC_score *= 1 / max_n_gram
            PINC_score_list.append(PINC_score)
        return PINC_score_list

    def score_two_questions(self, question_ones, question_twos, lengths=None, extra=None):
        """
        The PINC scoring function specifically for two question generation.
        Instead of evaluating the level of similarity betweena context and the
        generated questions. This function instead evaluates the level of similarity
        between the two sets of generated functions
        """
        assert len(question_ones) == len(question_twos), "The number of questions must be equal"
        pre_processed_first_questions = self.preprocess(question_ones)
        pre_processed_second_questions = self.preprocess(question_twos)

        PINC_score_list = []
        for i in range(len(pre_processed_second_questions)):
            # the N in the N-gram tokenization cannot exceed the number of words in the document
            max_n_gram = min(
                len(pre_processed_second_questions[i].split()),
                len(pre_processed_first_questions[i].split()),
                self.max_n_gram,
            )

            # if question is blank, then score is 0
            if max_n_gram == 0:
                PINC_score_list.append(0)
                continue

            question_ones_ngram_list = self.ngram(pre_processed_first_questions[i], max_n_gram)
            question_twos_ngram_list = self.ngram(pre_processed_second_questions[i], max_n_gram)
            # we tokenize the groundtruth document and the prediction sentences
            # and create a 1-D array which contains all the n grams, where n ranges
            # from 1 to N
            PINC_score = 0
            # Question2 -> Question 1 PINC score
            PINC_score_reverse = 0
            # Question1 -> Question 2 PINC score
            for j in range(max_n_gram):
                overlap_count = 0
                overlap_count_reverse = 0
                for elem in question_twos_ngram_list[j]:
                    if elem in question_ones_ngram_list[j]:
                        overlap_count += 1
                for elem in question_ones_ngram_list[j]:
                    if elem in question_twos_ngram_list[j]:
                        overlap_count_reverse += 1
                PINC_score += 1 - overlap_count / len(question_twos_ngram_list[j])
                PINC_score_reverse += 1 - overlap_count_reverse / len(question_ones_ngram_list[j])
            PINC_score *= 1 / max_n_gram
            PINC_score_reverse *= 1 / max_n_gram
            PINC_score_list.append((PINC_score + PINC_score_reverse) / 2)
        return PINC_score_list


class SBERTScore:
    def __init__(self, model_name="all-mpnet-base-v2", device="cpu"):
        self.model = SentenceTransformer(model_name)
        self.device = device
        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    def score_two_questions(self, question_ones, question_twos, lengths=None, extra=None):
        """
        Returns the cosine similarity between questions using SBERT embeddings
        """
        embeddings_one = self.model.encode(question_ones, convert_to_tensor=True, device=self.device)
        embeddings_two = self.model.encode(question_twos, convert_to_tensor=True, device=self.device)
        cosine_similarities = self.cos(embeddings_one, embeddings_two)
        return cosine_similarities.cpu().numpy().tolist()

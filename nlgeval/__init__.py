from __future__ import print_function

import six
import numpy as np
from six.moves import map
from collections import defaultdict

from nlgeval.pycocoevalcap.bleu.bleu import Bleu
from nlgeval.pycocoevalcap.cider.cider import Cider
from nlgeval.pycocoevalcap.meteor.meteor import Meteor
from nlgeval.pycocoevalcap.rouge.rouge import Rouge


def _strip(s):
    return s.strip()


def compute_metrics(hyp_list, ref_list, no_overlap=False):
    
    refs = {idx: lines.strip().split('\t') for (idx, lines) in enumerate(ref_list)}
    hyps = {idx: [lines.strip()] for (idx, lines) in enumerate(hyp_list)}
    assert len(refs) == len(hyps)

    ret_scores = {}
    if not no_overlap:
        scorers = [
            (Bleu(4), ["bleu_1", "bleu_2", "bleu_3", "bleu_4"]),
            # (Meteor(), "meteor"),
            (Rouge(), "rouge_l"),
            # (Cider(), "cider")
        ]
        for scorer, method in scorers:
            score, scores = scorer.compute_score(refs, hyps)
            if isinstance(method, list):
                for sc, scs, m in zip(score, scores, method):
                    ret_scores[m] = sc
            else:
                ret_scores[method] = score
            if isinstance(scorer, Meteor):
                scorer.close()
        del scorers

    return ret_scores


def compute_corpus_metrics(ref_list, hyp_list, no_overlap=False):

    corpus_metrics = defaultdict(list)
    for ref, hyp in zip(ref_list, hyp_list):
        metrics = compute_individual_metrics(ref, hyp, no_overlap=no_overlap)
        for k, v in metrics.items():
            corpus_metrics[k].append(v)
    
    corpus_metrics = {k: np.mean(v) for k, v in corpus_metrics.items()}

    return corpus_metrics


def compute_individual_metrics(ref, hyp, no_overlap=False):
    assert isinstance(hyp, six.string_types)

    if isinstance(ref, six.string_types):
        ref = ref.split('\t')  # special delimiter for backward compatibility
    refs = [_strip(a) for a in ref]

    ret_scores = {}
    if not no_overlap:
        scorers = [
            (Bleu(4), ["bleu_1", "bleu_2", "bleu_3", "bleu_4"]),
            # (Meteor(), "meteor"),
            (Rouge(), "rouge_l"),
            # (Cider(), "cider")
        ]

        for scorer, method in scorers:
            score, scores = scorer.compute_score({0: refs}, {0: [_strip(hyp)]})
            if isinstance(method, list):
                for sc, _, m in zip(score, scores, method):
                    ret_scores[m] = sc
            else:
                ret_scores[method] = score
            if isinstance(scorer, Meteor):
                scorer.close()
        del scorers

    return ret_scores


class NLGEval(object):

    valid_metrics = {'bleu_1', 'bleu_2', 'bleu_3', 'bleu_4', 'meteor', 'rouge_l', 'cider',}

    def __init__(self, no_overlap=False, metrics_to_omit=None):

        if metrics_to_omit is None:
            self.metrics_to_omit = set()
        else:
            self.metrics_to_omit = set(metrics_to_omit)
            # For backwards compatibility.
            if 'EmbeddingAverageCosineSimilairty' in self.metrics_to_omit:
                self.metrics_to_omit.remove('EmbeddingAverageCosineSimilairty')
                self.metrics_to_omit.add('EmbeddingAverageCosineSimilarity')

        assert len(self.metrics_to_omit - self.valid_metrics) == 0, \
            "Invalid metrics to omit: {}".format(self.metrics_to_omit - self.valid_metrics)

        self.no_overlap = no_overlap
        if not no_overlap:
            self.load_scorers()

    def load_scorers(self):
        self.scorers = []

        omit_bleu_i = False
        for i in range(1, 4 + 1):
            if 'Bleu_{}'.format(i) in self.metrics_to_omit:
                omit_bleu_i = True
                if i > 1:
                    self.scorers.append((Bleu(i - 1), ['bleu_{}'.format(j) for j in range(1, i)]))
                break
        if not omit_bleu_i:
            self.scorers.append((Bleu(4), ["bleu_1", "bleu_2", "bleu_3", "bleu_4"]))

        if 'meteor' not in self.metrics_to_omit:
            self.scorers.append((Meteor(), "meteor"))
        if 'rouge_l' not in self.metrics_to_omit:
            self.scorers.append((Rouge(), "rouge_l"))
        if 'cider' not in self.metrics_to_omit:
            self.scorers.append((Cider(), "cider"))

    def compute_individual_metrics(self, ref, hyp):
        assert isinstance(hyp, six.string_types)
        refs = [a.strip() for a in ref]

        ret_scores = {}
        if not self.no_overlap:
            for scorer, method in self.scorers:
                score, scores = scorer.compute_score({0: refs}, {0: [hyp.strip()]})
                if isinstance(method, list):
                    for sc, _, m in zip(score, scores, method):
                        ret_scores[m] = sc
                else:
                    ret_scores[method] = score

        return ret_scores

    def compute_metrics(self, ref_list, hyp_list):
        ref_list = [list(map(_strip, refs)) for refs in zip(*ref_list)]
        refs = {idx: strippedlines for (idx, strippedlines) in enumerate(ref_list)}
        hyps = {idx: [lines.strip()] for (idx, lines) in enumerate(hyp_list)}
        assert len(refs) == len(hyps)

        ret_scores = {}
        if not self.no_overlap:
            for scorer, method in self.scorers:
                score, scores = scorer.compute_score(refs, hyps)
                if isinstance(method, list):
                    for sc, _, m in zip(score, scores, method):
                        ret_scores[m] = sc
                else:
                    ret_scores[method] = score

        return ret_scores

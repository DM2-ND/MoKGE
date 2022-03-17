import os
import numpy as np
from collections import defaultdict

from nlgeval import compute_metrics
from nlgeval import compute_individual_metrics as compute_individual

def get_all_files(path):

    if os.path.isfile(path): return [path]

    return [f for d in os.listdir(path)
              for f in get_all_files(os.path.join(path, d))]
    

def eval_top1_acc(hyp_path, ref_path, step):

    with open(hyp_path, 'r') as hyp_file, open(ref_path, 'r') as ref_file:
        hyps = hyp_file.readlines()
        refs = ref_file.readlines()
        
        # hyp_top1 and refs are both a list of string
        # E.g., [hyp1 for data sample 1, hyp2 for ...]
        hyps_top1 = [hyps[i] for i in range(0, len(hyps), step)]

        top1_metrics = compute_metrics(hyp_list=hyps_top1, ref_list=refs)
        top1_metrics = {f'top1_{k}': v for k, v in top1_metrics.items()}

        return top1_metrics


def eval_topk_acc(hyp_path, ref_path, step):

    with open(hyp_path, 'r') as hyp_file, open(ref_path, 'r') as ref_file:
        hyps = hyp_file.readlines()
        refs = ref_file.readlines()

        hyps_topk = [hyps[i: i+step] for i in range(0, len(hyps), step)]
        hyps_best = []
        for hyp, ref in zip(hyps_topk, refs):
            hyp_score_list = [compute_individual(ref, h)['bleu_4'] for h in hyp]
            hyps_best.append(hyp[np.argmax(hyp_score_list)])
        
        topk_metrics = compute_metrics(ref_list=refs, hyp_list=hyps_best)
        topk_metrics = {f'topk_{k}': v for k, v in topk_metrics.items()}

        return topk_metrics

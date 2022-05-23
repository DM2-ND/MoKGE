import sys
import json
from tqdm import tqdm
import configparser
from collections import Counter

config = configparser.ConfigParser()
config.read("paths.cfg")

def read_json(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data

def save_json(data, filename):
    with open(filename, 'w') as f:
        for line in data:
            json.dump(line, f)
            f.write('\n')

def filter_directed_triple(data, max_concepts=200, max_triples=300):
    _data, _concepts = [], []
    max_len, max_neighbors = 0, 5
    for e in tqdm(data):
        triple_dict = {}
        triples = e['triples']
        concepts = e['concepts']
        labels = e['labels']
        distances = e['distances']
        _concepts += e['concepts']

        for t in triples:
            head, tail = t[0], t[-1]
            head_id = concepts.index(head)
            tail_id = concepts.index(tail)
            if distances[head_id] <= distances[tail_id]:
                if t[-1] not in triple_dict:
                    triple_dict[t[-1]] = [t]
                else:
                    if len(triple_dict[t[-1]]) < max_neighbors:
                        triple_dict[t[-1]].append(t)

        starts = []
        for l, c in zip(labels, concepts):
            if l == 1:
                starts.append(c)
        sources = []
        for d, c in zip(distances, concepts):
            if d == 0:
                sources.append(c)

        shortest_paths = []
        for start in starts:
            shortest_paths.extend(bfs(start, triple_dict, sources))
        ground_truth_triples = []
        for path in shortest_paths:
            for i, n in enumerate(path[:-1]):
                ground_truth_triples.append((n, path[i+1]))

        ground_truth_triples_set = set(ground_truth_triples)

        _triples = []
        triple_labels = []
        for _, v in triple_dict.items():
            for t in v:
                _triples.append(t)
                if (t[-1], t[0]) in ground_truth_triples_set:
                    triple_labels.append(1)
                else:
                    triple_labels.append(0)
        
        concepts = concepts[:max_concepts]
        _triples = _triples[:max_triples]
        triple_labels = triple_labels[:max_triples]
        
        heads, tails = [], []
        for triple in _triples:
            heads.append(concepts.index(triple[0]))
            tails.append(concepts.index(triple[-1]))
    
        max_len = max(max_len, len(_triples))
        e['relations'] = [x[1] for x in _triples]
        e['head_ids'] = heads
        e['tail_ids'] = tails
        e['triple_labels'] = triple_labels
        e.pop('triples')
        
        _data.append(e)

    return _data, _concepts

def bfs(start, triple_dict, source):
    paths = [[[start]]]
    shortest_paths = []
    count = 0
    while 1:
        last_paths = paths[-1]
        new_paths = []
        for path in last_paths:
            if triple_dict.get(path[-1], False):
                triples = triple_dict[path[-1]]
                for triple in triples:
                    new_paths.append(path + [triple[0]])

        for path in new_paths:
            if path[-1] in source:
                shortest_paths.append(path)
        
        if count == 2:
            break
        paths.append(new_paths)
        count += 1

    return shortest_paths


if __name__ == "__main__":
    dataset = sys.argv[1]

    DATA_PATH = config["paths"][dataset + "_dir"]
    T, max_B = 2, 100

    total_concepts = []
    for TYPE in ['train', 'val', 'test']:
        data = read_json(DATA_PATH + '/{}.{}hops_{}_triple.json'.format(TYPE, T, max_B))
        f_data, f_concepts = filter_directed_triple(data, max_concepts=400, max_triples=1000)
        total_concepts += f_concepts
        save_json(f_data, DATA_PATH + '/{}.kg.json'.format(TYPE, T, max_B))

    words_by_frequency = sorted(Counter(total_concepts).items(), key=lambda x: x[1], reverse=True)
    print('total word counts: ', len(words_by_frequency))
    with open(DATA_PATH + '/kg_vocab.txt', 'w') as vocab_file:
        for word, frequency in words_by_frequency:
            vocab_file.write('{} {}\n'.format(word, frequency))
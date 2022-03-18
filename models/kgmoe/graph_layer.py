import torch
import torch.nn as nn
from torch_scatter import scatter_max, scatter_mean, scatter_add

class GraphEncoder(nn.Module):
    def __init__(self, embed_size, gamma=0.8, alpha=1, beta=1, aggregate_method="max", tokenizer=None, hop_number=2):
        super(GraphEncoder, self).__init__()

        self.hop_number = hop_number
        
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.aggregate_method = aggregate_method
        self.tokenizer = tokenizer

        self.relation_embed = nn.Embedding(50, embed_size, padding_idx=0)
        
        self.triple_linear = nn.Linear(embed_size * 3, embed_size, bias=False)
        
        self.W_s = nn.ModuleList([nn.Linear(embed_size, embed_size, bias=False) for _ in range(self.hop_number)]) 
        self.W_n = nn.ModuleList([nn.Linear(embed_size, embed_size, bias=False) for _ in range(self.hop_number)])
        self.W_r = nn.ModuleList([nn.Linear(embed_size, embed_size, bias=False) for _ in range(self.hop_number)])
        self.gate_linear = nn.Linear(embed_size, 1)

    def multi_layer_comp_gcn(self, concept_hidden, relation_hidden, head, tail, triple_label, layer_number=2):
        for i in range(layer_number):
            concept_hidden, relation_hidden = self.comp_gcn(concept_hidden, relation_hidden, head, tail, triple_label, i)
        return concept_hidden, relation_hidden

    def comp_gcn(self, concept_hidden, relation_hidden, head, tail, triple_label, layer_idx):
        '''
        concept_hidden: bsz x mem x hidden
        relation_hidden: bsz x mem_t x hidden
        '''
        bsz = head.size(0)
        mem_t = head.size(1)
        mem = concept_hidden.size(1)
        hidden_size = concept_hidden.size(2)

        update_node = torch.zeros_like(concept_hidden).to(concept_hidden.device).float()
        count = torch.ones_like(head).to(head.device).masked_fill_(triple_label == -1, 0).float()
        count_out = torch.zeros(bsz, mem).to(head.device).float()

        o = concept_hidden.gather(1, head.unsqueeze(2).expand(bsz, mem_t, hidden_size))
        o = o.masked_fill(triple_label.unsqueeze(2) == -1, 0)

        scatter_add(o, tail, dim=1, out=update_node)
        scatter_add( - relation_hidden.masked_fill(triple_label.unsqueeze(2) == -1, 0), tail, dim=1, out=update_node)
        scatter_add(count, tail, dim=1, out=count_out)

        o = concept_hidden.gather(1, tail.unsqueeze(2).expand(bsz, mem_t, hidden_size))
        o = o.masked_fill(triple_label.unsqueeze(2) == -1, 0)
        scatter_add(o, head, dim=1, out=update_node)
        scatter_add( - relation_hidden.masked_fill(triple_label.unsqueeze(2) == -1, 0), head, dim=1, out=update_node)
        scatter_add(count, head, dim=1, out=count_out)

        act = nn.ReLU()
        update_node = self.W_s[layer_idx](concept_hidden) + self.W_n[layer_idx](update_node) / count_out.clamp(min=1).unsqueeze(2)
        update_node = act(update_node)

        return update_node, self.W_r[layer_idx](relation_hidden)

    def multi_layer_gcn(self, concept_hidden, head, tail, triple_label, layer_number=2):
        for i in range(layer_number):
            concept_hidden = self.gcn(concept_hidden, head, tail, triple_label, i)
        return concept_hidden

    def gcn(self, concept_hidden, head, tail, triple_label, layer_idx):
        '''
        concept_hidden: bsz x mem x hidden
        '''
        bsz = head.size(0)
        mem_t = head.size(1)
        mem = concept_hidden.size(1)
        hidden_size = concept_hidden.size(2)
        update_hidden = torch.zeros_like(concept_hidden).to(concept_hidden.device).float()
        count = torch.ones_like(head).to(head.device).masked_fill_(triple_label == -1, 0).float()
        count_out = torch.zeros(bsz, mem).to(head.device).float()

        o = concept_hidden.gather(1, head.unsqueeze(2).expand(bsz, mem_t, hidden_size))
        o = o.masked_fill(triple_label.unsqueeze(2) == -1, 0)
        scatter_add(o, tail, dim=1, out=update_hidden)
        scatter_add(count, tail, dim=1, out=count_out)

        o = concept_hidden.gather(1, tail.unsqueeze(2).expand(bsz, mem_t, hidden_size))
        o = o.masked_fill(triple_label.unsqueeze(2) == -1, 0)
        scatter_add(o, head, dim=1, out=update_hidden)
        scatter_add(count, head, dim=1, out=count_out)

        act = nn.ReLU()
        update_hidden = self.W_s[layer_idx](concept_hidden) + self.W_n[layer_idx](update_hidden) / count_out.clamp(min=1).unsqueeze(2)
        update_hidden = act(update_hidden)

        return update_hidden

    def multi_hop(self, triple_prob, distance, head, tail, concept_label, triple_label, gamma=0.8, iteration = 3, method="avg"):
        '''
        triple_prob: bsz x L x mem_t
        distance: bsz x mem
        head, tail: bsz x mem_t
        concept_label: bsz x mem
        triple_label: bsz x mem_t

        Init binary vector with source concept == 1 and others 0
        expand to size: bsz x L x mem
        '''
        concept_probs = []
        cpt_size = (triple_prob.size(0), triple_prob.size(1), distance.size(1))
        init_mask = torch.zeros_like(distance).unsqueeze(1).expand(*cpt_size).to(distance.device).float()
        init_mask.masked_fill_((distance == 0).unsqueeze(1), 1)
        final_mask = init_mask.clone()

        init_mask.masked_fill_((concept_label == -1).unsqueeze(1), 0)
        concept_probs.append(init_mask)

        head = head.unsqueeze(1).expand(triple_prob.size(0), triple_prob.size(1), -1)
        tail = tail.unsqueeze(1).expand(triple_prob.size(0), triple_prob.size(1), -1)
        for _ in range(iteration):
            ''' Calculate triple head score '''
            node_score = concept_probs[-1]
            triple_head_score = node_score.gather(2, head)
            triple_head_score.masked_fill_((triple_label == -1).unsqueeze(1), 0)
            # Method: 
            # - avg: s(v) = Avg_{u \in N(v)} gamma * s(u) + R(u->v) 
            # - max: s(v) = max_{u \in N(v)} gamma * s(u) + R(u->v)
            update_value = triple_head_score * gamma + triple_prob
            out = torch.zeros_like(node_score).to(node_score.device).float()
            if method == "max":
                scatter_max(update_value, tail, dim=-1, out=out)
            elif method == "avg":
                scatter_mean(update_value, tail, dim=-1, out=out)
            out.masked_fill_((concept_label == -1).unsqueeze(1), 0)
            
            concept_probs.append(out)
        
        ''' Natural decay of concept that is multi-hop away from source '''
        total_concept_prob = final_mask * -1e5
        for prob in concept_probs[1:]:
            total_concept_prob += prob
        # bsz x L x mem
        return total_concept_prob

    def forward(self, concept_ids, distance, head, tail, relation, triple_label, mixture_ids=None):
        
        memory = self.embed_word(concept_ids)
        rel_repr = self.relation_embed(relation)

        if mixture_ids is not None:
            mixture_embed = self.mixture_embed(mixture_ids)
            memory = memory + 1.0 * mixture_embed

        node_repr, rel_repr = self.multi_layer_comp_gcn(memory, rel_repr, head, tail, triple_label, layer_number=self.hop_number)
        head_repr = torch.gather(node_repr, 1, head.unsqueeze(-1).expand(node_repr.size(0), head.size(1), node_repr.size(-1)))
        tail_repr = torch.gather(node_repr, 1, tail.unsqueeze(-1).expand(node_repr.size(0), tail.size(1), node_repr.size(-1)))
        
        # bsz x mem_triple x hidden
        triple_repr = torch.cat((head_repr, rel_repr, tail_repr), dim=-1)

        return node_repr, triple_repr

    def generate(self, src_input_ids, attention_mask, src_position_ids, 
                    concept_ids, concept_label, distance, 
                    head, tail, relation, triple_label,
                    vocab_map, map_mask,
                    seq_generator):

        memory = self.word_embed(concept_ids)

        rel_repr = self.relation_embd(relation)

        node_repr, rel_repr = self.multi_layer_comp_gcn(memory, rel_repr, head, tail, triple_label, layer_number=self.hop_number)

        head_repr = torch.gather(node_repr, 1, head.unsqueeze(-1).expand(node_repr.size(0), head.size(1), node_repr.size(-1)))
        tail_repr = torch.gather(node_repr, 1, tail.unsqueeze(-1).expand(node_repr.size(0), tail.size(1), node_repr.size(-1)))
        
        # bsz x mem_triple x hidden
        triple_repr = torch.cat((head_repr, rel_repr, tail_repr), dim=-1)
        
        sample = {"input_ids": src_input_ids, "attention_mask": attention_mask, "position_ids": src_position_ids}
        memory = {"triple_repr": triple_repr,
                                        "distance": distance,
                                        "head": head,
                                        "tail": tail,
                                        "concept_label": concept_label,
                                        "triple_label": triple_label,
                                        "vocab_map": vocab_map,
                                        "map_mask": map_mask}

        return seq_generator.generate(self.autoreg_forward, sample, memory)

    def autoreg_forward(self, input_ids, attention_mask, position_ids, memory_dict, do_generate=False, lm_mask=None):

        hidden_states = self.transformer(input_ids, attention_mask = attention_mask, 
                                                    position_ids = position_ids)[0]

        if do_generate:
            hidden_states = hidden_states[:, -1, :].unsqueeze(1)

        sigmoid = nn.Sigmoid()
        softmax = nn.Softmax(dim=-1)
        triple_logits = torch.matmul(hidden_states, self.triple_linear(memory_dict["triple_repr"]).transpose(1, 2))
        
        triple_score = sigmoid(triple_logits)
        # bsz x L x mem_t
    
        triple_score = triple_score.masked_fill((memory_dict["triple_label"] == -1).unsqueeze(1), 0)

        # aggregate probability to nodes
        unorm_cpt_probs = self.multi_hop(triple_score, 
                                                memory_dict["distance"], 
                                                memory_dict["head"], 
                                                memory_dict["tail"], 
                                                memory_dict["concept_label"],
                                                memory_dict["triple_label"], 
                                                gamma = self.gamma,
                                                iteration = self.hop_number,
                                                method = self.aggregate_method)
        # bsz x L x mem 
        cpt_probs = softmax(unorm_cpt_probs)
        # bsz x L x mem

        cpt_probs_vocab = cpt_probs.gather(2, memory_dict["vocab_map"].unsqueeze(1).expand(cpt_probs.size(0), cpt_probs.size(1), -1))

        cpt_probs_vocab.masked_fill_((memory_dict["map_mask"] == 0).unsqueeze(1), 0)
        # bsz x L x vocab
        
        gate = sigmoid(self.gate_linear(hidden_states))
        # bsz x L x 1
        
        lm_logits = self.lm_head(hidden_states)
        lm_probs = softmax(lm_logits)
        
        if do_generate:
            hybrid_probs = lm_probs * (1 - gate) + gate * cpt_probs_vocab
        else:
            hybrid_probs = lm_probs * (1 - gate * lm_mask.unsqueeze(1)) + gate * lm_mask.unsqueeze(1) * cpt_probs_vocab

        return hybrid_probs, gate, triple_score
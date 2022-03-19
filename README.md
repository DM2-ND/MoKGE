## Diversifying Commonsense Reasoning Generation on Knowledge Graph

## Introduction

-- This is the pytorch implementation of our [ACL 2022](https://www.2022.aclweb.org/) paper "*Diversifying Content Generation for Commonsense Reasoning with Mixture of Knowledge Graph Experts*" [\[PDF\]](https://arxiv.org/abs/2203.07285). 
In this paper, we propose MoKGE, a novel method that diversifies the generative commonsense reasoning by a mixture of expert (MoE) strategy on knowledge graphs (KG). 
A set of knowledge experts seek diverse reasoning on KG to encourage various generation outputs.

<img src="logits/MoKGE.jpg" width="800" align=center> 

## Create an environment

```
transformers==3.3.1
torch==1.7.0
nltk==3.4.5
networkx==2.1
spacy==2.2.1
torch-scatter==2.0.5+${CUDA}
psutil==5.9.0
```

-- For `torch-scatter`, `${CUDA}` should be replaced by either `cu101` `cu102` `cu110` or `cu111` depending on your PyTorch installation. For more information check [here](https://github.com/rusty1s/pytorch_scatter).

-- A docker environment could be downloaded from `wenhaoyu97/divgen:5.0`

**We summarize some common environment installation problems and solutions [here](logits/EnvIssues.pdf)**.

## Preprocess the data

-- Extract English ConceptNet and build graph.

```bash
cd data
wget https://s3.amazonaws.com/conceptnet/downloads/2018/edges/conceptnet-assertions-5.6.0.csv.gz
gzip -d conceptnet-assertions-5.6.0.csv.gz
cd ../preprocess
python extract_cpnet.py
python graph_construction.py
```

-- Preprocess multi-hop relational paths. Set `$DATA` to either `anlg` or `eg`.

```bash
export DATA=eg
python ground_concepts_simple.py $DATA
python find_neighbours.py $DATA
python filter_triple.py $DATA
```

## Run Baseline

| Baseline Name | Run Baseline Model | Venue and Reference |
| :--- | :--- | :--- |
| Truncated Sampling | `bash scripts/TruncatedSampling.sh` | Fan et al., ACL 2018 [\[PDF\]](https://aclanthology.org/P18-1082.pdf) |
| Nucleus Sampling | `bash scripts/NucleusSampling.sh` | Holtzman et al., ICLR 2020 [\[PDF\]](https://openreview.net/forum?id=rygGQyrFvH) |
| Variational AutoEncoder | `bash scripts/VariationalAutoEncoder.sh` | Gupta et al., AAAI 2018 [\[PDF\]](https://ojs.aaai.org/index.php/AAAI/article/view/11956) |
| Mixture of Experts <br /> (MoE-embed) | `bash scripts/MixtureOfExpertCho.sh` | Cho et al., EMNLP 2019 [\[PDF\]](https://aclanthology.org/D19-1308/) |
| Mixture of Experts <br /> (MoE-prompt) | `bash scripts/MixtureOfExpertShen.sh` | Shen et al., ICML 2019 [\[PDF\]](http://proceedings.mlr.press/v97/shen19c.html) |

## Run MoKGE

-- Independently parameterizing each expert may exacerbate overfitting since the number of parameters increases linearly with the number of experts. We follow the parameter sharing schema in Cho et al., (2019); Shen et al., (2019) to avoid this issue. This only requires a negligible increase in parameters over the baseline model that does not uses MoE. Speficially, Cho et al., (2019) added a unique expert embedding to each input token, while Shen et al., (2019) added an expert prefix token before the input text sequence.

-- MoKGE-embed (Cho et al.,) `bash scripts/KGMixtureOfExpertCho.sh`

-- MoKGE-prompt (shen et al.,) `bash scripts/KGMixtureOfExpertShen.sh`

## Citation

```
@inproceedings{yu2022diversifying,
  title={Diversifying Content Generation for Commonsense Reasoning with Mixture of Knowledge Graph Experts},
  author={Yu, Wenhao and Zhu, Chenguang and Qin, Lianhui and Zhang, Zhihan and Zhao, Tong and Jiang, Meng},
  booktitle={Findings of Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2022}
}
```

Please kindly cite our paper if you find this paper and the codes helpful.

## Acknowledgements

Many thanks to the Github repository of [Transformers](https://github.com/huggingface/transformers), [KagNet](https://github.com/INK-USC/KagNet) and [MultiGen](https://github.com/cdjhz/multigen). 

Part of our codes are modified based on their codes.

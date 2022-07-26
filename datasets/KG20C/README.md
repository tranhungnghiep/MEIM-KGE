# KG20C: A Scholarly Knowledge Graph Benchmark Dataset

## Introduction

This knowledge graph is constructed to aid research in scholarly data analysis. It can serve as a standard benchmark dataset in scholarly data analysis research for several tasks, including knowledge graph embedding, link prediction, and question answering.

This has been used for my PhD thesis [Multi-Relational Embedding for Knowledge Graph Representation and Analysis](https://ir.soken.ac.jp/?action=pages_view_main&active_action=repository_view_main_item_detail&item_id=6334&item_no=1&page_id=29&block_id=155) and TPDL'19 paper [Exploring Scholarly Data by Semantic Query on Knowledge Graph Embedding Space](https://arxiv.org/abs/1909.08191). 

### Construction protocol
#### Scholarly data
From the [Microsoft Academic Graph](https://academic.microsoft.com/) dataset, we extracted high quality computer science papers published in top conferences between 1990 and 2010. The top conferences list are based on the [CORE ranking](http://portal.core.edu.au/conf-ranks/) A* conferences. The data was cleaned by removing conferences with less than 300 publications and papers with less than 20 citations. The final list includes 20 top conferences: *AAAI, AAMAS, ACL, CHI, COLT, DCC, EC, FOCS, ICCV, ICDE, ICDM, ICML, ICSE, IJCAI, NIPS, SIGGRAPH, SIGIR, SIGMOD, UAI, and WWW*.

#### Knowledge graph
The scholarly dataset was converted to a knowledge graph by defining the entities, the relations, and constructing the triples. The knowledge graph can be seen as a labeled multi-digraph between scholarly entities, where the edge labels express there relationships between the nodes. We use 5 intrinsic entity types including *Paper, Author, Affiliation, Venue, and Domain*. We also use 5 intrinsic relation types between the entities including *author\_in\_affiliation, author\_write\_paper, paper\_in\_domain, paper\_cite\_paper, and paper\_in\_venue*.

#### Benchmark data splitting
The knowledge graph was split uniformly at random into the training, validation, and test sets. We made sure that all entities and relations in the validation and test sets also appear in the training set so that their embeddings can be learned. We also made sure that there is no data leakage and no redundant triples in these splits. This is similar to the construction of the recent benchmark datasets WN18RR and FB15K-237 for link prediction.

### Data content
#### File format
All files are in tab-separated-values format. For example, train.txt includes "28674CFA	author_in_affiliation	075CFC38", which denotes the author with id 28674CFA works in the affiliation with id 075CFC38. The repo includes these files:
- *all_entity_info.txt* contains *id, name, type* of all entities
- *all_relation_info.txt* contains *id* of all entities
- *train.txt* contains training triples of the form *entity_1_id, relation_id, entity_2_id*
- *valid.txt* contains validation triples
- *test.txt* contains test triples

#### Statistics
Data statistics of the KG20C knowledge graph:

Author | Paper | Conference | Domain | Affiliation | All entities | All relations | Training triples | Validation triples | Test triples
:---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---:
8,680 | 5,047 | 20 | 1,923 | 692 | 16,362 | 5 | 48,213 | 3,670 | 3,724

### License
The dataset is free to use for research purpose. For other uses, please follow Microsoft Academic Graph license.

## Baselines
We include the results for link prediction and semantic queries on the KG20C dataset. Link prediction is a relational query task given a relation and the head or tail entity to predict the corresponding tail or head entities. Semantic queries include human-friendly query on the scholarly data. MRR is mean reciprocal rank, Hit@k is the percentage of correct predictions at top k. 

For more information, please refer to the citations.

### Link prediction results
We report results for 4 methods. Random, which is random guess to show the task difficulty. Word2vec, which is the popular embedding method. SimplE/CPh and MEI are two recent knowledge graph embedding methods. All models are in small size settings, equivalent to total embedding size 100 (50x2 for Word2vec and SimplE/CPh, 10x10 for MEI).

Models | MRR | Hit@1 | Hit@3 | Hit@10
:--- | :---: | :---: | :---: | :---:
Random | 0.001 | < 5e-4 | < 5e-4 | < 5e-4
Word2vec | 0.068 | 0.011 | 0.070 | 0.177
SimplE/CPh | 0.215 | 0.148 | 0.234 | 0.348
MEI | **0.230** | **0.157** | **0.258** | **0.368**

### Semantic queries results
The following results demonstrate semantic queries on knowledge graph embedding space, using the above MEI model.

Queries | MRR | Hit@1 | Hit@3 | Hit@10
:--- | :---: | :---: | :---: | :---:
Who may work at this organization? | 0.299 | 0.221 | 0.342 | 0.440
Where may this author work at? | 0.626 | 0.562 | 0.669 | 0.731
Who may write this paper? | 0.247 | 0.164 | 0.283 | 0.405
What papers may this author write? | 0.273 | 0.182 | 0.324 | 0.430
Which papers may cite this paper? | 0.116 | 0.033 | 0.120 | 0.290
Which papers may this paper cite? | 0.193 | 0.097 | 0.225 | 0.40
Which papers may belong to this domain? | 0.052 | 0.025 | 0.049 | 0.100
Which may be the domains of this paper? | 0.189 | 0.114 | 0.206 | 0.333
Which papers may publish in this conference? | 0.148 | 0.084 | 0.168 | 0.257
Which conferences may this paper publish in? | 0.693 | 0.542 | 0.810 | 0.976

## How to cite
If you found this dataset or our work useful, please cite us.

For the dataset and semantic query method, please cite:
- *Hung Nghiep Tran and Atsuhiro Takasu. <a href="https://arxiv.org/abs/1909.08191" target="_blank">Exploring Scholarly Data by Semantic Query on Knowledge Graph Embedding Space</a>. In Proceedings of International Conference on Theory and Practice of Digital Libraries (TPDL), 2019.*

For the MEI knowledge graph embedding model, please cite:
- *Hung Nghiep Tran and Atsuhiro Takasu. <a href="https://arxiv.org/abs/2006.16365" target="_blank">Multi-Partition Embedding Interaction with Block Term Format for Knowledge Graph Completion</a>. In Proceedings of the European Conference on Artificial Intelligence (ECAI), 2020.*

For the baseline results, please cite:
- *Hung Nghiep Tran. <a href="https://ir.soken.ac.jp/?action=pages_view_main&active_action=repository_view_main_item_detail&item_id=6334&item_no=1&page_id=29&block_id=155" target="_blank">Multi-Relational Embedding for Knowledge Graph Representation and Analysis</a>. PhD Dissertation, The Graduate University for Advanced Studies, SOKENDAI, Japan, 2020.*  

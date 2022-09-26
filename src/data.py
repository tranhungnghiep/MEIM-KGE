import os
import time
import copy

import numpy as np
import pandas as pd
import torch


class Data:
    def __init__(self, config):
        self.config = config

        self.rng = np.random.default_rng(self.config.seed)

        if self.config.single_exp or self.config.debug:
            print('Reading data from %s\n' % self.config.in_path)
            base_time_readingdata = time.time()  # in seconds

        self.train_triples_orig = self.read_triples(os.path.join(self.config.in_path, 'train.txt'))
        self.valid_triples_orig = self.read_triples(os.path.join(self.config.in_path, 'valid.txt'))
        self.test_triples_orig = self.read_triples(os.path.join(self.config.in_path, 'test.txt'))

        self.train_triples = self.inv(self.train_triples_orig)  # copy and may inverse
        self.valid_triples = self.inv(self.valid_triples_orig)
        self.test_triples = self.inv(self.test_triples_orig)

        self.ents = self.build_ents_list(self.train_triples + self.valid_triples + self.test_triples)
        self.rels = self.build_rels_list(self.train_triples + self.valid_triples + self.test_triples)

        # from here on, converting to id, working with id
        self.ent_id, self.id_ent, self.ents = self.build_id_dict(self.ents)
        self.rel_id, self.id_rel, self.rels = self.build_id_dict(self.rels)

        self.train_triples_orig = self.build_triples_id(self.train_triples_orig)
        self.valid_triples_orig = self.build_triples_id(self.valid_triples_orig)
        self.test_triples_orig = self.build_triples_id(self.test_triples_orig)

        self.train_triples = self.build_triples_id(self.train_triples)
        self.valid_triples = self.build_triples_id(self.valid_triples)
        self.test_triples = self.build_triples_id(self.test_triples)


        # for correcting false negative in negative sampling
        self.known_htr_train = set(self.train_triples)
        self.known_htr_valid = set(self.valid_triples)
        self.known_htr_test = set(self.test_triples)
        self.known_htr_all = set(self.train_triples + self.valid_triples + self.test_triples)
        # for correcting false negative in negfull scoring
        self.known_hr_t_train, self.known_tr_h_train = self.build_known_triples_dict(self.train_triples)
        self.known_hr_t_valid, self.known_tr_h_valid = self.build_known_triples_dict(self.valid_triples)
        self.known_hr_t_test, self.known_tr_h_test = self.build_known_triples_dict(self.test_triples)
        # for filtering false negative in test
        self.known_hr_t_all, self.known_tr_h_all = self.build_known_triples_dict(self.train_triples + self.valid_triples + self.test_triples)
        # for negfull sampling
        self.known_hr_train, self.known_tr_train = list(self.known_hr_t_train.keys()), list(self.known_tr_h_train.keys())
        self.known_hr_valid, self.known_tr_valid = list(self.known_hr_t_valid.keys()), list(self.known_tr_h_valid.keys())
        self.known_hr_test, self.known_tr_test = list(self.known_hr_t_test.keys()), list(self.known_tr_h_test.keys())
        self.known_hr_all, self.known_tr_all = list(self.known_hr_t_all.keys()), list(self.known_tr_h_all.keys())
        # for filter by ent type
        if self.config.eval_intype:
            self.relid_entid_intype = self.get_relid_entid_intype()


        # statistics
        self.data_str = ('Num_ent: %i\n' % len(self.ents)
                         + 'Num_rel: %i\n' % len(self.rels)
                         + 'Num_train: %i\n' % len(self.train_triples)
                         + 'Num_valid: %i\n' % len(self.valid_triples)
                         + 'Num_test: %i\n' % len(self.test_triples)
                         + 'Num_hr_train: %i\n' % len(self.known_hr_train)
                         + 'Num_hr_valid: %i\n' % len(self.known_hr_valid)
                         + 'Num_hr_test: %i\n' % len(self.known_hr_test)
                         + 'Num_hr_all: %i\n' % len(self.known_hr_all)
                         + 'Num_tr_train: %i\n' % len(self.known_tr_train)
                         + 'Num_tr_valid: %i\n' % len(self.known_tr_valid)
                         + 'Num_tr_test: %i\n' % len(self.known_tr_test)
                         + 'Num_tr_all: %i' % len(self.known_tr_all))


        # cache for data sampling
        # negsamp
        self.triple_sample = np.zeros((self.config.batch_size + self.config.batch_size * self.config.neg_ratio, 3), dtype=np.int64, order='F')  # pre-allocate
        self.triple_label = np.zeros((self.config.batch_size + self.config.batch_size * self.config.neg_ratio), dtype=np.float32)
        # negfull
        self.hr_sample = np.zeros((self.config.batch_size, 2), dtype=np.int64, order='F')
        if self.config.reuse_array in ['np0', 'torch0', 'torch0pin', 'torch0gpu']:  # creating new array in each batch
            self.t_label = None
        elif self.config.reuse_array == 'np1':  # pre-allocate
            self.t_label = np.zeros((self.config.batch_size, len(self.ents)), dtype=np.float32)
        elif self.config.reuse_array == 'torch1':
            self.t_label = torch.zeros((self.config.batch_size, len(self.ents)), dtype=torch.float32)
        elif self.config.reuse_array == 'torch1pin':
            self.t_label = torch.zeros((self.config.batch_size, len(self.ents)), dtype=torch.float32).pin_memory()
        elif self.config.reuse_array == 'torch1gpu':
            self.t_label = torch.zeros((self.config.batch_size, len(self.ents)), dtype=torch.float32, device=torch.device('cuda'))

        if self.config.single_exp or self.config.debug:
            print('Done reading data, %f (s).\n' % (time.time() - base_time_readingdata))

    def read_triples(self, filepath):
        """
        :param filepath: file format: each line is 'h	r	t'
        :return: list of tuple [(h, t, r)], raw data.
        """
        triples = []
        with open(filepath, "r") as f:
            for line in f.readlines():
                h, r, t = line.split()  # note the order
                triples.append((h, t, r))
        return triples

    def inv(self, triples):
        new_triples = copy.deepcopy(triples)
        if not self.config.model_inv:
            for (h, t, r) in triples:
                new_triples.append((t, h, r + '_inv'))
        return new_triples

    def build_ents_list(self, triples):
        """
        :param triples: (h, t, r)
        :return: sorted list of ent h, t
        """
        return sorted(list(set([e for (h, t, r) in triples for e in (h, t)])))

    def build_rels_list(self, triples):
        """
        :param triples: (h, t, r)
        :return: sorted list of rel r
        """
        return sorted(list(set([r for (h, t, r) in triples])))

    def build_id_dict(self, items):
        """
        Build bijective dictionary item to id.
        :param items: raw ents or rels
        :return: 2 dicts and items list in id format
        """
        item_id = {items[i]: i for i in range(len(items))}
        id_item = {i: items[i] for i in range(len(items))}
        items = [i for i in range(len(items))]
        return item_id, id_item, items

    def build_triples_id(self, triples):
        """
        :param triples: raw [(h, t, r)]
        :return: id [(h, t, r)]
        """
        return [(self.ent_id[h], self.ent_id[t], self.rel_id[r]) for (h, t, r) in triples]

    def build_known_triples_dict(self, triples):
        """
        Get known triples dict to filter positive/negative fast
        :param triples: (h, t, r)
        :return: (h, r): [t] and (t, r): [h]
        """
        known_hr_t = {}
        known_tr_h = {}
        for (h, t, r) in triples:
            if (h, r) not in known_hr_t:
                known_hr_t[(h, r)] = [t]
            elif t not in known_hr_t[(h, r)]:
                known_hr_t[(h, r)].append(t)

            if (t, r) not in known_tr_h:
                known_tr_h[(t, r)] = [h]
            elif h not in known_tr_h[(t, r)]:
                known_tr_h[(t, r)].append(h)

        return known_hr_t, known_tr_h

    def get_relid_entid_intype(self, all_ent_info_file='all_entity_info.txt'):
        assert 'KG20C' in self.config.in_path, 'Filtering by enttype is not supported for %s ' % self.config.in_path

        # define rel-based ent types for filtering
        # hard-coded for bib kg such as KG20C
        rel_enttypes = {
            "author_in_affiliation": ["affiliation"],
            "author_write_paper": ["paper"],
            "domain_in_domain": ["domain"],
            "paper_cite_paper": ["paper"],
            "paper_in_domain": ["domain"],
            "paper_in_venue": ["conference"],
            "paper_in_year": ["year"],
            "author_in_affiliation_inv": ["author"],
            "author_write_paper_inv": ["author"],
            "domain_in_domain_inv": ["domain"],
            "paper_cite_paper_inv": ["paper"],
            "paper_in_domain_inv": ["paper"],
            "paper_in_venue_inv": ["paper"],
            "paper_in_year_inv": ["paper"]
        }

        # get type of each entity
        all_ent_info_file_path = os.path.join(self.config.in_path, all_ent_info_file)
        all_ent_info = pd.read_csv(all_ent_info_file_path, sep='\t')
        ent_type = {ent: enttype for (ent, _, enttype) in all_ent_info.values}

        # build dict {relid: [entid in type]} based on rel_enttypes and ent_type
        relid_entid_intype = {self.rel_id[rel]: [self.ent_id[ent] for (ent, enttype) in ent_type.items()
                                                 if enttype in rel_enttypes[rel]]
                              for rel in rel_enttypes.keys()
                              if rel in self.rel_id.keys()}  # rel in rel_enttypes can be broader than rel in rel_id

        return relid_entid_intype

    def sampling_negsamp(self, triples, idx, batch_size, triple_sample, triple_label):
        """
        Getting next batch with negative sampling, uniformly.
        Remember to shuffle triples before each epoch.
        :param triples: data to sample from: such as self.train_triples
        :param idx: start index
        :param batch_size: batch size
        :param triple_sample: ref to output self.triple_sample ((htr)*(batch_size+batch_size*negratio))
        :param triple_label: ref to output self.label_sample (1*(batch_size+batch_size*negratio))
        :return: h, t, r, y: 4 arrays [batchsize + batchsize * neg_ratio,].
        """
        # automatically cyclic guard using itertools
        import itertools
        triple_sample[:batch_size] = list(itertools.islice(itertools.cycle(triples), idx, idx + batch_size))  # copy positive triples
        triple_label[:batch_size] = 1.0

        neg_size = batch_size * self.config.neg_ratio
        if self.config.neg_ratio > 0:  # negative sampling in train
            rdm_entities = np.random.randint(0, len(self.ents), neg_size)  # pre-sample all negative entities
            triple_sample[batch_size:batch_size + neg_size, :] = np.tile(triple_sample[:batch_size], (self.config.neg_ratio, 1))  # pre-copy negative triples
            triple_label[batch_size:batch_size + neg_size] = 0.0
            rdm_choices = np.random.random(neg_size) < 0.5  # pre-sample choices head/tail
            for i in range(neg_size):
                if rdm_choices[i]:
                    triple_sample[batch_size + i, 1] = rdm_entities[i]  # corrupt tail
                else:
                    triple_sample[batch_size + i, 0] = rdm_entities[i]  # corrupt head

                if tuple(triple_sample[batch_size + i, :]) in self.known_htr_train:
                    triple_label[batch_size + i] = 1.0  # correcting false negative, rare negative will require large neg_ratio and many epoches to learn

        return triple_sample[:batch_size + neg_size, 0], triple_sample[:batch_size + neg_size, 1], triple_sample[:batch_size + neg_size, 2], triple_label[:batch_size + neg_size]

    def sampling_kvsall(self, known_er, known_er_e, idx, batch_size, sample, label):
        """
        Getting next batch for softmax loss with k positive vs all negative,
        could use for both directions hr_t and tr_h depending on the content of known_er.
        Remember to shuffle known_er before each epoch and pass correct idx, sample, label.
        :param known_er: input pairs to sample from, hr or tr
        :param known_er_e: filter to set label, hr_t or tr_h
        :param idx: start index
        :param batch_size: batch size
        :param sample: ref to output pairs, ((hr)*batch_size) or ((tr)*batch_size)
        :param label: ref to output label (1*batch_size)
        :return: e: array [batchsize,]; r: array [batchsize,]; y: array [batchsize, num_ent]; sample_t for sampled scoring; active_e for weight decay.
        """
        # automatically cyclic guard using itertools
        import itertools
        sample[:batch_size] = list(itertools.islice(itertools.cycle(known_er), idx, idx + batch_size))  # get sample

        if self.config.reuse_array in ['np1', 'torch1', 'torch1pin', 'torch1gpu']:  # resetting and reusing old array
            label[:batch_size, :] = 0.0  # reset label
        elif self.config.reuse_array == 'np0':  # create new array for each batch
            label = np.zeros((self.config.batch_size, len(self.ents)), dtype=np.float32)
        elif self.config.reuse_array == 'torch0':
            label = torch.zeros((self.config.batch_size, len(self.ents)), dtype=torch.float32)
        elif self.config.reuse_array == 'torch0pin':
            label = torch.zeros((self.config.batch_size, len(self.ents)), dtype=torch.float32).pin_memory()
        elif self.config.reuse_array == 'torch0gpu':
            label = torch.zeros((self.config.batch_size, len(self.ents)), dtype=torch.float32, device=torch.device('cuda'))

        active_e = sample[:batch_size, 0].tolist()

        for i, (e, r) in enumerate(sample[:batch_size]):  # update label
            active_e.extend(known_er_e[(e, r)])
            label[i, known_er_e[(e, r)]] = 1.0

        sample_t = []
        if self.config.tail_fraction < 1.0:
            sample_t = self.rng.choice(len(self.ents), int(len(self.ents) * self.config.tail_fraction), replace=False)
            sample_t = list(set(sample_t) | set(active_e))  # make sure sample_t includes all active h and t

        y = label[:batch_size, :]
        y = y[:, sample_t] if len(sample_t) > 0 else y  # mask select y for sampled softmax

        return sample[:batch_size, 0], sample[:batch_size, 1], y, sample_t, active_e

    def sampling_1vsall(self, triples_orig, idx, batch_size, sample, label):
        """
        Getting next batch for full softmax loss with 1 positive vs all negative.
        This is exactly the same as Lacroix sampling.
        Remember to shuffle triples_orig before each epoch and pass correct idx, sample, label.
        :param triples_orig: input [{htr}]
        :param idx: start index
        :param batch_size: batch size, should be divided by 2, equivalent to batch size in kvsall
        :param sample: ref to output pairs, ((hr)*batch_size) or ((tr)*batch_size)
        :param label: ref to output label (1*batch_size)
        :return: e: array [batchsize,]; r: array [batchsize,]; y: array [batchsize, num_ent]; sample_t for sampled scoring; active_e for weight decay.
        """
        assert batch_size % 2 == 0, 'Always use even batch_size due to separating 1 triple into 2'

        sample[:batch_size, :] = 0  # reset sample

        if self.config.reuse_array in ['np1', 'torch1', 'torch1pin', 'torch1gpu']:  # resetting and reusing old array
            label[:batch_size, :] = 0.0  # reset label
        elif self.config.reuse_array == 'np0':  # create new array for each batch
            label = np.zeros((self.config.batch_size, len(self.ents)), dtype=np.float32)
        elif self.config.reuse_array == 'torch0':
            label = torch.zeros((self.config.batch_size, len(self.ents)), dtype=torch.float32)
        elif self.config.reuse_array == 'torch0pin':
            label = torch.zeros((self.config.batch_size, len(self.ents)), dtype=torch.float32).pin_memory()
        elif self.config.reuse_array == 'torch0gpu':
            label = torch.zeros((self.config.batch_size, len(self.ents)), dtype=torch.float32, device=torch.device('cuda'))

        for i in range(batch_size // 2):
            (h, t, r) = triples_orig[(idx + i) % len(triples_orig)]  # cyclic guard against out of data
            r_inv = self.rel_id[self.id_rel[r] + '_inv']
            sample[i, :] = (h, r)  # list of tuple can be assigned to array
            label[i, t] = 1.0
            sample[batch_size // 2 + i, :] = (t, r_inv)  # corresponding 2 ways
            label[batch_size // 2 + i, h] = 1.0

        active_e = sample[:batch_size, 0].tolist()

        sample_t = []
        if self.config.tail_fraction < 1.0:
            sample_t = self.rng.choice(len(self.ents), int(len(self.ents) * self.config.tail_fraction), replace=False)
            sample_t = list(set(sample_t) | set(active_e))  # make sure sample_t includes all active h and t

        y = label[:batch_size, :]
        y = y[:, sample_t] if len(sample_t) > 0 else y  # mask select y for sampled softmax

        return sample[:batch_size, 0], sample[:batch_size, 1], y, sample_t, active_e

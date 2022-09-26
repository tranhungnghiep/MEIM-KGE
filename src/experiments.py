#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Hung-Nghiep Tran

import os
import time
import datetime

import random
import numpy as np
import scipy as sp
import pandas as pd
import torch

from data import Data
import utils


class Experiment(object):
    def __init__(self, config):
        self.config = config
        self.configure()

        self.data = Data(config=self.config)

        self.model = self.config.model_class(data=self.data, config=self.config)  # construct new model instance
        if self.config.device == 'cuda':
            self.model.cuda()

        if self.config.opt_method.lower() == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr, betas=(0.9, 0.999), eps=1e-8, amsgrad=self.config.amsgrad)
        elif self.config.opt_method.lower() == "adagrad":
            self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.config.lr, initial_accumulator_value=self.config.adagrad_iav, eps=1e-10)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.lr, momentum=0.9, nesterov=True)

        if 'warmup' in self.config.lr_scheduler:
            baselr = self.config.lr
            maxlr = self.config.lr_warmup_maxlr
            num_step = self.config.lr_warmup_epoch
            if self.config.lr_warmup_space == 'lin':
                step_space = np.linspace(baselr, maxlr, num_step)  # warmup lr in linspace
            elif self.config.lr_warmup_space == 'log':
                step_space = np.logspace(np.log10(baselr), np.log10(maxlr), num_step)  # warmup lr in logspace
            lr_warmup_lambda = lambda epoch: step_space[epoch] / baselr
            self.scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_warmup_lambda, last_epoch=-1)
        if 'exp' in self.config.lr_scheduler:
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, self.config.lr_decay)
        if 'rdop' in self.config.lr_scheduler:
            self.scheduler_rdop = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',
                                                                             factor=self.config.lr_rdop_f, patience=self.config.lr_rdop_p,
                                                                             threshold=self.config.lr_rdop_t, threshold_mode='rel',
                                                                             cooldown=0, min_lr=self.config.lr_minlr, eps=1e-08, verbose=False)

        if self.config.tb_logdir:
            self.tb_writer = utils.get_tb_writer(log_dir=self.config.tb_logdir)
            h, r, *_ = next(self.batch_generator())
            self.tb_writer.add_graph(self.model, {'h': h, 'r': r})  # add graph of model, input needed for dynamic graph generation
            self.tb_writer.close()  # flush to write to disk
            # # to log embeddings
            # self.tb_writer.add_embedding(embs, metadata=class_labels, label_img=originalimages)
            # self.tb_writer.close()

        if self.config.logging:
            import logging
            self.logger = utils.get_logger(self.config.exp_id, loglevel=logging.DEBUG,
                                           file_paths=(os.path.join(self.config.out_path, 'log', self.config.config_id, self.config.exp_id + '.log'),), file_loglevel=logging.INFO,
                                           stream=True if self.config.single_exp or self.config.debug else False, stream_loglevel=logging.DEBUG)

            self.logger.info('New experiment\n')
            self.logger.info('%s\n' % self.config.exp_id)
            self.logger.info('%s\n' % self.config.arg_str)
            self.logger.info('%s\n' % self.data.data_str)

    def configure(self):
        # Re-set random seed for each new experiment
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        # tf.set_random_seed(self.config.seed)
        # tf.random.set_seed(self.config.seed)

        # deterministic gpu
        if self.config.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # GPU config
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.config.gpu)
        # self.config.tfcfg = tf.ConfigProto()
        # self.config.tfcfg.gpu_options.allow_growth = True
        # for gpu in tf.config.experimental.list_physical_devices('GPU'):
        #     tf.config.experimental.set_memory_growth(gpu, True)

        # create necessary dirs
        if self.config.out_path:
            os.makedirs(self.config.out_path, exist_ok=True)
        if self.config.out_path and self.config.logging and self.config.config_id:
            os.makedirs(os.path.join(self.config.out_path, 'log', self.config.config_id), exist_ok=True)
        if self.config.tb_logdir:
            os.makedirs(self.config.tb_logdir, exist_ok=True)

        self.config.exp_id += datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S_%fUTC') + '--model=%s--dataset=%s' % (
            self.config.model, os.path.basename(os.path.normpath(self.config.in_path)))  # dataset instead of in_path
        self.config.model_str = ''.join('--%s=%s' % (str(arg), str(vars(self.config)[arg])) if str(vars(self.config)[arg]) else '--%s=%s' % (str(arg), '""')
                                        for arg in self.config.argkey_orig)  # --arg=val, orig argkey, updated argval, declared order
        self.config.arg_str = self.config.model_str.replace('--', ' --').replace('=', ' ').strip()  # --arg val, same content as config.model_str

    def batch_generator(self):
        """
        Iterate over each batch in the data set, return a generator for each epoch.
        :return: generator h, r, y torch.Tensor, on gpu if applicable
        """
        if self.config.sampling == 'negsamp':  # transe
            raise NotImplementedError('Score computation for negsamp sampling is not implemented in this version yet.')

        elif self.config.sampling == 'kvsall':  # conve with loss binary xent; new one with loss softmax xent
            np.random.shuffle(self.data.known_hr_train)  # shuffle once for new generator init, every epoch, inplace
            for idx in range(0, len(self.data.known_hr_train), self.config.batch_size):  # iterate over batches
                h, r, y, sample_t, active_e = self.data.sampling_kvsall(self.data.known_hr_train, self.data.known_hr_t_train,
                                                           idx, self.config.batch_size, self.data.hr_sample, self.data.t_label)  # sampling to numpy
                h, r, y, sample_t, active_e = utils.to_torch(h, self.config.device), utils.to_torch(r, self.config.device), utils.to_torch(y, self.config.device), utils.to_torch(sample_t, self.config.device), utils.to_torch(active_e, self.config.device)
                yield h, r, y, sample_t, active_e  # generator

        elif self.config.sampling == '1vsall':  # lacroix with loss softmax xent
            np.random.shuffle(self.data.train_triples_orig)
            for idx in range(0, len(self.data.train_triples_orig), self.config.batch_size):
                h, r, y, sample_t, active_e = self.data.sampling_1vsall(self.data.train_triples_orig,
                                                           idx, self.config.batch_size, self.data.hr_sample, self.data.t_label)
                h, r, y, sample_t, active_e = utils.to_torch(h, self.config.device), utils.to_torch(r, self.config.device), utils.to_torch(y, self.config.device), utils.to_torch(sample_t, self.config.device), utils.to_torch(active_e, self.config.device)
                yield h, r, y, sample_t, active_e

    def run(self):
        self.logger.info('Start training\n')
        exp_time = time.time()  # in seconds

        if self.config.constraint:  # if constraint not "", enforce constraint on all embs at start
            self.model.enforce_hardconstraint()

        self.best_eval_train = self.evaluate(self.data.train_triples[:1000], test_name='TRAIN')
        self.best_eval_valid = self.evaluate(self.data.valid_triples, test_name='VALID')
        self.best_eval_test = self.evaluate(self.data.test_triples, test_name='TEST')
        self.best_epoch = 0
        self.best_epoch_loss = 0.0
        violation = 0
        for epoch in range(1, self.config.max_epoch + 1):  # 1 to == max_epoch
            eval_valid = None  # reset eval_valid before every epoch, recompute in early stop checking or lr rdop

            if self.config.write_params and 0 in self.config.write_params_epochs and epoch == 1:  # write epoch 0 (before training)
                self.write_params(epoch=0)

            epoch_time = time.time()  # in seconds
            epoch_losses = []
            for h, r, y, sample_t, active_e in self.batch_generator():  # every batch
                self.model.train()  # set model to train mode before forward
                score = self.model({'h': h, 'r': r, 'sample_t': sample_t})
                loss = self.model.compute_loss_total(score, y, active_e)
                self.optimizer.zero_grad()  # zero grad before loss backward compute gradient
                loss.backward()
                self.optimizer.step()
                if self.config.constraint:  # if constraint not "", enforce constraint on active embs
                    self.model.enforce_hardconstraint(h, r)
                epoch_losses.append(loss.item())

            if self.config.logging:  # logging after every epoch
                self.logger.info(epoch)
                self.logger.info('Epoch time (s): %s' % str(time.time() - epoch_time))
                self.logger.info('loss: %E' % (np.mean(epoch_losses)))
                for i, param_group in enumerate(self.optimizer.param_groups):  # current lr for sanity check
                    self.logger.info('lr: %E' % float(param_group['lr']))

            if epoch % self.config.check_period == 0:  # evaluate model after every check_period epoches
                eval_train = self.evaluate(self.data.train_triples[:1000], test_name='TRAIN')
                eval_valid = self.evaluate(self.data.valid_triples, test_name='VALID')
                eval_test = self.evaluate(self.data.test_triples, test_name='TEST')
                if eval_valid['filter'][1] > self.best_eval_valid['filter'][1]:  # new best f_mrr
                    self.best_eval_train = eval_train
                    self.best_eval_valid = eval_valid
                    self.best_eval_test = eval_test
                    self.best_epoch = epoch
                    self.best_epoch_loss = np.mean(epoch_losses)
                    violation = 0
                    self.logger.info('\n***New best valid f_mrr %.3f at epoch %i with loss %E.***\n\n' % (self.best_eval_valid['filter'][1], self.best_epoch, self.best_epoch_loss))
                    if self.config.save_model:  # save new best valid model
                        save_path = os.path.join(self.config.out_path, '%s.epoch=%06d.torch' % (self.config.exp_id, self.best_epoch))
                        torch.save(self.model.state_dict(), save_path)
                    if self.config.write_params >= 20 and epoch >= self.config.write_params_burnin_epoch and epoch not in self.config.write_params_epochs:  # verbose 20 write at new best valid epoch, write when not duplicate with write_params_epochs list
                        self.write_params(epoch=epoch)
                else:
                    violation += 1
                    if self.config.early_stop and violation > self.config.patience:
                        break

            if self.config.eval_eachpartition and epoch in self.config.eval_eachpartition_epochs:
                self.evaluate(self.data.train_triples[:1000], test_name='TRAIN')
                self.evaluate(self.data.valid_triples, test_name='VALID', write_ranks=self.config.write_ranks)
                self.evaluate(self.data.test_triples, test_name='TEST', write_ranks=self.config.write_ranks)
                for k in range(self.config.K):  # loop outside of evaluate() for more simple code reuse
                    self.evaluate(self.data.train_triples[:1000], test_name='TRAIN (partition %i)' % k, partition=k)
                    self.evaluate(self.data.valid_triples, test_name='VALID (partition %i)' % k, partition=k, write_ranks=self.config.write_ranks)
                    self.evaluate(self.data.test_triples, test_name='TEST (partition %i)' % k, partition=k, write_ranks=self.config.write_ranks)

            if 'warmup' in self.config.lr_scheduler and epoch < self.config.lr_warmup_epoch:
                self.scheduler_warmup.step()
            else:  # not warmup at any epoch OR warmup but at large epoch, start scheduler step freely
                # early guard: only lrdecay if there is at least 1 lr > minlr
                need_lrdecay = False
                for i, param_group in enumerate(self.optimizer.param_groups):
                    if param_group['lr'] > self.config.lr_minlr + self.config.epsilon:
                        need_lrdecay = True
                # lr decay: exp and/or rdop...
                if 'exp' in self.config.lr_scheduler and need_lrdecay:
                    self.scheduler.step()  # exponential decay lr = self.config.lr * (self.config.lr_decay ** epoch)
                if 'rdop' in self.config.lr_scheduler and need_lrdecay:
                    if self.config.lr_rdop_watch == 'val_loss':
                        raise NotImplementedError('Loss on validation set is not implemented yet.')
                    elif self.config.lr_rdop_watch == 'val_mrr':
                        if not eval_valid:  # if none, recompute eval_valid; otherwise, reuse eval_valid from early stop checking to save computation
                            eval_valid = self.evaluate(self.data.valid_triples, test_name='VALID', verbose=False)
                        self.scheduler_rdop.step(-eval_valid['filter'][1])  # mode min, pass in: - valid f_mrr
                # final guard: not use too small lr, reset to minlr
                for i, param_group in enumerate(self.optimizer.param_groups):
                    if param_group['lr'] < self.config.lr_minlr:
                        param_group['lr'] = self.config.lr_minlr

            if self.config.export_step and epoch % self.config.export_step == 0 and self.config.save_model:  # save model after every export_step epoches
                save_path = os.path.join(self.config.out_path, '%s.epoch=%06d.torch' % (self.config.exp_id, epoch))
                torch.save(self.model.state_dict(), save_path)  # for inference only

            if self.config.write_params and epoch in self.config.write_params_epochs:  # write after training
                self.write_params(epoch=epoch)

        if 0 < self.best_epoch < self.config.max_epoch:  # found a best epoch, can early stop
            self.logger.info('\nEarly stop by f_mrr on valid\n')
            if self.config.save_model:
                save_path = os.path.join(self.config.out_path, '%s.epoch=%06d.torch' % (self.config.exp_id, self.best_epoch))
                self.model.load_state_dict(torch.load(save_path))  # load best valid model
                save_path = os.path.join(self.config.out_path, '%s.epoch=%06d.best.torch' % (self.config.exp_id, self.best_epoch))
                torch.save(self.model.state_dict(), save_path)  # save as best model
        else:
            self.logger.info('\nStop by last epoch.\n')

        self.logger.info('%s\n' % self.format_eval_result_multiline_print(self.best_eval_train, 'TRAIN', print_each_rel=self.config.print_each_rel))
        self.logger.info('%s\n' % self.format_eval_result_multiline_print(self.best_eval_valid, 'VALID', print_each_rel=self.config.print_each_rel))
        self.logger.info('%s\n' % self.format_eval_result_multiline_print(self.best_eval_test, 'TEST', print_each_rel=self.config.print_each_rel))
        self.logger.info('Finish training. Best valid f_mrr %.3f at best epoch %i with loss %E.\n\n' % (self.best_eval_valid['filter'][1], self.best_epoch, self.best_epoch_loss))

        self.logger.info('Total training time: %s (%f seconds).\n' % (str(datetime.timedelta(seconds=time.time() - exp_time)), time.time() - exp_time))

    def write_params(self, epoch):
        """
        This method writes model params, including embeddings, core tensors, mappings.
        Params files in gensim w2v format (a simple csv is better, but anyway, just write it and check how to use later).
        This method also write self.config.arg_str to self.config.exp_id+'_config.txt' and current epoch in file name to reproduce the params.
        For example: out/semquery_cph_20200528_063618_830987UTC--model=CPh--dataset=dropydd_cs/ent_embs.epoch=%06d.best.txt
        """
        def write_emb(data, id_name, file_path):
            """
            1 file for all ents, 1 file for all rels. each line is an emb, all partitions concated.
            num_emb emb_size
            name1 v11 v12 v13...
            name2 v21 v22 v23...
            ...
            """
            data = pd.DataFrame(data[:, :])
            data.insert(loc=0, column='name', value=[id_name[id] for id in range(data.shape[0])])  # insert name column
            with open(file_path, 'w') as f:
                f.write('%i %i\n' % (data.shape[0], data.shape[1] - 1))  # write header for w2v gensim format first
            data.to_csv(file_path, sep=' ', header=False, index=False, mode='a')

        def write_core(data, file_path):
            """
            each line is a core cr*ce*ce at k (thus should only work with small c).
            k ce cr
            k1 v11 v12 v13...
            k2 v21 v22 v23...
            ...
            """
            data = pd.DataFrame(data[:, :])
            data.insert(loc=0, column='name', value=['k%i' % (k + 1) for k in range(self.config.K)])  # insert name column
            with open(file_path, 'w') as f:
                f.write('%i %i %i\n' % (self.config.K, self.config.Ce, self.config.Cr))  # write header
            data.to_csv(file_path, sep=' ', header=False, index=False, mode='a')

        def write_mapping(data, id_name, file_path):
            """
            1 file for all rels. each line is a mapping, all partitions concated.
            num_mapping k ce
            name1 v11 v12 v13...
            name2 v21 v22 v23...
            ...
            """
            data = pd.DataFrame(data[:, :])
            data.insert(loc=0, column='name', value=[id_name[id] for id in range(data.shape[0])])  # insert name column
            with open(file_path, 'w') as f:
                f.write('%i %i %i\n' % (data.shape[0], self.config.K, self.config.Ce))  # write header
            data.to_csv(file_path, sep=' ', header=False, index=False, mode='a')

        write_path = os.path.join(self.config.params_out_path, self.config.exp_id)
        os.makedirs(write_path, exist_ok=True)
        if not os.path.isfile(os.path.join(write_path, 'config.txt')):
            with open(os.path.join(write_path, 'config.txt'), 'w') as f:
                f.write(self.config.arg_str)  # write arg string to config.txt if not exists

        print('Write params at epoch %i to %s.' % (epoch, write_path))
        base_time = time.time()  # in seconds

        if self.config.write_params_emb:
            ent_embs = self.model.ent_embs.view(len(self.data.ents), -1)  # 2d array (num_ent, k*ce), unroll from inner out: group of all c entries in each partition k
            write_emb(utils.to_numpy(ent_embs), self.data.id_ent, os.path.join(write_path, 'ent_embs.epoch=%06d.txt' % epoch))
            rel_embs = self.model.rel_embs.view(len(self.data.rels), -1)
            write_emb(utils.to_numpy(rel_embs), self.data.id_rel, os.path.join(write_path, 'rel_embs.epoch=%06d.txt' % epoch))

        if self.config.write_params_core:
            cores = self.model.wv.view(-1, self.config.Cr * self.config.Ce * self.config.Ce)  # 2d array (1, cr*ce*ce) or (k, cr*ce*ce)
            write_core(utils.to_numpy(cores), os.path.join(write_path, 'cores.epoch=%06d.txt' % epoch))

        if self.config.write_params_mapping:
            Mr = self.model.get_mr(self.model.rel_embs, self.model.wv).view(len(self.data.rels), -1)  # 2d array (num_rel, k*ce*ce)
            write_mapping(utils.to_numpy(Mr), self.data.id_rel, os.path.join(write_path, 'mappings.epoch=%06d.txt' % epoch))

        print('Done writing, %f seconds.' % (time.time() - base_time))

    def evaluate_simple(self, triples, test_name, verbose=True):
        """
        Test link prediction purely in python. Simple basic version for easy understanding.
        :param triples: [(h, t, r)]
        :param test_name: name of test case for printing: test, valid, train
        :return: r_mr, r_mrr, r_hit1, r_hit3, r_hit10, f_mr, f_mrr, f_hit1, f_hit3, f_hit10
        """
        base_time = time.time()  # in seconds
        self.model.eval()  # set model to eval mode

        if not self.config.model_inv:
            total = len(triples) // 2  # data inverse, triples duplicate
        else:
            raise NotImplementedError('Score computation for model_inv 2 ways is not implemented in this version yet.')

        known_hr_t, known_tr_h = self.data.build_known_triples_dict(triples)  # only use hr_t because inv data
        known_hr, known_tr = list(known_hr_t.keys()), list(known_tr_h.keys())  # only use hr because inv data

        # Compute the rank of every triple; ranks size = 2x triples size, for tail ranks and head ranks
        raw_ranks, ranks, raw_ranks_max, ranks_max = np.zeros(2 * total), np.zeros(2 * total), np.zeros(2 * total), np.zeros(2 * total)  # init zero to used in on-progress report, raw_ranks are min rank of queries, ranks are filtered out known triples, raw_ranks_max of queries are to compute mean rank

        count = 0
        for idx in range(0, len(known_hr), self.config.batch_size):  # for batch of hr, loop over data set once
            hr = np.array(known_hr[idx:idx + self.config.batch_size], dtype=np.int64, order='F')  # list to numpy
            h, r = utils.to_torch(hr[:, 0], self.config.device), utils.to_torch(hr[:, 1], self.config.device)
            with torch.no_grad():
                score = self.model({'h': h, 'r': r})  # (batch, num_ent)
            h, r, score = utils.to_numpy(h), utils.to_numpy(r), utils.to_numpy(score)  # TODO: maybe can use torch tensor directly

            for row in range(len(score)):  # for each hr row: rank all positive t (cover all triples, both directions)
                score_raw = score[row, :]
                score_known = score[row, self.data.known_hr_t_all[(h[row], r[row])]]
                for t in known_hr_t[(h[row], r[row])]:  # for each positive t of this hr: compute rank
                    if self.config.eval_scoreorder in ['min', 'mean']:
                        raw_ranks[count] = 1 + np.sum(score_raw > score[row, t])  # t should have largest score: count how many score larger than t; min rank: strictly larger
                        ranks[count] = raw_ranks[count] - np.sum(score_known > score[row, t])  # remove known and ranked better, not include t because strictly larger
                    if self.config.eval_scoreorder in ['max', 'mean']:
                        raw_ranks_max[count] = 1 + (np.sum(score_raw >= score[row, t]) - 1)  # max rank: larger or equal, minus 1 for t itself
                        ranks_max[count] = raw_ranks_max[count] - (np.sum(score_known >= score[row, t]) - 1)  # remove known and ranked better or equal, minus 1 for t itself
                    count += 1  # next positive triple

        assert count == 2 * total, 'Check missing count in evaluation'

        def get_final_ranks(raw_ranks, ranks, raw_ranks_max, ranks_max):
            if self.config.eval_scoreorder == 'min':
                return raw_ranks, ranks
            elif self.config.eval_scoreorder == 'max':
                return raw_ranks_max, ranks_max
            elif self.config.eval_scoreorder == 'mean':
                return 1 / 2 * (raw_ranks + raw_ranks_max), 1 / 2 * (ranks + ranks_max)

        def get_eval_result(raw_ranks, ranks, eval_result={}, add_str=''):
            eval_result['raw%s' % add_str] = utils.metrics(raw_ranks)  # main result
            eval_result['filter%s' % add_str] = utils.metrics(ranks)
            return eval_result

        raw_ranks, ranks = get_final_ranks(raw_ranks, ranks, raw_ranks_max, ranks_max)
        eval_result = get_eval_result(raw_ranks, ranks)

        if verbose:
            self.logger.info('%s\n' % self.format_eval_result_multiline_print(eval_result, test_name, additional_str='%i triples, %f seconds' % (total, time.time() - base_time)))

        self.model.train()  # set model to train mode
        return eval_result

    def evaluate(self, triples, test_name, partition=None, write_ranks=False, verbose=True):
        """
        Test link prediction purely in python
        :param triples: [(h, t, r)]
        :param test_name: name of test case for printing: test, valid, train
        :return: r_mr, r_mrr, r_hit1, r_hit3, r_hit10, f_mr, f_mrr, f_hit1, f_hit3, f_hit10
        """
        base_time = time.time()  # in seconds
        self.model.eval()  # set model to eval mode

        if not self.config.model_inv:
            total = len(triples) // 2  # data inverse, triples duplicate
        else:
            raise NotImplementedError('Score computation for model_inv 2 ways is not implemented in this version yet.')

        known_hr_t, known_tr_h = self.data.build_known_triples_dict(triples)  # only use hr_t because inv data
        known_hr, known_tr = list(known_hr_t.keys()), list(known_tr_h.keys())  # only use hr because inv data

        # Compute the rank of every triple; ranks size = 2x triples size, for tail ranks and head ranks
        raw_ranks, ranks, raw_ranks_max, ranks_max = np.zeros(2 * total), np.zeros(2 * total), np.zeros(2 * total), np.zeros(2 * total)  # init zero to used in on-progress report, raw_ranks are min rank of queries, ranks are filtered out known triples, raw_ranks_max of queries are to compute mean rank
        if self.config.eval_intype:
            raw_ranks_intype, ranks_intype, raw_ranks_max_intype, ranks_max_intype = np.zeros(2 * total), np.zeros(2 * total), np.zeros(2 * total), np.zeros(2 * total)  # filtered by entity type
        if self.config.print_each_rel:
            relid_rank_indices = {self.data.rel_id[rel]: [] for rel in sorted(self.data.rel_id.keys())}  # {relid: [indices mask of result ranks]}, id includes inv rel, sorted by rel name for convenience
        if write_ranks:
            l_raw_ranks, l_ranks, l_raw_ranks_max, l_ranks_max = [], [], [], []

        count = 0
        for idx in range(0, len(known_hr), self.config.batch_size):  # for batch of hr, loop over data set once
            hr = np.array(known_hr[idx:idx + self.config.batch_size], dtype=np.int64, order='F')  # list to numpy
            h, r = utils.to_torch(hr[:, 0], self.config.device), utils.to_torch(hr[:, 1], self.config.device)
            with torch.no_grad():
                score = self.model({'h': h, 'r': r, 'partition': partition})  # (batch, num_ent)
            h, r, score = utils.to_numpy(h), utils.to_numpy(r), utils.to_numpy(score)  # TODO: maybe can use torch tensor directly

            for row in range(len(score)):  # for each hr row: rank all positive t (cover all triples, both directions)
                score_raw = score[row, :]
                score_known = score[row, self.data.known_hr_t_all[(h[row], r[row])]]
                if self.config.eval_intype:
                    score_raw_intype = score[row, self.data.relid_entid_intype[r[row]]]
                    score_known_intype = score[row, list(set(self.data.known_hr_t_all[(h[row], r[row])]) & set(self.data.relid_entid_intype[r[row]]))]  # TODO: may pre-convert all sets instead of converting here to save time
                for t in known_hr_t[(h[row], r[row])]:  # for each positive t of this hr: compute rank
                    if self.config.eval_scoreorder in ['min', 'mean']:
                        raw_ranks[count] = 1 + np.sum(score_raw > score[row, t])  # t should have largest score: count how many score larger than t; min rank: strictly larger
                        ranks[count] = raw_ranks[count] - np.sum(score_known > score[row, t])  # remove known and ranked better, not include t because strictly larger
                    if self.config.eval_scoreorder in ['max', 'mean']:
                        raw_ranks_max[count] = 1 + (np.sum(score_raw >= score[row, t]) - 1)  # max rank: larger or equal, minus 1 for t itself
                        ranks_max[count] = raw_ranks_max[count] - (np.sum(score_known >= score[row, t]) - 1)  # remove known and ranked better or equal, minus 1 for t itself
                    if self.config.eval_intype:
                        if self.config.eval_scoreorder in ['min', 'mean']:
                            raw_ranks_intype[count] = 1 + np.sum(score_raw_intype > score[row, t])
                            ranks_intype[count] = raw_ranks_intype[count] - np.sum(score_known_intype > score[row, t])
                        if self.config.eval_scoreorder in ['max', 'mean']:
                            raw_ranks_max_intype[count] = 1 + (np.sum(score_raw_intype >= score[row, t]) - 1)
                            ranks_max_intype[count] = raw_ranks_max_intype[count] - (np.sum(score_known_intype >= score[row, t]) - 1)
                    if self.config.print_each_rel:
                        relid_rank_indices[r[row]].append(count)  # mask of ranks of each rel (and each direction via inv)
                    if write_ranks:
                        l_raw_ranks.append((h[row], t, r[row], raw_ranks[count]))
                        l_ranks.append((h[row], t, r[row], ranks[count]))
                        l_raw_ranks_max.append((h[row], t, r[row], raw_ranks_max[count]))
                        l_ranks_max.append((h[row], t, r[row], ranks_max[count]))
                    count += 1  # next positive triple

        assert count == 2 * total, 'Check missing count in evaluation'

        def get_final_ranks(raw_ranks, ranks, raw_ranks_max, ranks_max):
            if self.config.eval_scoreorder == 'min':
                return raw_ranks, ranks
            elif self.config.eval_scoreorder == 'max':
                return raw_ranks_max, ranks_max
            elif self.config.eval_scoreorder == 'mean':
                return 1 / 2 * (raw_ranks + raw_ranks_max), 1 / 2 * (ranks + ranks_max)

        def get_eval_result(raw_ranks, ranks, eval_result={}, add_str=''):
            eval_result['raw%s' % add_str] = utils.metrics(raw_ranks)  # main result
            eval_result['filter%s' % add_str] = utils.metrics(ranks)
            if self.config.print_each_rel:
                for (relid, rank_indices) in relid_rank_indices.items():  # each rel result
                    eval_result['raw%s__%s' % (add_str, self.data.id_rel[relid])] = utils.metrics(raw_ranks[rank_indices])
                    eval_result['filter%s__%s' % (add_str, self.data.id_rel[relid])] = utils.metrics(ranks[rank_indices])
            return eval_result

        raw_ranks, ranks = get_final_ranks(raw_ranks, ranks, raw_ranks_max, ranks_max)
        eval_result = get_eval_result(raw_ranks, ranks)
        if self.config.eval_intype:
            raw_ranks_intype, ranks_intype = get_final_ranks(raw_ranks_intype, ranks_intype, raw_ranks_max_intype, ranks_max_intype)
            eval_result = get_eval_result(raw_ranks_intype, ranks_intype, eval_result=eval_result, add_str='_intype')
        if write_ranks:
            l_raw_ranks, l_ranks, l_raw_ranks_max, l_ranks_max = np.array(l_raw_ranks), np.array(l_ranks), np.array(l_raw_ranks_max), np.array(l_ranks_max)
            l_raw_ranks[:, 3], l_ranks[:, 3] = get_final_ranks(l_raw_ranks[:, 3], l_ranks[:, 3], l_raw_ranks_max[:, 3], l_ranks_max[:, 3])
            l_ranks = pd.DataFrame(l_ranks, columns=['h', 't', 'r', 'rank'])
            # l_ranks = l_ranks.sort_values(['r', 'h', 't'])  # no need standard order, choose visualizing order later
            write_path = os.path.join(self.config.params_out_path, self.config.exp_id)
            os.makedirs(write_path, exist_ok=True)
            file_path = os.path.join(write_path, 'ranks.part=%s.txt' % 'full' if partition is None else str(partition))
            l_ranks.to_csv(file_path, sep=' ', header=True, index=False, mode='w')

        if verbose:
            self.logger.info('%s\n' % self.format_eval_result_multiline_print(eval_result, test_name, additional_str='%i triples, %f seconds' % (total, time.time() - base_time)))

        self.model.train()  # set model to train mode
        return eval_result

    def format_eval_result_tuple(self, eval_result_tuple):
        return '%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f' % tuple(eval_result_tuple)

    def format_eval_result_line(self, line, eval_result):
        return '%s:\t%s\n' % (line, self.format_eval_result_tuple(eval_result[line]))

    def format_eval_result_multiline(self, eval_result, print_each_rel=False):
        result_str = 'Metric\tmr\tmrr\thit1\thit3\thit10\n'
        for line in eval_result.keys():
            if '__' not in line:  # main result line
                result_str += self.format_eval_result_line(line, eval_result)
            elif print_each_rel:  # each rel line
                result_str += self.format_eval_result_line(line, eval_result)
        return result_str

    def format_eval_result_multiline_print(self, eval_result, test_name, additional_str='', print_each_rel=False):
        print_str = ('='*50 + '\n'
                     + 'Result on %s%s:\n' % (test_name, ' (%s)' % additional_str if additional_str else '')
                     + str(self.format_eval_result_multiline(eval_result, print_each_rel))
                     + '='*50)
        return print_str

    def show_link_prediction(self, h, t, r, raw=True):
        """
        Show top tail and top head predictions for this triple (h, t, r)
        """
        self.model.eval()  # set model to eval mode

        if raw:
            r_inv = self.data.rel_id[r + '_inv']  # raw, get _inv before convert to id
            h = self.data.ent_id[h]
            t = self.data.ent_id[t]
            r = self.data.rel_id[r]
        else:
            r_inv = self.data.rel_id[self.data.id_rel[r] + '_inv']

        h, t, r, r_inv = np.array([h]), np.array([t]), np.array([r]), np.array([r_inv])  # number to numpy
        h, t, r, r_inv = utils.to_torch(h, self.config.device), utils.to_torch(t, self.config.device), utils.to_torch(r, self.config.device), utils.to_torch(r_inv, self.config.device)

        with torch.no_grad():
            # batch predict score, to numpy, argsort get the top prediction index
            top_tid = (utils.to_numpy(-self.model({'h': h, 'r': r})).reshape(-1)).argsort()  # argsort() return indices of values sorted ascending, -predict makes score descending, the indices is also ent id.
            top_hid = (utils.to_numpy(-self.model({'h': t, 'r': r_inv})).reshape(-1)).argsort()  # argsort() return indices of values sorted ascending, -predict makes score descending, the indices is also ent id.

        h, t, r, r_inv = utils.to_numpy(h), utils.to_numpy(t), utils.to_numpy(r), utils.to_numpy(r_inv)
        h, t, r, r_inv = h.item(), t.item(), r.item(), r_inv.item()  # back to number

        print('Prediction for groundtruth (%i, %i, %i) : (%s, %s, %s)' % (h, t, r, self.data.id_ent[h], self.data.id_ent[t], self.data.id_rel[r]))
        print('Top predicted tail:')
        [print('\t%i. %i : %s' % ((top + 1), id, self.data.id_ent[id])) for top, id in enumerate(top_tid[:10])]
        print('Top predicted head:')
        [print('\t%i. %i : %s' % ((top + 1), id, self.data.id_ent[id])) for top, id in enumerate(top_hid[:10])]

        self.model.train()  # set model to train mode

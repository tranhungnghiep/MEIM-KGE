#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Hung-Nghiep Tran

"""Reference:
Hung-Nghiep Tran and Atsuhiro Takasu (2020),
Multi-Partition Embedding Interaction with Block Term Format for Knowledge Graph Completion,
In Proceedings of the European Conference on Artificial Intelligence (ECAI'20).

Hung-Nghiep Tran and Atsuhiro Takasu (2022),
MEIM: Multi-partition Embedding Interaction Beyond Block Term Format for Efficient and Expressive Link Prediction,
In Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI'22).
"""

import time

import numpy as np
import torch

import utils


class MEI(torch.nn.Module):
    def __init__(self, data, config):
        super().__init__()

        self.data = data
        self.config = config

        self.ent_embs, self.rel_embs = self.embedding_def()
        self.wv = self.param_def()

        self.dropout_w = torch.nn.Dropout(self.config.droprate_w)  # usually not used
        self.dropout_r = torch.nn.Dropout(self.config.droprate_r)
        self.dropout_mr = torch.nn.Dropout(self.config.droprate_mr)
        self.dropout_h = torch.nn.Dropout(self.config.droprate_h)
        self.dropout_mrh = torch.nn.Dropout(self.config.droprate_mrh)
        self.dropout_t = torch.nn.Dropout(self.config.droprate_t)  # usually not used
        self.ortho_dropout_mr = torch.nn.Dropout(self.config.ortho_droprate_mr)  # only test with ortho
        self.rowrelnorm_dropout_r = torch.nn.Dropout(self.config.rowrelnorm_droprate_r)  # only test with ortho

        if self.config.norm == 'bn':
            # batchnorm's num_features: each feature to normalize over all batch dimension
            if self.config.n_sepK:
                self.n_r_shape = self.config.K * self.config.Cr
                self.n_mr_shape = self.config.K * self.config.Ce * self.config.Ce
                self.n_h_shape = self.config.K * self.config.Ce
                self.n_mrh_shape = self.config.K * self.config.Ce
                self.n_t_shape = self.config.K * self.config.Ce
            else:
                self.n_r_shape = self.config.Cr
                self.n_mr_shape = self.config.Ce * self.config.Ce
                self.n_h_shape = self.config.Ce
                self.n_mrh_shape = self.config.Ce
                self.n_t_shape = self.config.Ce
            self.norm_w = None  # not used batchnorm, not make sense and will not work in pytorch
            self.norm_r = torch.nn.BatchNorm1d(self.n_r_shape, momentum=self.config.bn_momentum, eps=self.config.n_epsilon)  # usually not used
            self.norm_mr = torch.nn.BatchNorm1d(self.n_mr_shape, momentum=self.config.bn_momentum, eps=self.config.n_epsilon)  # usually not used
            self.norm_h = torch.nn.BatchNorm1d(self.n_h_shape, momentum=self.config.bn_momentum, eps=self.config.n_epsilon)
            self.norm_mrh = torch.nn.BatchNorm1d(self.n_mrh_shape, momentum=self.config.bn_momentum, eps=self.config.n_epsilon)
            self.norm_t = torch.nn.BatchNorm1d(self.n_t_shape, momentum=self.config.bn_momentum, eps=self.config.n_epsilon)  # usually not used
        elif self.config.norm == 'ln':
            # layernorm's normalized_shape: other dimension except batch dimension
            if not self.config.n_sepK:
                self.n_w_shape = self.wv.numel()
                self.n_r_shape = self.config.K * self.config.Cr
                self.n_mr_shape = self.config.K * self.config.Ce * self.config.Ce
                self.n_h_shape = self.config.K * self.config.Ce
                self.n_mrh_shape = self.config.K * self.config.Ce
                self.n_t_shape = self.config.K * self.config.Ce
            else:
                self.n_w_shape = self.config.Cr * self.config.Ce * self.config.Ce
                self.n_r_shape = self.config.Cr
                self.n_mr_shape = self.config.Ce * self.config.Ce
                self.n_h_shape = self.config.Ce
                self.n_mrh_shape = self.config.Ce
                self.n_t_shape = self.config.Ce
            self.norm_w = torch.nn.LayerNorm(self.n_w_shape, eps=self.config.n_epsilon)
            self.norm_r = torch.nn.LayerNorm(self.n_r_shape, eps=self.config.n_epsilon)
            self.norm_mr = torch.nn.LayerNorm(self.n_mr_shape, eps=self.config.n_epsilon)
            self.norm_h = torch.nn.LayerNorm(self.n_h_shape, eps=self.config.n_epsilon)
            self.norm_mrh = torch.nn.LayerNorm(self.n_mrh_shape, eps=self.config.n_epsilon)
            self.norm_t = torch.nn.LayerNorm(self.n_t_shape, eps=self.config.n_epsilon)

    def embedding_def(self):
        """
        Define embedding matrices.
        All embeddings are defined in vector format of size k*c.
        """
        ent_embs = torch.nn.Parameter(torch.empty(len(self.data.ents), self.config.K * self.config.Ce,
                                                  dtype=torch.float, device=self.config.device, requires_grad=True))
        rel_embs = torch.nn.Parameter(torch.empty(len(self.data.rels), self.config.K * self.config.Cr,
                                                  dtype=torch.float, device=self.config.device, requires_grad=True))
        with torch.no_grad():
            if self.config.init_emb == 1:
                torch.nn.init.xavier_normal_(ent_embs.data, gain=self.config.init_emb_gain)  # linear, sigmoid, tanh use xavier std=gain*sqrt(2/(fanin+fanout)); for relu use he (~double variance of xavier to compensate for relu); use xavier and gain 1e-2 to reproduce old tensorflow code
                torch.nn.init.xavier_normal_(rel_embs.data, gain=self.config.init_emb_gain)
            elif self.config.init_emb == 2:
                torch.nn.init.normal_(ent_embs.data, 0., np.sqrt(1./(self.config.K * self.config.Ce)))  # use 1/fanout init, so sum all emb dims var 1, and sum each partition var 1/k
                torch.nn.init.normal_(rel_embs.data, 0., np.sqrt(1./(self.config.K * self.config.Cr)))
            elif self.config.init_emb == 3:
                torch.nn.init.normal_(ent_embs.data, 0., np.sqrt(1./self.config.Ce))  # use separate partition fanout init, so sum each partition var 1, just try it
                torch.nn.init.normal_(rel_embs.data, 0., np.sqrt(1./self.config.Cr))
            else:
                raise NotImplementedError('Unsupported initialization.')

        return ent_embs, rel_embs

    def param_def(self):
        """
        Define weight vector used for combining embeddings.
        """
        if self.config.core_tensor == 'shared':
            wv = torch.nn.Parameter(torch.empty(self.config.Cr * self.config.Ce * self.config.Ce,
                                                dtype=torch.float, device=self.config.device, requires_grad=True))
        if self.config.core_tensor == 'nonshared':
            wv = torch.nn.Parameter(torch.empty(self.config.K * self.config.Cr * self.config.Ce * self.config.Ce,
                                                dtype=torch.float, device=self.config.device, requires_grad=True))
        with torch.no_grad():
            if self.config.init_w == 0:
                utils.truncated_normal_(wv.data, -1, 1, 0.0, 0.5)  # shape similar to normal(0, 0.5), but no outlier tail
            elif self.config.init_w == 1:
                torch.nn.init.normal_(wv.data, 0., np.sqrt(self.config.K * self.config.K))  # sum, emb 1/kc
            elif self.config.init_w == 2:
                torch.nn.init.normal_(wv.data, 0., np.sqrt(1. / self.config.K))  # sum, emb 1/c
            elif self.config.init_w == 3:
                torch.nn.init.normal_(wv.data, 0., np.sqrt(self.config.K))  # mean, emb 1/kc
            elif self.config.init_w == 4:
                torch.nn.init.normal_(wv.data, 0., 1.)  # mean, emb 1/c
            else:
                raise NotImplementedError('Unsupported initialization.')

        return wv

    def forward(self, kwargs: {str: torch.Tensor}) -> torch.Tensor:
        """
        Dispatch call to compute score for each scoring strategy:
        htr score, hr to all T score, tr to all H score, ht to all R score, htr to all hrT and trH score

        :param kwargs: h, t, r... size (batch,)
        :return: score, shape depends on scoring strategy
        """
        # save reference to 1-hot input tensor to use later (in computing losses);
        # copied to gpu; no update, no require_grad, no tracking by optimizer
        self.h = kwargs['h']  # save reference for adaptive weight decay
        self.r = kwargs['r']
        self.sample_t = kwargs['sample_t'] if 'sample_t' in kwargs else utils.to_torch([], self.config.device)
        self.partition = kwargs['partition'] if 'partition' in kwargs else None

        score = self.compute_score(self.h, self.r)

        if self.config.combine_score == 'mean':
            score = score / self.config.K

        return score

    def compute_score(self, h: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """
        Main logic: compute the score.

        Input: tensor in batch: (batch,) of indices (1-hot vector)
        :return: tensor in batch: score: (batch, num_ent)
        """

        M = self.wv  # (k*cr*ce*ce) or (cr*ce*ce) for nonshared or shared core
        if self.config.n_w and self.config.norm != 'bn':
            M = M.view(-1, self.n_w_shape)
            M = self.norm_w(M)
        M = self.dropout_w(M)

        r = self.rel_embs[r]  # (batch, k*cr)
        if self.config.n_r:
            r = r.view(-1, self.n_r_shape)
            r = self.norm_r(r)
        r = self.dropout_r(r)  # drop any, not care about shape

        Mr = self.get_mr(r, M)  # (batch, k, ce, ce)
        if self.config.mapping_constraint:  # save reference for mapping constraint, before drop and bn on Mr
            if self.config.ortho_by_w:  # compute separate mr by passing in r.detach(), so ortho only update w, not r
                self.mr = self.get_mr(r.detach(), M)
            else:  # reuse to save computation
                self.mr = Mr
        if self.config.n_mr:
            Mr = Mr.view(-1, self.n_mr_shape)
            Mr = self.norm_mr(Mr)
        Mr = self.dropout_mr(Mr)
        Mr = Mr.view(-1, self.config.Ce, self.config.Ce)  # (batch*k, ce, ce)

        h = self.ent_embs[h]  # (batch, k*ce)
        if self.config.n_h:
            h = h.view(-1, self.n_h_shape)
            h = self.norm_h(h)
        h = self.dropout_h(h)
        h = h.view(-1, 1, self.config.Ce)  # (batch*k, 1, ce); add dim: row vector h; batch*k match mr for batch matmul

        Mrh = torch.matmul(h, Mr)  # (batch*k, 1, ce); only same item same k interact: correct mein.
        if self.config.n_mrh:
            Mrh = Mrh.view(-1, self.n_mrh_shape)
            Mrh = self.norm_mrh(Mrh)
        Mrh = self.dropout_mrh(Mrh)
        Mrh = Mrh.view(-1, self.config.K * self.config.Ce)  # (batch, k*ce)
        if self.partition is not None:
            Mrh = Mrh.view(-1, self.config.K, self.config.Ce)[:, self.partition, :].squeeze(dim=1)  # (batch, k, ce) -> (batch, 1, ce) -> (batch, ce)

        t = self.ent_embs  # (num_ent, k*ce)
        t = t[self.sample_t] if len(self.sample_t) > 0 else t  # mask select tails for sampled scoring
        if self.config.n_t:
            t = t.view(-1, self.n_t_shape)
            t = self.norm_t(t)
        t = self.dropout_t(t)
        t = t.view(-1, self.config.K * self.config.Ce)  # (num_ent, k*ce); unrolled the same way as Mrh: correct mein.
        t = t.permute(1, 0)  # (k*ce, num_ent); transpose for dot product matmul
        if self.partition is not None:
            t = t.view(self.config.K, self.config.Ce, -1)[self.partition, :, :].squeeze(dim=0)  # (k, ce, num_ent) -> (1, ce, num_ent) -> (ce, num_ent)

        Mrht = torch.matmul(Mrh, t)  # (batch, num_ent)

        score = Mrht

        return score

    def get_mr(self, r, M):
        """
        Compute Mr by einsum notation.

        :param r: rel emb (batch, k, cr), already drop or bn, already detach
        :param M: core tensor as weight matrix of hypernetwork (k, r, ce*ce), or (1, r, ce*ce) if shared core
        :return: Mr (batch, k, ce, ce)
        """
        r = r.view(-1, self.config.K, self.config.Cr)  # (batch, k, cr); reshape unroll from inner most mode out
        M = M.view(-1, self.config.Cr, self.config.Ce * self.config.Ce)  # (k, cr, ce*ce) or (1, cr, ce*ce)
        if self.wv.numel() == self.config.K * self.config.Cr * self.config.Ce * self.config.Ce:  # NONSHARED CORE
            Mr = torch.einsum('bkr,kre->bke', r, M)  # (batch, k, ce*ce)
            Mr = Mr.contiguous()  # fix "non contiguous error" when view() after nonshared core einsum, will copy data
        elif self.wv.numel() == self.config.Cr * self.config.Ce * self.config.Ce:  # SHARED CORE
            Mr = torch.einsum('bkr,re->bke', r, M.squeeze())  # (batch, k, ce*ce)

        return Mr.view(-1, self.config.K, self.config.Ce, self.config.Ce)  # (batch, k, ce, ce)

    def compute_loss_total(self, score: torch.Tensor, y: torch.Tensor, active_e: torch.Tensor):
        """
        Combine and return total loss of the model

        score: model prediction raw logits
        y: true data distribution

        NOTE: using self.h, self.r, self.mr reference to the cached tensors in previous self.forward() call
        """

        total_loss = self.compute_loss_main(score, y)

        # WEIGHT DECAY:
        # we will weight decay active entities, including h and positive t, already prepared in the sampling methods
        total_loss += self.compute_loss_weightdecay(active_e, self.r)

        # MAPPING CONSTRAINT:
        if self.config.mapping_constraint:  # self.mr exists
            total_loss += self.compute_loss_mappingconstraint(self.mr, self.r)

        # SOFT CONSTRAINT:
        total_loss += self.compute_loss_softconstraint(self.r)

        return total_loss

    def compute_loss_main(self, score: torch.Tensor, y: torch.Tensor):
        """
        Compute cross-entropy loss: binary, softmax...
        with label smoothing, score scaling,...

        score: (batch, num_ent), raw logit
        y: (batch, num_ent), float in [0.0, 1.0] represents data distribution

        :return: mean over batch loss
        """
        assert score.shape == y.shape, 'Model output score and data distribution y must be the same shape'

        if self.config.shift_score != 0:
            # shift score if not centralized around 0, like -norm
            score = score + self.config.shift_score
        if self.config.scale_score != 1:
            # scale up score magnitude for faster softplus gradient
            score = score * self.config.scale_score

        # normalize y to distribution for multinomial (softmax) cross entropy, before label smooth
        y_distribution = y / y.sum(dim=1, keepdim=True)

        if self.config.label_smooth > 0:
            if self.config.label_smooth_style == 'tensorflow':
                y = (1.0 - self.config.label_smooth) * y + self.config.label_smooth / 2
                y_distribution = (1.0 - self.config.label_smooth) * y_distribution + self.config.label_smooth / 2
            elif self.config.label_smooth_style == 'conve':
                # (https://github.com/TimDettmers/ConvE/blob/853d7010a4b569c3d24c9e6eeaf9dc3535d78832/main.py#L156)
                y = (1.0 - self.config.label_smooth) * y + 1.0 / len(self.data.ents)
                y_distribution = (1.0 - self.config.label_smooth) * y_distribution + 1.0 / len(self.data.ents)

        # compute and return loss to backward to compute gradient, minimize with optimizer later
        # loss for one mini batch. Note: mean
        if self.config.loss_mode == 'cross-entropy':
            # directly binary cross-entropy
            loss = torch.nn.functional.binary_cross_entropy_with_logits(input=score, target=y, reduction='mean')

        elif self.config.loss_mode == 'softplus':
            # rewrite loss by softplus, change label from 1/0 to 1/-1
            # softplus is the same as bare binary cross-entropy: pushes score > 0 for y==1, < 0 for y==-1
            y_polar = y * 2 - 1
            loss = ((-y_polar * score).exp() + 1).log().mean()

        elif self.config.loss_mode == 'softmax-cross-entropy':
            # this can work with 1-vs-all or k-vs-all y sampling.
            loss = utils.softmax_cross_entropy_with_softtarget(input=score, target=y_distribution, reduction='mean')

        elif self.config.loss_mode == 'mix-cross-entropy':
            # here try to mix binary and softmax cross-entropy;
            loss_binary = torch.nn.functional.binary_cross_entropy_with_logits(input=score, target=y, reduction='mean')
            loss_multi = utils.softmax_cross_entropy_with_softtarget(input=score, target=y_distribution, reduction='mean')
            loss = self.config.binary_weight * loss_binary + (1 - self.config.binary_weight) * loss_multi

        elif self.config.loss_mode == 'weightcorrected-softmax-cross-entropy':
            # this is the new trick, to approximate the single positive class softmax in lacroix.
            num_positive = y.sum(dim=1)
            loss = (num_positive * utils.softmax_cross_entropy_with_softtarget(input=score, target=y_distribution, reduction='none')).mean()

        return loss

    def compute_loss_weightdecay(self, e: torch.Tensor, r: torch.Tensor):
        """
        Compute weight decay regularization loss if applicable.
        Aready applied weight lambda. Separately for better tuning lambda.
        emb l2 reg loss, only on active emb to reduce computation.

        :return: MEAN OVER BATCH, SUM OVER FEATURE L_p^p loss
        """
        loss = 0.0

        def get_weight(x):
            """
            For adaptive reg weight: frequency of each ent or rel in a batch
            (temperature=1.: raw triple weight (default), temperature=0.: raw batch weight (every one has weight 1));
            :param x: entities or relations
            :return: unique entities or relations, weights for the unique entities or relations
            """
            unique, count = torch.unique(x, sorted=False, return_counts=True)
            weight = count.float() ** self.config.reg_temp
            if self.config.reg_weightedsum:  # weighted sum over batch is mean over batch, otherwise is sum over batch
                weight /= weight.sum()  # normalize weight to sum 1.0
            return unique, weight

        def get_factor(x):
            """
            Get factor for weight decay: raw emb entry, or rownorm
            :param x: entities or relations
            :return: factor
            """
            factor = x.view(x.shape[0], -1)  # (batch, k*c)
            if self.config.reg_decayedfactor == 'rownorm':
                factor = factor.view(factor.shape[0], self.config.K, -1)  # (batch, k, c)
                factor = torch.norm(factor, p=2.0, dim=2)  # decay emb row Frobenius norm (N3 for ComplEx)
            return factor

        if self.config.lambda_ent > 0:
            unique, weight = get_weight(e)
            factor = get_factor(self.ent_embs[unique])
            loss += self.config.lambda_ent * torch.sum(weight.detach()
                                                      * torch.sum(torch.abs(factor) ** self.config.reg_pnorm, dim=1))

        if self.config.lambda_rel > 0:
            unique, weight = get_weight(r)
            factor = get_factor(self.rel_embs[unique])
            loss += self.config.lambda_rel * torch.sum(weight.detach()
                                                      * torch.sum(torch.abs(factor) ** self.config.reg_pnorm, dim=1))

        if self.config.lambda_params > 0:
            loss += self.config.lambda_params * torch.sum(torch.abs(self.wv) ** self.config.reg_pnorm)

        return loss

    def compute_loss_mappingconstraint(self, mr: torch.Tensor, r: torch.Tensor=None):
        """
        Compute soft constraint regularization loss such as soft orthogonality (mei) if applicable.

        mr: (batch, k, ce, ce)
        r: index of rel (batch, )
        :return: MEAN OVER BATCH, SUM OVER FEATURE L_2^2 loss vs identity matrix, already multiplied lambda weight
        """
        loss = 0.0

        if self.config.lambda_ortho > 0:
            def get_orthogonal_loss(M, I, ortho_p):
                """
                M: (batch, k, ce, ce)
                I: (batch, k, ce, ce)

                :return: scalar (), SUM OVER FEATURES AND PARTITIONS, MEAN OVER BATCHES, L_2^2 loss
                """
                return ((torch.matmul(M.permute(0, 1, 3, 2), M) - I).abs() ** ortho_p).sum(dim=[1, 2, 3]).mean()  # sum over features and partitions, mean over batches

            identity = torch.eye(self.config.Ce, dtype=torch.float, device=self.config.device, requires_grad=False)
            identity = identity.view(1, 1, self.config.Ce, self.config.Ce)
            identity = identity.repeat(mr.shape[0], mr.shape[1], 1, 1)  # (batch, k, ce, ce)

            if self.config.n_mr:
                mr_shape_backup = mr.shape
                mr = mr.view(-1, self.n_mr_shape)
                mr = self.norm_mr(mr)
                mr = mr.view(mr_shape_backup)
            mr = self.ortho_dropout_mr(mr)  # drop can be separate, but bn must be the same as compute_score
            if self.config.ortho_dim == 'col':
                loss += self.config.lambda_ortho * get_orthogonal_loss(mr, identity, ortho_p=self.config.ortho_p)
            elif self.config.ortho_dim == 'row':
                loss += self.config.lambda_ortho * get_orthogonal_loss(mr.permute(0, 1, 3, 2), identity, ortho_p=self.config.ortho_p)
            elif self.config.ortho_dim == 'both':
                loss_col = get_orthogonal_loss(mr, identity, ortho_p=self.config.ortho_p)
                loss_row = get_orthogonal_loss(mr.permute(0, 1, 3, 2), identity, ortho_p=self.config.ortho_p)
                loss += self.config.lambda_ortho * (loss_col + loss_row) / 2.0

        return loss

    def compute_loss_softconstraint(self, r: torch.Tensor=None):
        """
        Compute soft constraint regularization loss such as relation row unit norm if applicable.

        r: index of rel (batch, )
        :return: MEAN OVER BATCH, SUM OVER FEATURE, already multiplied lambda weight
        """
        loss = 0.0

        if self.config.lambda_rowrelnorm > 0:  # push mr ortho but additionally push rowrel norm
            def get_rownorm_loss(x, row_c, row_p):
                """
                Push row_p norm of each row (batch, k, :) close to row_c.
                x: (batch, k, c)
                return: scalar, sum over k, mean over batch
                """
                return ((torch.norm(x, p=2.0, dim=-1) - row_c).abs() ** row_p).sum(dim=-1).mean()

            r_emb = self.rel_embs[r].view(-1, self.config.K, self.config.Cr)
            if self.config.n_r:
                r_shape = r_emb.shape
                r_emb = r_emb.view(-1, self.n_r_shape)
                r_emb = self.norm_r(r_emb)
                r_emb = r_emb.view(r_shape)
            r_emb = self.rowrelnorm_dropout_r(r_emb)  # drop can be separate, but bn must be the same as compute_score
            rowrelnorm_loss = get_rownorm_loss(x=r_emb, row_c=self.config.rowrelnorm_c, row_p=self.config.rowrelnorm_p)
            loss += self.config.lambda_rowrelnorm * rowrelnorm_loss

        return loss

    def enforce_hardconstraint(self, e: torch.Tensor=None, r: torch.Tensor=None):
        """
        Constraint on embedding vector, such as unit norm, non negative
        """
        with torch.no_grad():
            if 'ent' in self.config.to_constrain:
                if e is None:
                    emb = self.ent_embs.view(-1, self.config.K, self.config.Ce)
                else:
                    e = torch.unique(e, sorted=False)
                    emb = self.ent_embs.view(-1, self.config.K, self.config.Ce)[e]
                if 'nonneg' in self.config.constraint:
                    emb[:] = utils.nonneg(emb)
                if 'unitnorm' in self.config.constraint:
                    emb[:] = utils.unitnorm(emb, self.config.constrain_axis_ent)
                if 'minmaxnorm' in self.config.constraint:
                    emb[:] = utils.minmaxnorm(emb, self.config.constrain_axis_ent)

            if 'rel' in self.config.to_constrain:
                if r is None:
                    emb = self.rel_embs.view(-1, self.config.K, self.config.Cr)
                else:
                    r = torch.unique(r, sorted=False)
                    emb = self.rel_embs.view(-1, self.config.K, self.config.Cr)[r]
                if 'nonneg' in self.config.constraint:
                    emb[:] = utils.nonneg(emb)
                if 'unitnorm' in self.config.constraint:
                    emb[:] = utils.unitnorm(emb, self.config.constrain_axis_rel)
                if 'minmaxnorm' in self.config.constraint:
                    emb[:] = utils.minmaxnorm(emb, self.config.constrain_axis_rel)


MEIM = MEI  # alias, MEIM is MEI iMproved with suitable configurations.


class DistMult(MEI):
    def __init__(self, data, config):
        super().__init__(data, config)

    def compute_score(self, h: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """
        Main logic: compute the score.
        Input: tensor in batch: (batch,) of indices (1-hot vector)
        :return: tensor in batch: score: (batch, num_ent)
        """
        # Look up
        he = self.ent_embs[h]  # embedding vector format (batch, K*Ce)
        re = self.rel_embs[r]  # (batch, K*Cr)
        hem = he.view(-1, self.config.K, self.config.Ce)  # vector format -> matrix format (batch, K, Ce)
        rem = re.view(-1, self.config.K, self.config.Cr)  # (batch, K, Cr)
        h0 = hem[:, :, 0].squeeze()  # (batch, k)
        r0 = rem[:, :, 0].squeeze()

        t = self.ent_embs  # (num_ent, k*ce)
        t = t.view(-1, self.config.K, self.config.Ce)  # (num_ent, k, ce)
        t0 = t[:, :, 0].squeeze()  # (num_ent, k)
        t0 = t0.permute(1, 0)  # (k, num_ent)

        # Compute score = h0t0r0 = (h0*r0) @ t0
        score = torch.matmul(h0 * r0, t0)  # (batch, num_ent)

        return score


class CP(MEI):
    def __init__(self, data, config):
        super().__init__(data, config)

    def compute_score(self, h: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """
        Note: CPh is CP with data_inv 1. SimplE is CP with model_inv 1.
        To reproduce CPh: run CP with data_inv 1 (model_inv 0).
        To reproduce SimplE: run SimplE with either model_inv 0 (data_inv 1) or model_inv 1 (data_inv 0).

        Main logic: compute the score.
        Input: tensor in batch: (batch,) of indices (1-hot vector)
        :return: tensor in batch: score: (batch, num_ent)
        """
        # Look up
        he = self.ent_embs[h]  # embedding vector format (batch, K*Ce)
        re = self.rel_embs[r]  # (batch, K*Cr)
        hem = he.view(-1, self.config.K, self.config.Ce)  # vector format -> matrix format (batch, K, Ce)
        rem = re.view(-1, self.config.K, self.config.Cr)  # (batch, K, Cr)
        h0 = hem[:, :, 0].squeeze()  # (batch, k)
        r0 = rem[:, :, 0].squeeze()

        t = self.ent_embs  # (num_ent, k*ce)
        t = t.view(-1, self.config.K, self.config.Ce)  # (num_ent, k, ce)
        t1 = t[:, :, 1].squeeze()  # (num_ent, k)
        t1 = t1.permute(1, 0)  # (k, num_ent)

        # Compute score = h0t1r0 = (h0*r0) @ t1
        score = torch.matmul(h0 * r0, t1)  # (batch, num_ent)

        return score


class SimplE(MEI):
    def __init__(self, data, config):
        super().__init__(data, config)

    def compute_score(self, h: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """
        Note: CPh is CP with data_inv 1. SimplE is CP with model_inv 1.
        To reproduce CPh: run CP with data_inv 1 (model_inv 0).
        To reproduce SimplE: run SimplE with either model_inv 0 (data_inv 1) or model_inv 1 (data_inv 0).

        Main logic: compute the score.
        Input: tensor in batch: (batch,) of indices (1-hot vector)
        :return: tensor in batch: score: (batch, num_ent)
        """
        # Look up
        he = self.ent_embs[h]  # embedding vector format (batch, K*Ce)
        re = self.rel_embs[r]  # (batch, K*Cr)
        hem = he.view(-1, self.config.K, self.config.Ce)  # vector format -> matrix format (batch, K, Ce)
        rem = re.view(-1, self.config.K, self.config.Cr)  # (batch, K, Cr)
        h0 = hem[:, :, 0].squeeze()  # (batch, k)
        h1 = hem[:, :, 1].squeeze()
        r0 = rem[:, :, 0].squeeze()
        r1 = rem[:, :, 1].squeeze()

        t = self.ent_embs  # (num_ent, k*ce)
        t = t.view(-1, self.config.K, self.config.Ce)  # (num_ent, k, ce)
        t0 = t[:, :, 0].squeeze()  # (num_ent, k)
        t0 = t0.permute(1, 0)  # (k, num_ent)
        t1 = t[:, :, 1].squeeze()
        t1 = t1.permute(1, 0)

        # Compute score = h0t1r0 + h1t0r1 = (h0*r0) @ t1 + (h1r1) @ t0
        score = torch.matmul(h0 * r0, t1) \
              + torch.matmul(h1 * r1, t0)  # (batch, num_ent)

        return score


CPh = SimplE  # alias, not the same but equivalent at convergence after SGD training.


class ComplEx(MEI):
    def __init__(self, data, config):
        super().__init__(data, config)

    def compute_score(self, h: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """
        Main logic: compute the score.

        For RotatE, use constraint unitnorm rowrel.

        Input: tensor in batch: (batch,) of indices (1-hot vector)
        :return: tensor in batch: score: (batch, num_ent)
        """
        # Look up
        he = self.ent_embs[h]  # embedding vector format (batch, K*Ce)
        re = self.rel_embs[r]  # (batch, K*Cr)
        hem = he.view(-1, self.config.K, self.config.Ce)  # vector format -> matrix format (batch, K, Ce)
        rem = re.view(-1, self.config.K, self.config.Cr)  # (batch, K, Cr)
        h0 = hem[:, :, 0].squeeze()  # (batch, k)
        h1 = hem[:, :, 1].squeeze()
        r0 = rem[:, :, 0].squeeze()
        r1 = rem[:, :, 1].squeeze()

        t = self.ent_embs  # (num_ent, k*ce)
        t = t.view(-1, self.config.K, self.config.Ce)  # (num_ent, k, ce)
        t0 = t[:, :, 0].squeeze()  # (num_ent, k)
        t0 = t0.permute(1, 0)  # (k, num_ent)
        t1 = t[:, :, 1].squeeze()
        t1 = t1.permute(1, 0)

        # Compute score = h0t0r0 + h0t1r1 - h1t0r1 + h1t1r0 = (h0*r0-h1*r1) @ t0 + (h0*r1+h1*r0) @ t1
        score = torch.matmul(h0 * r0 - h1 * r1, t0) \
              + torch.matmul(h0 * r1 + h1 * r0, t1)  # (batch, num_ent)

        return score


class Quaternion(MEI):
    def __init__(self, data, config):
        super().__init__(data, config)

    def compute_score(self, h: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """
        Main logic: compute the score.

        Relation unit norm is important, with hard constraint unitnorm rowrel or soft constraint rowrelnorm_c = 1.

        Quaternion embedding was proposed in several papers, one of the first is:
        Hung-Nghiep Tran and Atsuhiro Takasu (2019),
        Analyzing Knowledge Graph Embedding Methods from a Multi-Embedding Interaction Perspective,
        In Proceedings of DSI4 at EDBT/ICDT'19.

        Input: tensor in batch: (batch,) of indices (1-hot vector)
        :return: tensor in batch: score: (batch, num_ent)
        """
        # Look up
        he = self.ent_embs[h]  # embedding vector format (batch, K*Ce)
        re = self.rel_embs[r]  # (batch, K*Cr)
        hem = he.view(-1, self.config.K, self.config.Ce)  # vector format -> matrix format (batch, K, Ce)
        rem = re.view(-1, self.config.K, self.config.Cr)  # (batch, K, Cr)
        h0 = hem[:, :, 0].squeeze()  # (batch, k)
        h1 = hem[:, :, 1].squeeze()
        h2 = hem[:, :, 2].squeeze()
        h3 = hem[:, :, 3].squeeze()
        r0 = rem[:, :, 0].squeeze()
        r1 = rem[:, :, 1].squeeze()
        r2 = rem[:, :, 2].squeeze()
        r3 = rem[:, :, 3].squeeze()

        t = self.ent_embs  # (num_ent, k*ce)
        t = t.view(-1, self.config.K, self.config.Ce)  # (num_ent, k, ce)
        t0 = t[:, :, 0].squeeze()  # (num_ent, k)
        t0 = t0.permute(1, 0)  # (k, num_ent)
        t1 = t[:, :, 1].squeeze()
        t1 = t1.permute(1, 0)
        t2 = t[:, :, 2].squeeze()
        t2 = t2.permute(1, 0)
        t3 = t[:, :, 3].squeeze()
        t3 = t3.permute(1, 0)

        # Compute score
        # = (h0r0-h1r1-h2r2-h3r3)@t0
        # + (h0r1+h1r0-h2r3+h3r2)@t1
        # + (h0r2+h1r3+h2r0-h3r1)@t2
        # + (h0r3-h1r2+h2r1+h3r0)@t3
        score = torch.matmul(h0 * r0 - h1 * r1 - h2 * r2 - h3 * r3, t0) \
              + torch.matmul(h0 * r1 + h1 * r0 - h2 * r3 + h3 * r2, t1) \
              + torch.matmul(h0 * r2 + h1 * r3 + h2 * r0 - h3 * r1, t2) \
              + torch.matmul(h0 * r3 - h1 * r2 + h2 * r1 + h3 * r0, t3)  # (batch, num_ent)

        return score


class W2V(MEI):
    def __init__(self, data, config):
        super().__init__(data, config)

    def compute_score(self, h: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """
        Main logic: compute the score.
        Input: tensor in batch: (batch,) of indices (1-hot vector)
        :return: tensor in batch: score: (batch, num_ent)
        """
        # Look up
        he = self.ent_embs[h]  # embedding vector format (batch, K*Ce)
        hem = he.view(-1, self.config.K, self.config.Ce)  # vector format -> matrix format (batch, K, Ce)
        h0 = hem[:, :, 0].squeeze()  # (batch, k)

        t = self.ent_embs  # (num_ent, k*ce)
        t = t.view(-1, self.config.K, self.config.Ce)  # (num_ent, k, ce)
        t1 = t[:, :, 1].squeeze()
        t1 = t1.permute(1, 0)

        # Compute score = h0t1 = h0 @ t1
        score = torch.matmul(h0, t1)  # (batch, num_ent); note: same as CP but ignore rel

        return score


class W2Vh(MEI):
    def __init__(self, data, config):
        super().__init__(data, config)

    def compute_score(self, h: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """
        Main logic: compute the score.
        Input: tensor in batch: (batch,) of indices (1-hot vector)
        :return: tensor in batch: score: (batch, num_ent)
        """
        # Look up
        he = self.ent_embs[h]  # embedding vector format (batch, K*Ce)
        hem = he.view(-1, self.config.K, self.config.Ce)  # vector format -> matrix format (batch, K, Ce)
        h0 = hem[:, :, 0].squeeze()  # (batch, k)
        h1 = hem[:, :, 1].squeeze()

        t = self.ent_embs  # (num_ent, k*ce)
        t = t.view(-1, self.config.K, self.config.Ce)  # (num_ent, k, ce)
        t0 = t[:, :, 0].squeeze()  # (num_ent, k)
        t0 = t0.permute(1, 0)  # (k, num_ent)
        t1 = t[:, :, 1].squeeze()
        t1 = t1.permute(1, 0)

        # Compute score = h0t1 + h1t0 = h0 @ t1 + h1 @ t0
        score = torch.matmul(h0, t1) \
              + torch.matmul(h1, t0)  # (batch, num_ent); note: same as SimplE (CPh) but ignore rel

        return score


class Random(MEI):
    def __init__(self, data, config):
        super().__init__(data, config)

    def compute_score(self, h: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """
        Main logic: compute the score.
        Input: tensor in batch: (batch,) of indices (1-hot vector)
        :return: tensor in batch: score: (batch, num_ent)
        """

        # random at uniform scores
        batch_size = h.shape[0]
        num_ent = self.ent_embs.shape[0]
        score = torch.rand(batch_size, num_ent, dtype=torch.float, device=self.config.device, requires_grad=False)  # (batch, num_ent)

        return score

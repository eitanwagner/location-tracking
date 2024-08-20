from typing import List, Optional

import torch
import torch.nn as nn
from torch import BoolTensor, FloatTensor, LongTensor

# based on TorchCRF with small modifications

class CRF(nn.Module):
    def __init__(
            self, num_labels: int, pad_idx: Optional[int] = None, use_gpu: bool = True, pad_idx_val=None, const=None,
            theta=None, weights=None, divide_h2=False
    ) -> None:
        """

        :param num_labels: number of labels
        :param pad_idxL padding index. default None
        :return None
        """

        if num_labels < 1:
            raise ValueError("invalid number of labels: {0}".format(num_labels))

        super().__init__()
        self.num_labels = num_labels
        self._use_gpu = torch.cuda.is_available() and use_gpu

        # transition matrix setting
        # transition matrix format (source, destination)
        self.trans_matrix = nn.Parameter(torch.empty(num_labels, num_labels))
        # transition matrix of start and end settings
        self.start_trans = nn.Parameter(torch.empty(num_labels))
        self.end_trans = nn.Parameter(torch.empty(num_labels))
        if theta is not None:
            self.theta = nn.Parameter(torch.tensor(theta, dtype=torch.float32))
        else:
            self.theta = None
        if weights is not None:
            self.weights = nn.Parameter(torch.tensor(weights, dtype=torch.float32))
        else:
            self.weights = None
        self.divide_h2 = divide_h2

        # ADDED HERE:
        self._initialize_parameters(pad_idx, pad_idx_val=pad_idx_val, const=const)

    def forward(
            self, h: FloatTensor, labels: LongTensor, mask: BoolTensor, h2=None
    ) -> FloatTensor:
        """

        :param h: hidden matrix (batch_size, seq_len, num_labels)
        :param labels: answer labels of each sequence
                       in mini batch (batch_size, seq_len)
        :param mask: mask tensor of each sequence
                     in mini batch (batch_size, seq_len)
        :return: The log-likelihood (batch_size)
        """


        # ADDED HERE:
        # h2 is of shape (batch_size, seq_len, num_labels, num_labels)
        # h2[b, t, l1, l2] is the log-probability to go from l1 to l2 at segment t (l1 is for segment t)

        # compute self.trans_matrix using a network. Then continue as before
        log_numerator = self._compute_numerator_log_likelihood(h, labels, mask, h2=h2)
        log_denominator = self._compute_denominator_log_likelihood(h, mask, h2=h2)

        return log_numerator - log_denominator

    def viterbi_decode(self, h: FloatTensor, mask: BoolTensor, h2=None) -> List[List[int]]:
        """
        decode labels using viterbi algorithm
        :param h: hidden matrix (batch_size, seq_len, num_labels)
        :param mask: mask tensor of each sequence
                     in mini batch (batch_size, batch_size)
        :return: labels of each sequence in mini batch
        """

        batch_size, seq_len, _ = h.size()
        # prepare the sequence lengths in each sequence
        seq_lens = mask.sum(dim=1)
        # In mini batch, prepare the score
        # from the start sequence to the first label
        score = [self.start_trans.data + h[:, 0]]
        path = []

        for t in range(1, seq_len):
            # extract the score of previous sequence
            # (batch_size, num_labels, 1)
            previous_score = score[t - 1].view(batch_size, -1, 1)

            # extract the score of hidden matrix of sequence
            # (batch_size, 1, num_labels)
            h_t = h[:, t].view(batch_size, 1, -1)

            # extract the score in transition
            # from label of t-1 sequence to label of sequence of t
            # self.trans_matrix has the score of the transition
            # from sequence A to sequence B
            # (batch_size, num_labels, num_labels)
            if h2 is None:
                score_t = previous_score + self.trans_matrix + h_t
            elif self.theta is None:
                score_t = previous_score + h2[torch.arange(batch_size), t-1, :, :].to(h_t.device) + h_t  # only for batch_size=1
            else:
                score_t = previous_score + (1-torch.sigmoid(self.theta)) * self.trans_matrix + \
                          torch.sigmoid(self.theta) * h2[torch.arange(batch_size), t-1, :, :].to(h_t.device) + h_t  # only for batch_size=1

            # keep the maximum value
            # and point where maximum value of each sequence
            # (batch_size, num_labels)
            best_score, best_path = score_t.max(1)
            score.append(best_score)
            path.append(best_path)

        # predict labels of mini batch
        best_paths = [
            self._viterbi_compute_best_path(i, seq_lens, score, path)
            for i in range(batch_size)
        ]

        return best_paths

    def _viterbi_compute_best_path(
            self,
            batch_idx: int,
            seq_lens: torch.LongTensor,
            score: List[FloatTensor],
            path: List[torch.LongTensor],
    ) -> List[int]:
        """
        return labels using viterbi algorithm
        :param batch_idx: index of batch
        :param seq_lens: sequence lengths in mini batch (batch_size)
        :param score: transition scores of length max sequence size
                      in mini batch [(batch_size, num_labels)]
        :param path: transition paths of length max sequence size
                     in mini batch [(batch_size, num_labels)]
        :return: labels of batch_idx-th sequence
        """

        seq_end_idx = seq_lens[batch_idx] - 1
        # extract label of end sequence
        _, best_last_label = (score[seq_end_idx][batch_idx] + self.end_trans).max(0)
        best_labels = [int(best_last_label)]

        # predict labels from back using viterbi algorithm
        for p in reversed(path[:seq_end_idx]):
            best_last_label = p[batch_idx][best_labels[0]]
            best_labels.insert(0, int(best_last_label))

        return best_labels

    def _compute_denominator_log_likelihood(self, h: FloatTensor, mask: BoolTensor, h2=None):
        """

        compute the denominator term for the log-likelihood
        :param h: hidden matrix (batch_size, seq_len, num_labels)
        :param mask: mask tensor of each sequence
                     in mini batch (batch_size, seq_len)
        :return: The score of denominator term for the log-likelihood
        """
        device = h.device
        batch_size, seq_len, _ = h.size()

        # (num_labels, num_labels) -> (1, num_labels, num_labels)
        trans = self.trans_matrix.unsqueeze(0)

        # add the score from beginning to each label
        # and the first score of each label
        score = self.start_trans + h[:, 0]

        # iterate through processing for the number of words in the mini batch
        for t in range(1, seq_len):
            # (batch_size, self.num_labels, 1)
            before_score = score.unsqueeze(2)

            # prepare t-th mask of sequences in each sequence
            # (batch_size, 1)
            mask_t = mask[:, t].unsqueeze(1)
            mask_t = mask_t.to(device)

            # prepare the transition probability of the t-th sequence label
            # in each sequence
            # (batch_size, 1, num_labels)
            h_t = h[:, t].unsqueeze(1)

            # calculate t-th scores in each sequence
            # (batch_size, num_labels)
            if h2 is None:
                score_t = before_score + h_t + trans
            elif self.theta is None:
                if self.divide_h2:
                    if t <= seq_len//5:
                        score_t = before_score + h_t + h2[0][torch.arange(batch_size), t-1, :, :]
                    elif t <= 2*(seq_len//5):
                        score_t = before_score + h_t + h2[1][torch.arange(batch_size), t-1-seq_len//5, :, :].to(device)
                    elif t <= 3*(seq_len//5):
                        score_t = before_score + h_t + h2[2][torch.arange(batch_size), t-1-2*(seq_len//5), :, :].to(device)
                    elif t <= 4*(seq_len//5):
                        score_t = before_score + h_t + h2[3][torch.arange(batch_size), t-1-3*(seq_len//5), :, :].to(device)
                    else:
                        score_t = before_score + h_t + h2[4][torch.arange(batch_size), t-1-4*(seq_len//5), :, :].to(device)
                else:
                    score_t = before_score + h_t + h2[torch.arange(batch_size), t-1, :, :]
            else:
                score_t = before_score + h_t + (1-torch.sigmoid(self.theta)) * trans + torch.sigmoid(self.theta) * h2[torch.arange(batch_size), t-1, :, :]
            score_t = torch.logsumexp(score_t, 1)

            # update scores
            # (batch_size, num_labels)
            score = torch.where(mask_t, score_t, score)

        # add the end score of each label
        score += self.end_trans

        # return the log likely food of all data in mini batch
        return torch.logsumexp(score, 1)

    def _compute_numerator_log_likelihood(
            self, h: FloatTensor, y: LongTensor, mask: BoolTensor, h2=None
    ) -> FloatTensor:
        """
        compute the numerator term for the log-likelihood
        :param h: hidden matrix (batch_size, seq_len, num_labels)
        :param y: answer labels of each sequence
                  in mini batch (batch_size, seq_len)
        :param mask: mask tensor of each sequence
                     in mini batch (batch_size, seq_len)
        :return: The score of numerator term for the log-likelihood
        """

        batch_size, seq_len, _ = h.size()

        h_unsqueezed = h.unsqueeze(-1)
        trans = self.trans_matrix.unsqueeze(-1)

        arange_b = torch.arange(batch_size)

        # extract first vector of sequences in mini batch
        calc_range = seq_len - 1
        if h2 is None:
            score = self.start_trans[y[:, 0]] + sum(
                [self._calc_trans_score_for_num_llh(
                    h_unsqueezed, y, trans, mask, t, arange_b
                ) for t in range(calc_range)])
        elif self.theta is None:
            if not self.divide_h2:
                score = self.start_trans[y[:, 0]] + sum(
                    [self._calc_trans_score_for_num_llh(
                        h_unsqueezed, y, h2[arange_b, t, :, :].unsqueeze(-1), mask, t, arange_b, h2=h2
                    ) for t in range(calc_range)])
            else:
                score = self.start_trans[y[:, 0]] + sum(
                    [self._calc_trans_score_for_num_llh(
                        h_unsqueezed, y, h2[0][arange_b, t, :, :].unsqueeze(-1), mask, t, arange_b, h2=h2
                    ) for t in range(seq_len//5)]) + sum(
                    [self._calc_trans_score_for_num_llh(
                        h_unsqueezed, y, h2[1][arange_b, t - seq_len//5, :, :].unsqueeze(-1), mask, t, arange_b, h2=h2
                    ) for t in range(seq_len//5, 2*(seq_len//5))]) + sum(
                    [self._calc_trans_score_for_num_llh(
                        h_unsqueezed, y, h2[2][arange_b, t - 2*(seq_len//5), :, :].unsqueeze(-1), mask, t, arange_b, h2=h2
                    ) for t in range(2*(seq_len//5), 3*(seq_len//5))]) + sum(
                    [self._calc_trans_score_for_num_llh(
                        h_unsqueezed, y, h2[1][arange_b, t - 3*(seq_len//5), :, :].unsqueeze(-1), mask, t, arange_b, h2=h2
                    ) for t in range(3*(seq_len//5), 4*(seq_len//5))]) + sum(
                    [self._calc_trans_score_for_num_llh(
                        h_unsqueezed, y, h2[2][arange_b, t - 4*(seq_len//5), :, :].unsqueeze(-1), mask, t, arange_b, h2=h2
                    ) for t in range(4*(seq_len//3), calc_range)])
        else:
            score = self.start_trans[y[:, 0]] + sum(
                [self._calc_trans_score_for_num_llh(
                    h_unsqueezed, y, (1-torch.sigmoid(self.theta)) * trans + torch.sigmoid(self.theta) * h2[arange_b, t, :, :].unsqueeze(-1).to(h_unsqueezed.device), mask, t, arange_b, h2=h2
                ) for t in range(calc_range)])
        # extract end label number of each sequence in mini batch
        # (batch_size)
        last_mask_index = mask.sum(1) - 1
        last_labels = y[arange_b, last_mask_index]
        each_last_score = h[arange_b, -1, last_labels] * mask[:, -1]

        # Add the score of the sequences of the maximum length in mini batch
        # Add the scores from the last tag of each sequence to EOS
        score += each_last_score + self.end_trans[last_labels]

        return score

    def _calc_trans_score_for_num_llh(
            self,
            h: FloatTensor,
            y: LongTensor,
            trans: FloatTensor,
            mask: BoolTensor,
            t: int,
            arange_b: FloatTensor, h2=None
    ) -> torch.Tensor:
        """
        calculate transition score for computing numberator llh
        :param h: hidden matrix (batch_size, seq_len, num_labels)
        :param y: answer labels of each sequence
                  in mini batch (batch_size, seq_len)
        :param trans: transition score
        :param mask: mask tensor of each sequence
                     in mini batch (batch_size, seq_len)
        :paramt t: index of hidden, transition, and mask matrixex
        :param arange_b: this param is seted torch.arange(batch_size)
        :param batch_size: batch size of this calculation
        """
        device = h.device
        mask_t = mask[:, t]
        mask_t = mask_t.to(device)
        mask_t1 = mask[:, t + 1]
        mask_t1 = mask_t1.to(device)

        # extract the score of t+1 label
        # (batch_size)
        h_t = h[arange_b, t, y[:, t]].squeeze(1)

        # extract the transition score from t-th label to t+1 label
        # (batch_size)
        if h2 is None:
            trans_t = trans[y[:, t], y[:, t + 1]].squeeze(1)
        else:
            trans_t = trans[arange_b, y[:, t], y[:, t + 1]].squeeze(1).to(device)

        # add the score of t+1 and the transition score
        # (batch_size)
        return h_t * mask_t + trans_t * mask_t1

    # def _set_trans(self, trans_matrix):  # !!!!!
    #     self.trans_matrix = trans_matrix

    def _initialize_parameters(self, pad_idx: Optional[int], pad_idx_val=None, const=None) -> None:
        """
        initialize transition parameters
        :param: pad_idx: if not None, additional initialize
        :return: None
        """
        if const is not None:
            nn.init.constant_(self.trans_matrix, 0.1)
            nn.init.constant_(self.start_trans, 0.1)
            nn.init.constant_(self.end_trans, 0.1)
            return

        nn.init.uniform_(self.trans_matrix, -0.1, 0.1)
        nn.init.uniform_(self.start_trans, -0.1, 0.1)
        nn.init.uniform_(self.end_trans, -0.1, 0.1)
        if pad_idx is not None:
            with torch.no_grad():  # I changes this!!!
                self.start_trans[pad_idx] = -10000.0 if pad_idx_val is None else pad_idx_val
                self.trans_matrix[pad_idx, :] = -10000.0 if pad_idx_val is None else pad_idx_val
                self.trans_matrix[:, pad_idx] = -10000.0 if pad_idx_val is None else pad_idx_val
                self.trans_matrix[pad_idx, pad_idx] = 0.0

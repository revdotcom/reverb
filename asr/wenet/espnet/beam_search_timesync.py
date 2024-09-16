"""
Time Synchronous One-Pass Beam Search.

Implements joint CTC/attention decoding where
hypotheses are expanded along the time (input) axis,
as described in https://arxiv.org/abs/2210.05200.
Supports CPU and GPU inference.
References: https://arxiv.org/abs/1408.2873 for CTC beam search
Author: Brian Yan

copied from https://github.com/espnet/espnet/blob/7c140c2ac9b4f642acb36131217dd984d4601681/espnet/nets/beam_search_timesync.py

Updated 2023 Jenny Drexler Fox 
- works with WeNet structures
- works with torchscript
- added token-level timestamps and confidence scores
"""

from typing import Any, Dict, List, Tuple, NamedTuple, Union, Optional

import torch

from wenet.utils.mask import subsequent_mask
from wenet.transformer.decoder import LanguageSpecificTransformerDecoder
from wenet.transformer.decoder import TransformerDecoder
import math

# Utility functions required for torchscript
def log_add(args: List[float]) -> float:
    """
    Stable log add
    """
    if all(a == -float('inf') for a in args):
        return -float('inf')
    a_max = max(args)
    lsp = math.log(sum(math.exp(a - a_max) for a in args))
    return a_max + lsp

def list_list_contains(a: List[List[int]], b:List[int]) -> bool:
    for t in a:
        if t == b:
            return True

    return False

def str_to_list(s: str) -> List[int]:
    elements = s[1:-1].split(',')
    return [int(x) for x in elements]

def default_val():
    return (float("-inf"), float("-inf"))

class Hypothesis(NamedTuple):
    """Hypothesis data type."""

    yseq: torch.Tensor
    score: Union[float, torch.Tensor] = 0
    scores: Dict[str, Union[float, torch.Tensor]] = dict()
    states: Dict[str, Any] = dict()
    # dec hidden state corresponding to yseq, used for searchable hidden ints
    hs: List[torch.Tensor] = []

    def asdict(self) -> dict:
        """Convert data to JSON-friendly dict."""
        return self._replace(
            yseq=self.yseq.tolist(),
            score=float(self.score),
            scores={k: float(v) for k, v in self.scores.items()},
        )._asdict()
    
class CacheItem:
    """For caching attentional decoder and LM states."""
    # NOTE: JDF - LM functionality currently removed b/c not compatible with torchscript
    
    def __init__(
            self,
            state: List[torch.Tensor],
            scores: torch.Tensor,
            log_sum: float,
    ):
        self.state = state
        self.scores = scores
        self.log_sum = log_sum


class BeamSearchTimeSync(object):
    """Time synchronous beam search algorithm."""

    def __init__(
        self,
        sos: int,
        beam_size: int,
        ctc_probs: torch.Tensor,
        decoder: LanguageSpecificTransformerDecoder, #TransformerDecoder, #
        weights: Dict[str, float],
        words: Dict[str, int] = dict(),
        word_prefixes: Dict[str, int] = dict(),
        tok_to_str: Dict[int, str] = dict(),
        pre_beam_ratio: float = 1.5,
        blank: int = 0,
        blank_threshold: float = 1.,
    ):
        """Initialize beam search.

        Args:
            beam_size: num hyps
            sos: sos index
            ctc: CTC module
            pre_beam_ratio: pre_beam_ratio * beam_size = pre_beam
                pre_beam is used to select candidates from vocab to extend hypotheses
            decoder: decoder ScorerInterface
            ctc_weight: ctc_weight
            blank: blank index
            blank_threshold: skip frames with blank probability > threshold

        """
        self.ctc_probs = ctc_probs
        self.decoder = decoder
        self.lm = None
        self.beam_size = beam_size
        self.pre_beam_size = int(pre_beam_ratio * beam_size)
        self.ctc_weight = weights["ctc"]
        self.decoder_weight:float = weights["decoder"]
        self.penalty = weights["length_bonus"]
        self.sos = sos
        self.sos_th = torch.tensor([self.sos]).unsqueeze(1)
        self.blank = blank
        self.attn_cache: Dict[str, CacheItem] = dict()  # cache for p_attn(Y|X)
        self.enc_output = torch.tensor([0.0])  # log p_ctc(Z|X)
        self.cat_embs = torch.tensor([0.0])  # log p_ctc(Z|X)
        self.running_size = self.beam_size
        self.encoder_mask = torch.ones(self.running_size, 1, 1)
        self.blank_threshold = math.log(blank_threshold)

        # JDF - for lexicon constraint
        self.words: Dict[str, int] = words
        self.word_prefixes: Dict[str, int] = word_prefixes
        self.tok_to_str: Dict[int, str] = tok_to_str
        self.word_start_char = '▁'

        # JDF - LM components removed b/c not torchscript compatible
        #self.lm_weight = weights["lm"]
        #self.lm_cache = dict()  # cache for p_lm(Y)
        #self.token_list = token_list

    def reset(self, enc_output: torch.Tensor, cat_embs: torch.Tensor):
        """Reset object for a new utterance."""
        self.enc_output = enc_output
        self.cat_embs = cat_embs
        
        self.sos_th = self.sos_th.to(enc_output.device)

        batch_size = enc_output.size(0)
        maxlen = enc_output.size(1)
        self.encoder_mask = torch.ones(1, 1, maxlen)

        hyps_mask = subsequent_mask(1).unsqueeze(0).repeat(1, 1, 1)  # (B*N, i, i)

        # initialize decoder with sos
        decoder_scores, decoder_state, att = self.decoder.forward_one_step_with_attn(
            self.enc_output, self.encoder_mask, self.sos_th, hyps_mask, None, cat_embs)

        # initialize cache with first token attention scores
        self.attn_cache = {str([self.sos]): CacheItem(
            state=decoder_state,
            scores=decoder_scores,
            log_sum=0.0,
        )}
        
        '''
        # JDF - LM initialization, LM needs to be torchscript-ized
        # initialize LM cache
        self.lm_cache = dict()
        if self.lm is not None:
            init_lm_state = self.lm.init_state(enc_output)
            lm_scores, lm_state = self.lm.score(self.sos_th, init_lm_state, enc_output)
            self.lm_cache[(self.sos,)] = CacheItem(
                state=lm_state,
                scores=lm_scores,
                log_sum=0.0,
            )
        '''
        
    def cached_score(self, h: List[int], cache: Dict[str, CacheItem]) -> float:
        """Retrieve decoder/LM scores which may be cached."""
        '''
        Basic algorithm:
        - h = hypothesis we want to get the attention score for = list of tokens
        - the score for h is the score for h[:-1] + the score for h[-1]
        - we've definitely already cached the score for h[:-1]
        - we may or may not have the score for h[-1] given h[:-1]
        - to get the score for h[-1], we run h[:-1] through the decoder, get the outputs, and then take outputs[h[-1]]
        - if we have already run h[:-1] through the decoder, then the outputs will be in cache[h[:-1]].scores
        Note: can't index into a dictionary (the cache) with a list of ints, so we turn every list of ints into a string for indexing
        '''
        root = h[:-1]  # prefix
        if str(root) in cache:
            # we've already run h[:-1] through the decoder
            root_scores = cache[str(root)].scores
            root_state = cache[str(root)].state
            root_log_sum = cache[str(root)].log_sum
        else:
            # we haven't already run h[:-1] through the decoder, so run decoder fwd one step and update cache
            # we know we have run h[:-2] through the decoder, so that's where we start
            root_root = root[:-1] # h[:-2]
            root_root_state: List[torch.Tensor] = cache[str(root_root)].state
            
            hyps_mask = subsequent_mask(len(root)).unsqueeze(0).repeat(1, 1, 1)  # (B*N, i, i)
            root_scores, root_state, att = self.decoder.forward_one_step_with_attn(
                self.enc_output, self.encoder_mask,
                torch.tensor(root, device=self.enc_output.device).unsqueeze(0),
                hyps_mask, root_root_state, self.cat_embs)
            root_log_sum = cache[str(root_root)].log_sum + float(
                cache[str(root_root)].scores[0, root[-1]]
            )
            cache[str(root)] = CacheItem(state=root_state, scores=root_scores, log_sum=root_log_sum)

        # get score for h[-1] and add it to score for h[:-1]
        cand_score = float(root_scores[0, h[-1]])
        score = root_log_sum + cand_score

        return score

    def joint_score(self, hyps: List[List[int]],
                    ctc_score_dp: Dict[str, Tuple[float, float]],
                    confs: Dict[str, List[Tuple[float, float]]]) -> Tuple[Dict[str, float], Dict[str, List[Tuple[float, float]]]]:
        """Calculate joint score for hyps and update confidences with attention scores."""
        scores: Dict[str, float] = dict()
        for h in hyps:
            # initialize with CTC score
            score = self.ctc_weight * log_add(list(ctc_score_dp[str(h)]))

            # add attention score and update confidences
            if len(h) > 1 and self.decoder_weight > 0 and self.decoder is not None:
                score += self.cached_score(h, self.attn_cache) * self.decoder_weight

                # leave first entry in confidence (ctc_conf) alone, update second entry in confidence
                confs[str(h)][-1] = (confs[str(h)][-1][0], float(self.attn_cache[str(h[:-1])].scores[0, h[-1]]))
                
            # add LM score
            '''
            if len(h) > 1 and self.lm is not None and self.lm_weight > 0:
                score += (
                    self.cached_score(h, self.lm_cache, self.lm) * self.lm_weight
                )  # lm score
            '''
            
            # length penalty
            score += self.penalty * (len(h) - 1)
            scores[str(h)] = score

            # JDF - best alternate (this idea is TBD)
            '''
            if len(h) > 1 and self.decoder_weight > 0 and self.decoder is not None:
                root = h[:-1]  # prefix
                if len(root) == 1 or root[-1] != self.sos:
                    root_scores = self.attn_cache[str(root)].scores
                    best_cand = int(torch.argmax(root_scores[0]))
                    new_hyp = root + [best_cand]
                    print(new_hyp)
                    if not True: #list_list_contains(hyps, new_hyp):
                        cand_score = float(root_scores[0, best_cand])
                        root_log_sum = self.attn_cache[str(root)].log_sum
                        alt_score = (root_log_sum + cand_score)*self.decoder_weight + self.penalty * (len(h) - 1)
                        
                        scores[str(new_hyp)] = alt_score
                        confs[str(new_hyp)] = confs[str(root)] + [(float("-inf"), float("-inf"))]
                        confs[str(new_hyp)][-1] = (confs[str(new_hyp)][-1][0], float(root_scores[0, best_cand]))
                        alternates.append(root + [best_cand])
            '''
        return scores, confs

    def time_step(self, t: int, p_ctc: torch.Tensor,
                  ctc_score_dp: Dict[str, Tuple[float, float]],
                  hyps: List[List[int]],
                  times: Dict[str, Tuple[List[int], List[int]]],
                  confs: Dict[str, List[Tuple[float, float]]],
                  scores: Dict[str, float],
    ) -> Tuple[Dict[str, Tuple[float, float]], List[List[int]], Dict[str, float], Dict[str, Tuple[List[int], List[int]]], Dict[str, List[Tuple[float, float]]]]:
        """Execute a single time step."""
        '''
        Basic algorithm:
        - execute CTC beam search
          - pull out top (beam x pre_beam) candidate tokens from ctc_outputs at this frame
          - loop over every hypothesis from last frame
            - loop over every candidate token
              - update/save overall scores for hyp + candidate
              - update timestamps for hyp + candidate
              - update ctc confidences for hyp + candidate
        - calculate joint scores for all candidates proposed by CTC beam search
        - sort candidates by joint score and prune beam
        '''

        # blank threshold - skip frame if blank score is higher than threshold
        best_cand = int(torch.argmax(p_ctc[0]))
        if best_cand == self.blank and p_ctc[best_cand] >= self.blank_threshold:
            return ctc_score_dp, hyps, scores, times, confs

        # get candidate tokens for this frame
        pre_beam_threshold = torch.sort(p_ctc)[0][-self.pre_beam_size]
        tmp_cands: List[List[int]] = (p_ctc >= pre_beam_threshold).nonzero().tolist()
        cands: List[int] = [z[0] for z in tmp_cands]
        if len(cands) == 0:
            cands = [int(torch.argmax(p_ctc[0]))]
            
        new_hyps: List[List[int]] = []
        ctc_score_dp_next:Dict[str, Tuple[float, float]] = dict()
        # ctc beam search loop
        for hyp_l in hyps:
            '''
            # JDF - code related to idea of adding best candidate from attention decoder, not fully implemented yet
            if not str(hyp_l) in ctc_score_dp:
                # created by attention decoder
                ctc_score_dp[str(hyp_l)] = self.default_val()
                times[str(hyp_l)] = (times[str(hyp_l[:-1])][0] + [t-1], times[str(hyp_l[:-1])][1] + [t])
            '''
            # add up blank and non-blank probs for this hypothesis
            p_prev_l = log_add(list(ctc_score_dp[str(hyp_l)]))
            for c in cands:
                if c == self.blank:
                    # Note: timestamps and confidences don't need to be updated if we're just adding a blank token
                    if str(hyp_l) not in ctc_score_dp_next:
                        ctc_score_dp_next[str(hyp_l)] = default_val()

                    # update p_b (blank probability) for this hypothesis
                    p_nb, p_b = ctc_score_dp_next[str(hyp_l)]
                    p_b = log_add([p_b, float(p_ctc[c]) + p_prev_l])
                    ctc_score_dp_next[str(hyp_l)] = (p_nb, p_b)
                    
                    if not list_list_contains(new_hyps, hyp_l):
                        new_hyps.append(hyp_l)
                else:
                    # new hypothesis = hypothesis from last frame + new candidate token
                    l_plus = hyp_l + [int(c)]
                    
                    if str(l_plus) not in ctc_score_dp_next:
                        ctc_score_dp_next[str(l_plus)] = default_val()                        
                    p_nb, p_b = ctc_score_dp_next[str(l_plus)]
                        
                    # update time-stamps
                    # - set both start and end times if we haven't seen this hypothesis before
                    # - update end time if we have seen this hypothesis before
                    if str(l_plus) not in times:
                        times[str(l_plus)] = (times[str(hyp_l)][0] + [t], times[str(hyp_l)][1] + [t+1])
                    else:
                        times[str(l_plus)][1][-1] = t+1

                    # update confidences
                    # - initialize if we haven't seen this hypothesis before
                    # - update ctc confidence
                    if str(l_plus) not in confs:
                        confs[str(l_plus)] = confs[str(hyp_l)] + [(float("-inf"), float("-inf"))]
                    confs[str(l_plus)][-1] = (max([float(confs[str(l_plus)][-1][0]), float(p_ctc[c])]), confs[str(l_plus)][-1][1])

                    # update likelihoods
                    if c == hyp_l[-1]:
                        # repeated token
                        # 1) update p_nb for l_plus based on p_b of hyp_l (repeated token with blank in between)
                        if str(hyp_l) not in ctc_score_dp:
                            ctc_score_dp[str(hyp_l)] = default_val()
                        p_nb_prev, p_b_prev = ctc_score_dp[str(hyp_l)]
                        p_nb = log_add([p_nb, float(p_ctc[c]) + p_b_prev])

                        # 2) keep hyp_l as a hypothesis for this timestep, update likelihoods, timestamps, and confidences
                        if str(hyp_l) not in ctc_score_dp_next:
                            ctc_score_dp_next[str(hyp_l)] = default_val()
                        p_nb_l, p_b_l = ctc_score_dp_next[str(hyp_l)]
                        p_nb_l = log_add([p_nb_l, float(p_ctc[c]) + p_nb_prev])
                        ctc_score_dp_next[str(hyp_l)] = (p_nb_l, p_b_l)
                        times[str(hyp_l)][1][-1] = t+1
                        confs[str(hyp_l)][-1] = (max([float(confs[str(hyp_l)][-1][0]), float(p_ctc[c])]), confs[str(hyp_l)][-1][1])
                    else:
                        # easy update if not a repeated token
                        p_nb = log_add([p_nb, float(p_ctc[c]) + p_prev_l])

                    # if l_plus was considered before but didn't make it into the top hypotheses, add in the scores we calculated last frame
                    if not list_list_contains(hyps, l_plus) and str(l_plus) in ctc_score_dp:
                        p_b = log_add([p_b, float(p_ctc[self.blank]) + log_add(list(ctc_score_dp[str(l_plus)]))])
                        p_nb = log_add([p_nb, float(p_ctc[c]) + ctc_score_dp[str(l_plus)][0]])
                        
                    ctc_score_dp_next[str(l_plus)] = (p_nb, p_b)
                    if not list_list_contains(new_hyps, l_plus):
                        new_hyps.append(l_plus)

        # lexicon constraint
        good_hyps: List[List[int]] = []
        if len(self.words.keys()) > 0:
            for hyp in new_hyps:
                if len(hyp) == 1:
                    # just <sos> token - keep and continue
                    good_hyps.append(hyp)
                    continue

                if self.tok_to_str[hyp[-1]].startswith(self.word_start_char):
                    # assuming any single token that starts with _ is a valid word prefix
                    # we're going to instead test the previous word to make sure it's a complete word
                    if len(hyp) == 2:
                        # if previous word is just the intial <sos> token
                        good_hyps.append(hyp)
                        continue
                    start = 2
                else:
                    start = 1

                last_word = ''
                for i in range(start, len(hyp)):
                    subword = self.tok_to_str[hyp[-i]]
                    last_word = subword + last_word
                    if subword.startswith(self.word_start_char):
                        break

                good = False
                if self.tok_to_str[hyp[-1]].startswith(self.word_start_char):
                    # we've started a new word on this time-step so we're testing whether the previous word is valid
                    # accept any word if it ends in a dash - good idea??
                    if self.tok_to_str[hyp[-2]].endswith('-') or last_word in self.words:
                        good = True
                else:
                    # we're still in the middle of a word - just make sure it's a valid prefix
                    if last_word in self.word_prefixes:
                        good = True
                if good:
                    good_hyps.append(hyp)

        # add attention scores to each hypothesis
        if len(good_hyps) > 0:
            scores, confs = self.joint_score(good_hyps, ctc_score_dp_next, confs)
        else:
            # either we skipped the lexicon constraint or, in very rare cases,
            # the lexicon constraint eliminated every hypothesis
            # we could probably do a better fix later, but for now if no hypothesis is valid
            # we're just going to let the system output an invalid word
            scores, confs = self.joint_score(new_hyps, ctc_score_dp_next, confs)

        # complicated sorting stuff b/c of torchscript issues
        reverse_dict: Dict[float, str] = dict()
        for k, v in scores.items():
            # assuming two hyps won't have the exact same score - bad assumption?
            reverse_dict[v] = k
        sorted_scores = sorted(reverse_dict.keys())
        sorted_scores.reverse()

        # prune down to desired beam size
        sorted_scores = sorted_scores[:self.beam_size]
        hyps = [str_to_list(reverse_dict[p]) for p in sorted_scores]

        # save scores for next time step
        ctc_score_dp = ctc_score_dp_next.copy()
        
        return ctc_score_dp, hyps, scores, times, confs

    def __call__(
            self, x: torch.Tensor, cat_embs: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """Perform beam search.

        Args:
            enc_output (torch.Tensor)

        Return:
            n_best hypotheses
            n_best scores
            n_best start times
            n_best end times
            n_best confidences times
        """
        lpz = self.ctc_probs
        if len(lpz.shape) > 2:
            lpz = lpz.squeeze(0)
        self.reset(x, cat_embs)

        # hyps = list of hyps, where each hyp is a list of integers
        hyps = [[self.sos]]
        # scores = dictionary of overall scores per hypothesis
        scores: Dict[str, float] = dict()
        # times = dictionary of (start_times, end_times) for each hypothesis, where times are lists of integer frames for each token
        times: Dict[str, Tuple[List[int], List[int]]] = dict()
        times[str([self.sos])] = ([0], [0])
        # confs = dictionary of confidences for each hypothesis, where confidences is a list of (ctc_conf, att_conf) tuples for each token
        confs: Dict[str, List[Tuple[float, float]]] = dict()
        confs[str([self.sos])] = [(float("-inf"), float("-inf"))]
        # ctc_score_dp = dictionary of ctc scores for each hypothesis, where scores are (p_non_blank, p_blank) tuples
        ctc_score_dp: Dict[str, Tuple[float, float]] = dict() # (p_nb, p_b) - dp object tracking p_ctc
        ctc_score_dp[str([self.sos])] = (float("-inf"), 0.0)

        # loop over time_steps/frames
        for t in range(lpz.shape[0]):
            ctc_score_dp, hyps, scores, times, confs = self.time_step(t, lpz[t, :], ctc_score_dp, hyps, times, confs, scores)
            
        confs_type = "max"
        if confs_type == "att":
            # attention scores only
            n_best_confs: List[torch.Tensor] = [torch.tensor([c[1] for c in confs[str(h)]]) for h in hyps]
        elif confs_type == "avg":
            # average ctc and attention scores
            n_best_confs: List[torch.Tensor] = [torch.tensor([(math.exp(c[0])+math.exp(c[1]))/2. for c in confs[str(h)]]) for h in hyps]
            n_best_confs = [torch.log(c) for c in n_best_confs]
        elif confs_type == "weighted":
            # weighted average of ctc and attention scores as in decoding
            n_best_confs: List[torch.Tensor] = [torch.tensor([self.ctc_weight*c[0]+self.decoder_weight*c[1] for c in confs[str(h)]]) for h in hyps]
        elif confs_type == "max":
            # max of ctc and attention scores per token
            n_best_confs: List[torch.Tensor] = [torch.tensor([max(c[0], c[1]) for c in confs[str(h)]]) for h in hyps]
        else:
            # ctc scores only
            n_best_confs: List[torch.Tensor] = [torch.tensor([c[0] for c in confs[str(h)]]) for h in hyps]

        return [torch.tensor(h) for h in hyps], [torch.tensor([scores[str(h)]]) for h in hyps], [torch.tensor([times[str(h)][0]]) for h in hyps], [torch.tensor([times[str(h)][1]]) for h in hyps], n_best_confs

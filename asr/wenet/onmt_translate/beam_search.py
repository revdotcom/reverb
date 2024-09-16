import torch
from wenet.onmt_translate import penalties
import warnings

from typing import List, Optional, Tuple, Dict, Any
from operator import itemgetter

def tile(x, count: int, dim: int =0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm)
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    if len(out_size) == 1:
        x = (
            x.contiguous()
            .view(batch, -1)
            .transpose(0, 1)
            .repeat(count, 1)
            .transpose(0, 1)
            .contiguous()
            .view(out_size[0])
        )
    elif len(out_size) == 2:
        x = (
            x.contiguous()
            .view(batch, -1)
            .transpose(0, 1)
            .repeat(count, 1)
            .transpose(0, 1)
            .contiguous()
            .view(out_size[0], out_size[1])
        )
    elif len(out_size) == 3:
        x = (
            x.contiguous()
            .view(batch, -1)
            .transpose(0, 1)
            .repeat(count, 1)
            .transpose(0, 1)
            .contiguous()
            .view(out_size[0], out_size[1], out_size[2])
        )
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x

class GNMTGlobalScorer(object):
    """NMT re-ranking.

    Args:
       alpha (float): Length parameter.
       beta (float):  Coverage parameter.
       length_penalty (str): Length penalty strategy.
       coverage_penalty (str): Coverage penalty strategy.

    Attributes:
        alpha (float): See above.
        beta (float): See above.
        length_penalty (callable): See :class:`penalties.PenaltyBuilder`.
        coverage_penalty (callable): See :class:`penalties.PenaltyBuilder`.
        has_cov_pen (bool): See :class:`penalties.PenaltyBuilder`.
        has_len_pen (bool): See :class:`penalties.PenaltyBuilder`.
    """

    #@classmethod
    #def from_opt(cls, opt):
    #    return cls(opt.alpha, opt.beta, opt.length_penalty, opt.coverage_penalty)

    def __init__(self, alpha: float, beta: float, length_penalty: str, coverage_penalty: str):
        self._validate(alpha, beta, length_penalty, coverage_penalty)
        self.alpha = alpha
        self.beta = beta
        self.penalty_builder = penalties.PenaltyBuilder(coverage_penalty, length_penalty)
        self.has_cov_pen = self.penalty_builder.has_cov_pen
        # Term will be subtracted from probability
        self.cov_penalty = coverage_penalty

        self.has_len_pen = self.penalty_builder.has_len_pen
        # Probability will be divided by this
        self.length_penalty = length_penalty

    @classmethod
    def _validate(cls, alpha: float, beta: float, length_penalty: str, coverage_penalty: str):
        # these warnings indicate that either the alpha/beta
        # forces a penalty to be a no-op, or a penalty is a no-op but
        # the alpha/beta would suggest otherwise.
        if length_penalty is not None and alpha == 0.0:
            warnings.warn(
                "Using length penalty with alpha==0 "
                "is equivalent to using length penalty none."
            )
        if coverage_penalty is None or coverage_penalty == "none":
            if beta != 0:
                warnings.warn(
                    "Non-default `beta` with no coverage penalty. "
                    "`beta` has no effect."
                )
        else:
            # using some coverage penalty
            if beta == 0.0:
                warnings.warn(
                    "Non-default coverage penalty with beta==0 "
                    "is equivalent to using coverage penalty none."
                )

class BeamSearch(object):
    """Base class for generation strategies.

    Args:
      pad (int): Magic integer in output vocab.
      bos (int): Magic integer in output vocab.
      eos (int): Magic integer in output vocab.
      unk (int): Magic integer in output vocab.
      start (int): Magic integer in output vocab.
      batch_size (int): Current batch size.
      parallel_paths (int): Decoding strategies like beam search
        use parallel paths. Each batch is repeated ``parallel_paths``
        times in relevant state tensors.
      min_length (int): Shortest acceptable generation, not counting
        begin-of-sentence or end-of-sentence.
      max_length (int): Longest acceptable sequence, not counting
        begin-of-sentence (presumably there has been no EOS
        yet if max_length is used as a cutoff).
      ban_unk_token (Boolean): Whether unk token is forbidden
      block_ngram_repeat (int): Block beams where
        ``block_ngram_repeat``-grams repeat.
      exclusion_tokens (set[int]): If a gram contains any of these
        tokens, it may repeat.
      return_attention (bool): Whether to work with attention too. If this
        is true, it is assumed that the decoder is attentional.

    Attributes:
      pad (int): See above.
      bos (int): See above.
      eos (int): See above.
      unk (int): See above.
      start (int): See above.
      predictions (list[list[LongTensor]]): For each batch, holds a
        list of beam prediction sequences.
        scores (list[list[FloatTensor]]): For each batch, holds a
        list of scores.
      attention (list[list[FloatTensor or list[]]]): For each
        batch, holds a list of attention sequence tensors
        (or empty lists) having shape ``(step, inp_seq_len)`` where
        ``inp_seq_len`` is the length of the sample (not the max
        length of all inp seqs).
      alive_seq (LongTensor): Shape ``(B x parallel_paths, step)``.
        This sequence grows in the ``step`` axis on each call to
        :func:``advance()``.
        is_finished (ByteTensor or NoneType): Shape ``(B, parallel_paths)``.
        Initialized to ``None``.
      alive_attn (FloatTensor or NoneType): If tensor, shape is
        ``(step, B x parallel_paths, inp_seq_len)``, where ``inp_seq_len``
        is the (max) length of the input sequence.
      target_prefix (LongTensor or NoneType): If tensor, shape is
        ``(B x parallel_paths, prefix_seq_len)``, where ``prefix_seq_len``
        is the (max) length of the pre-fixed prediction.
      min_length (int): See above.
      max_length (int): See above.
      ban_unk_token (Boolean): See above.
      block_ngram_repeat (int): See above.
      exclusion_tokens (set[int]): See above.
      return_attention (bool): See above.
      done (bool): See above."""

    """Generation beam search.

    Note that the attributes list is not exhaustive. Rather, it highlights
    tensors to document their shape. (Since the state variables' "batch"
    size decreases as beams finish, we denote this axis with a B rather than
    ``batch_size``).

    Args:
        beam_size (int): Number of beams to use (see base ``parallel_paths``).
        batch_size (int): See base.
        pad (int): See base.
        bos (int): See base.
        eos (int): See base.
        unk (int): See base.
        start (int): See base.
        n_best (int): Don't stop until at least this many beams have
            reached EOS.
        global_scorer (onmt.translate.GNMTGlobalScorer): Scorer instance.
        min_length (int): See base.
        max_length (int): See base.
        return_attention (bool): See base.
        block_ngram_repeat (int): See base.
        exclusion_tokens (set[int]): See base.

    Attributes:
        top_beam_finished (ByteTensor): Shape ``(B,)``.
        _batch_offset (LongTensor): Shape ``(B,)``.
        _beam_offset (LongTensor): Shape ``(batch_size x beam_size,)``.
        alive_seq (LongTensor): See base.
        topk_log_probs (FloatTensor): Shape ``(B, beam_size,)``. These
            are the scores used for the topk operation.
        src_len (LongTensor): Lengths of encodings. Used for
            masking attentions.
        select_indices (LongTensor or NoneType): Shape
            ``(B x beam_size,)``. This is just a flat view of the
            ``_batch_index``.
        topk_scores (FloatTensor): Shape
            ``(B, beam_size)``. These are the
            scores a sequence will receive if it finishes.
        topk_ids (LongTensor): Shape ``(B, beam_size)``. These are the
            word indices of the topk predictions.
        _batch_index (LongTensor): Shape ``(B, beam_size)``.
        _prev_penalty (FloatTensor or NoneType): Shape
            ``(B, beam_size)``. Initialized to ``None``.
        _coverage (FloatTensor or NoneType): Shape
            ``(1, B x beam_size, inp_seq_len)``.
        hypotheses (list[list[Tuple[Tensor]]]): Contains a tuple
            of score (float), sequence (long), and attention (float or None).
    """

    def __init__(
        self,
        beam_size: int,
        batch_size: int,
        pad: int,
        bos: int,
        eos: int,
        unk: int,
        start: int,
        n_best: int,
        global_scorer: GNMTGlobalScorer,
        min_length: int,
        max_length: int,
        return_attention: bool,
        block_ngram_repeat: int,
        stepwise_penalty: bool,
        ratio: float,
        ban_unk_token: bool,
    ):
        # from DecodeStrategy
        
        # magic indices
        self.pad = pad
        self.bos = bos
        self.eos = eos
        self.unk = unk
        self.start = start

        self.batch_size = batch_size
        self.parallel_paths = beam_size
        self.global_scorer = global_scorer

        # result caching
        self.predictions = [[] for _ in range(batch_size)]
        self.scores = [[] for _ in range(batch_size)]
        self.attention = [[] for _ in range(batch_size)]
        self.hypotheses: List[List[Tuple[Tensor, Tensor, Tensor]]] = [[(torch.tensor([0]), torch.tensor([0]), torch.tensor([0]))] for _ in range(batch_size)]
        for i in range(len(self.hypotheses)):
            for j in range(len(self.hypotheses[i])):
                del self.hypotheses[i][j]
        self.alive_attn = torch.tensor([0.0])

        self.min_length = min_length
        self.max_length = max_length
        self.ban_unk_token = ban_unk_token

        self.block_ngram_repeat = block_ngram_repeat
        n_paths = batch_size * beam_size
        self.forbidden_tokens = [dict() for _ in range(n_paths)]

        self.exclusion_tokens = None #exclusion_tokens
        self.return_attention = return_attention

        self.done = False

        # from BeamSearchBase
        
        # beam parameters
        self.beam_size = beam_size
        self.n_best = n_best
        self.ratio = ratio

        # beam state
        self.top_beam_finished = torch.zeros([batch_size], dtype=torch.uint8)
        # BoolTensor was introduced in pytorch 1.2
        #try:
        #self.top_beam_finished = self.top_beam_finished.bool()
        #except AttributeError:
        #    pass
        self._batch_offset = torch.arange(batch_size, dtype=torch.long)

        self.select_indices = torch.tensor([0.0])
        self.done = False
        # "global state" of the old beam
        self._prev_penalty = torch.tensor([0.0])
        self._coverage = torch.tensor([0.0])

        self._stepwise_cov_pen = stepwise_penalty and self.global_scorer.has_cov_pen
        self._vanilla_cov_pen = not stepwise_penalty and self.global_scorer.has_cov_pen
        self._cov_pen = self.global_scorer.has_cov_pen

        self.src_len = torch.tensor([-1])
        self.alive_seq = torch.full(
            [self.batch_size * self.parallel_paths, 1],
            self.start,
            dtype=torch.long,
        )
        self.target_prefix = None #torch.tensor([0]) : Optional[torch.Tensor]
        self.is_finished = torch.zeros(
            [self.batch_size, self.parallel_paths], dtype=torch.uint8
        )

        self.best_scores = torch.full(
            [self.batch_size], -1e10, dtype=torch.float
        )
        self._beam_offset = torch.arange(
            0,
            self.batch_size * self.beam_size,
            step=self.beam_size,
            dtype=torch.long,
        )

        self.topk_log_probs = (
            torch.tensor([0.0] + [float("-inf")] * (self.beam_size - 1))
            .repeat(self.batch_size)
            .reshape(self.batch_size, self.beam_size)
        )
        self.topk_scores = torch.empty(
            (self.batch_size, self.beam_size), dtype=torch.float
        )
        self.topk_ids = torch.empty(
            (self.batch_size, self.beam_size), dtype=torch.long
        )
        self._batch_index = torch.empty(
            [self.batch_size, self.beam_size], dtype=torch.long
        )

    def get_device_from_enc_out(self, enc_out):
        if isinstance(enc_out, tuple):
            mb_device = enc_out[0].device
        else:
            mb_device = enc_out.device
        return mb_device

    def initialize_tile(self, enc_out, src_len):
        #def fn_map_state(state, dim):
        #    return tile(state, self.beam_size, dim=dim)

        if isinstance(enc_out, tuple):
            enc_out = tuple(tile(x, self.beam_size, dim=0) for x in enc_out)
        elif enc_out is not None:
            enc_out = tile(enc_out, self.beam_size, dim=0)

        self.src_len = tile(src_len, self.beam_size)

        return None, enc_out

    def initialize_ds(
        self, enc_out, src_len, device: Optional[torch.device]=None
    ):
        """DecodeStrategy subclasses should override :func:`initialize()`.

        `initialize` should be called before all actions.
        used to prepare necessary ingredients for decode."""

        if device is None:
            device = "cpu" #torch.device("cpu")
        # Here we set the decoder to start with self.start (BOS or EOS)
        self.alive_seq = torch.full(
            [self.batch_size * self.parallel_paths, 1],
            self.start,
            dtype=torch.long,
            device=device,
        )
        self.is_finished = torch.zeros(
            [self.batch_size, self.parallel_paths], dtype=torch.uint8, device=device
        )
        return None, enc_out, src_len

    def __len__(self):
        return self.alive_seq.shape[1]

    def ensure_min_length(self, log_probs):
        if len(self) <= self.min_length:
            log_probs[:, self.eos] = -1e20

    def ensure_unk_removed(self, log_probs):
        if self.ban_unk_token:
            log_probs[:, self.unk] = -1e20

    def ensure_max_length(self):
        # add one to account for BOS. Don't account for EOS because hitting
        # this implies it hasn't been found.
        if len(self) == self.max_length + 1:
            self.is_finished.fill_(1)
    
    def block_ngram_repeats(self, log_probs):
        """We prevent the beam from going in any direction that would repeat
        any ngram of size <block_ngram_repeat> more thant once.

        The way we do it: we maintain a list of all ngrams of size
        <block_ngram_repeat> that is updated each time the beam advances, and
        manually put any token that would lead to a repeated ngram to 0.

        This improves on the previous version's complexity:
        - previous version's complexity: batch_size * beam_size * len(self)
        - current version's complexity: batch_size * beam_size

        This improves on the previous version's accuracy;
        - Previous version blocks the whole beam, whereas here we only
        block specific tokens.
        - Before the translation would fail when all beams contained
        repeated ngrams. This is sure to never happen here."""

        # we don't block nothing if the user doesn't want it
        if self.block_ngram_repeat <= 0:
            return

        # we can't block nothing beam's too short
        if len(self) < self.block_ngram_repeat:
            return

        return
    '''
        n = self.block_ngram_repeat - 1
        for path_idx in range(self.alive_seq.shape[0]):
            # we check paths one by one
            current_ngram = tuple([0]*n)
            for i in range(n):
                current_ngram[i] = int(self.alive_seq[path_idx, -n+i])
            forbidden_tokens = self.forbidden_tokens[path_idx].get(current_ngram, None)
            if forbidden_tokens is not None:
                log_probs[path_idx, list(forbidden_tokens)] = -10e20
    '''
    
    def maybe_update_forbidden_tokens(self):
        """We complete and reorder the list of forbidden_tokens"""

        # we don't forbid nothing if the user doesn't want it
        if self.block_ngram_repeat <= 0:
            return

        # we can't forbid nothing if beam's too short
        if len(self) < self.block_ngram_repeat:
            return

        return
    '''
        n = self.block_ngram_repeat

        forbidden_tokens = list()
        for path_idx, seq in zip(self.select_indices, self.alive_seq):
            # Reordering forbidden_tokens following beam selection
            # We rebuild a dict to ensure we get the value and not the pointer
            forbidden_tokens.append(deepcopy(self.forbidden_tokens[path_idx]))

            # Grabing the newly selected tokens and associated ngram
            current_ngram = tuple(seq[-n:].tolist())

            # skip the blocking if any token in current_ngram is excluded
            if set(current_ngram) & self.exclusion_tokens:
                continue

            forbidden_tokens[-1].setdefault(current_ngram[:-1], set())
            forbidden_tokens[-1][current_ngram[:-1]].add(current_ngram[-1])

        self.forbidden_tokens = forbidden_tokens
    '''
    
    def target_prefixing(self, log_probs):
        """Fix the first part of predictions with `self.target_prefix`.

        Args:
        log_probs (FloatTensor): logits of size ``(B, vocab_size)``.

        Returns:
        log_probs (FloatTensor): modified logits in ``(B, vocab_size)``.
        """
        _B, vocab_size = log_probs.size()
        step = len(self)
        if self.target_prefix is not None:
            if step <= self.target_prefix.size(1):
                pick_idx: List[int] = self.target_prefix[:, step - 1].tolist()  # (B)
                '''
                pick_coo = [
                [path_i, pick]
                for path_i, pick in enumerate(pick_idx)
                if pick not in [self.eos, self.pad]
                ]
                mask_pathid = [
                path_i
                for path_i, pick in enumerate(pick_idx)
                if pick in [self.eos, self.pad]
                ]
                '''
                pick_coo: List[List[int]] = []
                mask_pathid: List[int] = []
                pick = 0
                path_i = 0
                for path_i, pick in enumerate(pick_idx):
                    if pick in [self.eos, self.pad]:
                        mask_pathid.append(path_i)
                    else:
                        pick_coo.append([path_i, pick])
                if len(pick_coo) > 0:
                    pick_coo_t = torch.tensor(pick_coo).to(self.target_prefix)
                    pick_fill_value = torch.ones([pick_coo_t.size(0)], dtype=log_probs.dtype)
                    # pickups: Tensor where specified index were set to 1, others 0
                    pickups = torch.sparse_coo_tensor(
                        pick_coo_t.t(),
                        pick_fill_value,
                        size=log_probs.size(),
                        device=log_probs.device,
                    ).to_dense()
                    # dropdowns: opposite of pickups, 1 for those shouldn't pick
                    dropdowns = torch.ones_like(pickups) - pickups
                    if len(mask_pathid) > 0:
                        path_mask = torch.zeros(_B).to(self.target_prefix)
                        path_mask[mask_pathid] = 1
                        path_mask = path_mask.unsqueeze(1)#.to(dtype=bool)
                        dropdowns = dropdowns.masked_fill(path_mask, 0)
                    # Minus dropdowns to log_probs making probabilities of
                    # unspecified index close to 0
                    log_probs -= 10000 * dropdowns
        return log_probs

    def maybe_update_target_prefix(self, select_index):
        """We update / reorder `target_prefix` for alive path."""
        if self.target_prefix is None:
            return
        # prediction step have surpass length of given target_prefix,
        # no need to further change this attr
        if len(self) > self.target_prefix.size(1):
            return
        self.target_prefix = self.target_prefix.index_select(0, select_index)

    def initialize_(self, enc_out, src_len, device:Optional[torch.device]):
        self.initialize_ds(
            enc_out, src_len, device)

        self.best_scores = torch.full(
            [self.batch_size], -1e10, dtype=torch.float, device=device
        )
        self._beam_offset = torch.arange(
            0,
            self.batch_size * self.beam_size,
            step=self.beam_size,
            dtype=torch.long,
            device=device,
        )
        self.topk_log_probs = (
            torch.tensor([0.0] + [float("-inf")] * (self.beam_size - 1), device=device)
            .repeat(self.batch_size)
            .reshape(self.batch_size, self.beam_size)
        )
        # buffers for the topk scores and 'backpointer'
        self.topk_scores = torch.empty(
            (self.batch_size, self.beam_size), dtype=torch.float, device=device
        )
        self.topk_ids = torch.empty(
            (self.batch_size, self.beam_size), dtype=torch.long, device=device
        )
        self._batch_index = torch.empty(
            [self.batch_size, self.beam_size], dtype=torch.long, device=device
        )
    
    #@property
    def current_predictions(self):
        return self.alive_seq[:, -1]

    #@property
    def current_backptr(self):
        # for testing
        return self.select_indices.view(self.batch_size, self.beam_size).fmod(
            self.beam_size
        )

    #@property
    def batch_offset(self):
        return self._batch_offset

    def _pick(self, log_probs): #, out=None):
        """Take a token pick decision for a step.

        Args:
            log_probs (FloatTensor): (B * beam_size, vocab_size)
            out (Tensor, LongTensor): output buffers to reuse, optional.

        Returns:
            topk_scores (FloatTensor): (B, beam_size)
            topk_ids (LongTensor): (B, beam_size)
        """
        vocab_size = log_probs.size(-1)
        # maybe fix some prediction at this step by modifying log_probs
        log_probs = self.target_prefixing(log_probs)

        # Flatten probs into a list of possibilities.
        curr_scores = log_probs.reshape(-1, self.beam_size * vocab_size)
        topk_scores, topk_ids = torch.topk(curr_scores, self.beam_size, dim=-1)
        return topk_scores, topk_ids

    def update_finished(self):
        # Penalize beams that finished.
        _B_old = self.topk_log_probs.shape[0]
        step = self.alive_seq.shape[-1]  # 1 greater than the step in advance
        self.topk_log_probs.masked_fill_(self.is_finished, -1e10)
        # on real data (newstest2017) with the pretrained transformer,
        # it's faster to not move this back to the original device
        self.is_finished = self.is_finished.to("cpu")
        self.top_beam_finished |= self.is_finished[:, 0].eq(1)
        predictions = self.alive_seq.view(_B_old, self.beam_size, step)
        print(step)
        print(self.alive_attn.shape)
        attention = (
            self.alive_attn.view(
                _B_old, self.beam_size, step - 1, self.alive_attn.size(-1)
            )
            if self.alive_attn is not None
            else None
        )
        non_finished_batch: List[int] = []
        for i in range(self.is_finished.size(0)):  # Batch level
            b = self._batch_offset[i]
            #finished_hyp = self.is_finished[i].nonzero(as_tuple=False).view(-1)
            finished_hyp = self.is_finished[i].nonzero().view(-1)
            # Store finished hypotheses for this batch.
            for j in finished_hyp:  # Beam level: finished beam j in batch i
                if self.ratio > 0:
                    s = self.topk_scores[i, j] / (step + 1)
                    if self.best_scores[b] < s:
                        self.best_scores[b] = s
                self.hypotheses[b].append(
                    (
                        self.topk_scores[i, j],
                        predictions[i, j, 1:],  # Ignore start_token.
                        attention[i, j, :, : self.src_len[i]]
                        if attention is not None
                        else None,
                    )
                )
            # End condition is the top beam finished and we can return
            # n_best hypotheses.
            if self.ratio > 0:
                pred_len = self.src_len[i] * self.ratio
                finish_flag = (
                    (self.topk_scores[i, 0] / pred_len) <= self.best_scores[b]
                ) or self.is_finished[i].all()
            else:
                finish_flag = int(self.top_beam_finished[i]) != 0
            if finish_flag and len(self.hypotheses[b]) >= self.beam_size:
                hyp_scores = torch.tensor([float(h[0].item()) for h in self.hypotheses[b]])
                #best_hyp = sorted(self.hypotheses[b], key=itemgetter(0), reverse=True)[
                #    : self.n_best
                #]
                best_scores, best_scores_idx = torch.topk(hyp_scores, self.n_best)
                for j in best_scores_idx:
                    #for n, (score, pred, attn) in enumerate(best_hyp):
                    (score, pred, attn) = self.hypotheses[b][j]
                    self.scores[b].append(score)
                    self.predictions[b].append(pred)  # ``(batch, n_best,)``
                    self.attention[b].append(attn if attn is not None else [])
            else:
                non_finished_batch.append(i)

        non_finished = torch.tensor(non_finished_batch)
        # If all sentences are translated, no need to go further.
        if len(non_finished) == 0:
            self.done = True
            return

        _B_new = non_finished.shape[0]
        self.remove_finished_batches(
            _B_new, _B_old, non_finished, predictions, attention, step
        )

    def remove_finished_batches(
        self, _B_new:int, _B_old:int, non_finished, predictions, attention, step:int
    ):
        # Remove finished batches for the next step.
        self.top_beam_finished = self.top_beam_finished.index_select(0, non_finished)
        self._batch_offset = self._batch_offset.index_select(0, non_finished)
        non_finished = non_finished.to(self.topk_ids.device)
        self.topk_log_probs = self.topk_log_probs.index_select(0, non_finished)
        self._batch_index = self._batch_index.index_select(0, non_finished)
        self.select_indices = self._batch_index.view(_B_new * self.beam_size)
        self.alive_seq = predictions.index_select(0, non_finished).view(
            -1, self.alive_seq.size(-1)
        )
        self.topk_scores = self.topk_scores.index_select(0, non_finished)
        self.topk_ids = self.topk_ids.index_select(0, non_finished)
        self.maybe_update_target_prefix(self.select_indices)
        if self.alive_attn is not None:
            inp_seq_len = self.alive_attn.size(-1)
            self.alive_attn = attention.index_select(0, non_finished).view(
                _B_new * self.beam_size, step - 1, inp_seq_len
            )
            if self._cov_pen:
                self._coverage = (
                    self._coverage.view(_B_old, self.beam_size, 1, inp_seq_len)
                    .index_select(0, non_finished)
                    .view(_B_new * self.beam_size, 1, inp_seq_len)
                )
                if self._stepwise_cov_pen:
                    self._prev_penalty = self._prev_penalty.index_select(
                        0, non_finished
                    )

    def advance(self, log_probs, attn):
        vocab_size = log_probs.size(-1)

        # using integer division to get an integer _B without casting
        _B = log_probs.shape[0] // self.beam_size

        if self._stepwise_cov_pen and self._prev_penalty is not None:
            self.topk_log_probs += self._prev_penalty
            #self.topk_log_probs -= self.global_scorer.cov_penalty(
            #    self._coverage + attn, self.global_scorer.beta
            #).view(_B, self.beam_size)
            if self.global_scorer.cov_penalty == "wu":
                self.topk_log_probs -= self.global_scorer.penalty_builder.coverage_wu(
                    self._coverage + attn, self.global_scorer.beta
                ).view(_B, self.beam_size)
            elif self.global_scorer.cov_penalty == "summary":
                self.topk_log_probs -= self.global_scorer.penalty_builder.coverage_summary(
                    self._coverage + attn, self.global_scorer.beta
                ).view(_B, self.beam_size)
            else:
                self.topk_log_probs -= self.global_scorer.penalty_builder.coverage_none(
                    self._coverage + attn, self.global_scorer.beta
                ).view(_B, self.beam_size)

        # force the output to be longer than self.min_length
        step = len(self)
        self.ensure_min_length(log_probs)
        self.ensure_unk_removed(log_probs)
        # Multiply probs by the beam probability.
        log_probs += self.topk_log_probs.view(_B * self.beam_size, 1)

        # if the sequence ends now, then the penalty is the current
        # length + 1, to include the EOS token
        #length_penalty = self.global_scorer.length_penalty(
        #    step + 1, alpha=self.global_scorer.alpha
        #)
        if self.global_scorer.length_penalty == "wu":
            length_penalty = self.global_scorer.penalty_builder.length_wu(
                step + 1, alpha=self.global_scorer.alpha
            )
        elif self.global_scorer.length_penalty == "avg":
            length_penalty = self.global_scorer.penalty_builder.length_average(
                step + 1, alpha=self.global_scorer.alpha
            )
        else:
            length_penalty = self.global_scorer.penalty_builder.length_none(
                step + 1, alpha=self.global_scorer.alpha
            )

        curr_scores = log_probs / length_penalty

        # Avoid any direction that would repeat unwanted ngrams
        self.block_ngram_repeats(curr_scores)

        # Pick up candidate token by curr_scores
        (self.topk_scores, self.topk_ids) = self._pick(curr_scores)

        # Recover log probs.
        # Length penalty is just a scalar. It doesn't matter if it's applied
        # before or after the topk.
        torch.mul(self.topk_scores, length_penalty, out=self.topk_log_probs)

        # Resolve beam origin and map to batch index flat representation.
        self._batch_index = torch.div(self.topk_ids, vocab_size, rounding_mode="trunc")
        self._batch_index += self._beam_offset[:_B].unsqueeze(1)
        self.select_indices = self._batch_index.view(_B * self.beam_size)
        self.topk_ids.fmod_(vocab_size)  # resolve true word ids

        # Append last prediction.
        self.alive_seq = torch.cat(
            [
                self.alive_seq.index_select(0, self.select_indices),
                self.topk_ids.view(_B * self.beam_size, 1),
            ],
            -1,
        )

        self.maybe_update_forbidden_tokens()

        if self.return_attention or self._cov_pen:
            current_attn = attn.index_select(0, self.select_indices)
            if step == 1:
                self.alive_attn = current_attn
                print("step 1")
                print(self.alive_attn.shape)
                # update global state (step == 1)
                if self._cov_pen:  # coverage penalty
                    self._prev_penalty = torch.zeros_like(self.topk_log_probs)
                    self._coverage = current_attn
            else:
                self.alive_attn = self.alive_attn.index_select(0, self.select_indices)
                self.alive_attn = torch.cat([self.alive_attn, current_attn], 1)
                print("step not 1")
                print(self.alive_attn.shape)
                # update global state (step > 1)
                if self._cov_pen:
                    self._coverage = self._coverage.index_select(0, self.select_indices)
                    self._coverage += current_attn
                    #self._prev_penalty = self.global_scorer.cov_penalty(
                    #    self._coverage, beta=self.global_scorer.beta
                    #).view(_B, self.beam_size)
                    if self.global_scorer.cov_penalty == "wu":
                        self._prev_penalty = self.global_scorer.penalty_builder.coverage_wu(
                            self._coverage, beta=self.global_scorer.beta
                        ).view(_B, self.beam_size)
                    elif self.global_scorer.cov_penalty == "summary":
                        self._prev_penalty = self.global_scorer.penalty_builder.coverage_summary(
                            self._coverage, beta=self.global_scorer.beta
                        ).view(_B, self.beam_size)
                    else:
                        self._prev_penalty = self.global_scorer.penalty_builder.coverage_none(
                            self._coverage, beta=self.global_scorer.beta
                        ).view(_B, self.beam_size)

        if self._vanilla_cov_pen:
            # shape: (batch_size x beam_size, 1)
            #cov_penalty = self.global_scorer.cov_penalty(
            #    self._coverage, beta=self.global_scorer.beta
            #)
            if self.global_scorer.cov_penalty == "wu":
                cov_penalty = self.global_scorer.penalty_builder.coverage_wu(
                    self._coverage, beta=self.global_scorer.beta
                )
            elif self.global_scorer.cov_penalty == "summary":
                cov_penalty = self.global_scorer.penalty_builder.coverage_summary(
                    self._coverage, beta=self.global_scorer.beta
                )
            else:
                cov_penalty = self.global_scorer.penalty_builder.coverage_none(
                    self._coverage, beta=self.global_scorer.beta
                )
            self.topk_scores -= cov_penalty.view(_B, self.beam_size).float()

        self.is_finished = self.topk_ids.eq(self.eos)
        self.ensure_max_length()

    # from BeamSearch
    def initialize(
            self, enc_out, src_len, device:Optional[torch.device]=None
    ):
        """Initialize for decoding.
        Repeat src objects `beam_size` times.
        """

        (fn_map_state, enc_out) = self.initialize_tile(
            enc_out, src_len
        )
        if device is None:
            device = self.get_device_from_enc_out(enc_out)

        self.initialize_(
            enc_out, self.src_len, device
        )

        return fn_map_state, enc_out, self.src_len

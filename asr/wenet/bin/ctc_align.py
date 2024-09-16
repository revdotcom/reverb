import math

wenet_space_symbol = '▁'


def id_to_token(tok, tokenizer):
    return tokenizer.detokenize([tok])[1][0]


def is_special_token(word):
    open_bracket = word.find('<')
    close_bracket = word.find('>')
    return (open_bracket != -1) & (close_bracket != -1) & (open_bracket < close_bracket)


def is_empty_word(word):
    return (word == "" or word == wenet_space_symbol)


def is_start_of_word_token(word):
    return (word.find(wenet_space_symbol) != -1)


def ctc_align(hypothesis, time_stamp, confidence_scores, tokenizer, 
              frame_shift_ms, time_shift_ms):
    """ Convert tokens to words and assign timestamps based on frame indices from CTC output
    """
    assert len(hypothesis) == len(time_stamp)
    word = ""
    unit_ids = []
    start_ts_ms = -1
    unit_start = -1
    path = []
    g_time_stamp_gap_ms = 100
    for i in range(len(hypothesis)):

        token = id_to_token(hypothesis[i], tokenizer)
        next_token = id_to_token(hypothesis[i + 1], tokenizer)if i + 1 < len(hypothesis) else wenet_space_symbol
        pos = token.find(wenet_space_symbol)

        # Trim starting _ if necessary
        if pos != -1:
            word += token[len(wenet_space_symbol):]
        else:
            word += token

        unit_ids.append(hypothesis[i])

        if start_ts_ms == -1:
            # To ensure start is always greater than 0
            start_ts_ms = max(time_stamp[i] * frame_shift_ms - g_time_stamp_gap_ms, 0)
            if i > 0:
                start_ts_ms = ((time_stamp[i - 1] + time_stamp[i]) // 2 * frame_shift_ms
                              if (time_stamp[i] - time_stamp[i - 1]) * frame_shift_ms < g_time_stamp_gap_ms
                              else start_ts_ms)
            unit_start = i

        # Cutting a word if the word is a special token
        if not is_empty_word(word) and is_special_token(word):
            end_ts_ms = time_stamp[i] * frame_shift_ms
            if i < len(hypothesis) - 1:
                end_ts_ms = ((time_stamp[i + 1] + time_stamp[i]) // 2 * frame_shift_ms
                             if (time_stamp[i + 1] - time_stamp[i]) * frame_shift_ms < g_time_stamp_gap_ms
                             else end_ts_ms)

            if confidence_scores:
                confidence = max(c for c in confidence_scores[unit_start:i+1])
            else:
                confidence = 0
            assert start_ts_ms < end_ts_ms
            assert len(unit_ids) == 1
            path.append({
                'word': word,
                'unit_id': unit_ids[0],
                'start_time_ms': start_ts_ms + time_shift_ms,
                'end_time_ms': end_ts_ms + time_shift_ms,
                'confidence': confidence,
                'unit_ids': unit_ids
            })

            start_ts_ms = -1
            unit_start = 0
            unit_ids = []
            word = ""

        # Cutting a word if next token starts from _ or next token is a special token
        if is_start_of_word_token(next_token) or is_special_token(next_token):
            end_ts_ms = time_stamp[i] * frame_shift_ms
            if i < len(hypothesis) - 1:
                end_ts_ms = ((time_stamp[i + 1] + time_stamp[i]) // 2 * frame_shift_ms
                             if (time_stamp[i + 1] - time_stamp[i]) * frame_shift_ms < g_time_stamp_gap_ms
                             else end_ts_ms)
            if not is_empty_word(word):
                assert len(unit_ids) > 0
                if confidence_scores:
                    confidence = max(c for c in confidence_scores[unit_start:i+1])
                else:
                    confidence = 0
                assert start_ts_ms <= end_ts_ms
                assert not is_special_token(word)
                path.append({
                    'word': word,
                    'unit_id': -1,
                    'start_time_ms': start_ts_ms + time_shift_ms,
                    'end_time_ms': end_ts_ms + time_shift_ms,
                    'confidence': confidence,
                    'unit_ids': unit_ids
                })
            start_ts_ms = -1
            unit_start = 0
            unit_ids = []
            word = ""
    return path


def adjust_model_time_offset(hypothesis, adjustment):
    if adjustment == 0:
        return

    adjusted_hyp = []
    for i in range(len(hypothesis)):
        word = hypothesis[i]
        assert word['start_time_ms'] >= 0
        assert word['start_time_ms'] <= word['end_time_ms']
        word_adjustment = 0
        if i == 0:
            word_adjustment = min(adjustment, word['start_time_ms'])
        else:
            prev_word = hypothesis[i-1]
            assert word['start_time_ms'] >= prev_word['end_time_ms'], f"ERROR! {word} >= {prev_word}"
            if word['start_time_ms'] >= prev_word['end_time_ms']:
                word_adjustment = min(adjustment, word['start_time_ms'] - prev_word['end_time_ms'])
        assert word_adjustment >= 0
        word['start_time_ms'] -= word_adjustment
        word['end_time_ms'] -= word_adjustment
        adjusted_hyp.append(word)
    
    return adjusted_hyp

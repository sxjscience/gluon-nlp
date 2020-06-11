import numpy as np
from typing import List


def get_trimmed_lengths(lengths: List[int],
                        max_length: int,
                        do_merge: bool = False) -> np.ndarray:
    """Get the trimmed lengths of multiple text data. It will make sure that
    the trimmed length is smaller than or equal to the max_length

    - do_merge is True
        Make sure that sum(trimmed_lengths) <= max_length.
        The strategy is to always try to trim the longer lengths.
    - do_merge is False
        Make sure that all(trimmed_lengths <= max_length)

    Parameters
    ----------
    lengths
        The original lengths of each sample
    max_length
        When do_merge is True,
            We set the max_length constraint on the total length.
        When do_merge is False,
            We set the max_length constraint on individual sentences.
    do_merge
        Whether these sentences will be merged

    Returns
    -------
    trimmed_lengths
        The trimmed lengths of the
    """
    lengths = np.array(lengths)
    if do_merge:
        total_length = sum(lengths)
        if total_length <= max_length:
            return lengths
        trimmed_lengths = np.zeros_like(lengths)
        while sum(trimmed_lengths) != max_length:
            remainder = max_length - sum(trimmed_lengths)
            budgets = lengths - trimmed_lengths
            nonzero_idx = (budgets > 0).nonzero()[0]
            nonzero_budgets = budgets[nonzero_idx]
            if remainder // len(nonzero_idx) == 0:
                for i in range(remainder):
                    trimmed_lengths[nonzero_idx[i]] += 1
            else:
                increment = min(min(nonzero_budgets), remainder // len(nonzero_idx))
                trimmed_lengths[nonzero_idx] += increment
        return trimmed_lengths
    else:
        return np.minimum(lengths, max_length)


def match_token_with_char_spans(token_offsets: np.ndarray,
                                spans: np.ndarray):
    """Match the span offsets with the character-level offsets.

    1: Set

        spans[0] = max(spans[0], token_offsets[0])
        spans[1] = min(spans[1], token_offsets[-1][1])

    2: We try to select the smallest number of tokens that cover the entity, i.e.

        token_offsets[start][0] <= spans[0] < token_offsets[start][1]
        token_offsets[end][0] < spans[1] <= token_offsets[end][1]

    3: If it is not possible, we will use the fallback strategy.

        We choose the smallest range of tokens that has the largest overlap with the
        selected span:



    Parameters
    ----------
    token_offsets
        The offsets of the input tokens. Must be sorted.
        That is, it will satisfy
            1. token_offsets[i][0] <= token_offsets[i][1]
            2. token_offsets[i][0] <= token_offsets[i + 1][0]
        Shape (#num_tokens, 2)
    spans
        The character-level offsets (begin/end) of the selected spans.
        Shape (#spans, 2)

    Returns
    -------
    span_token_start_end
        The token-level starts and ends.
    """


"""Backend-independent CTC path decoding and error counts."""


def decode_ctc_path(path, vocabulary):
    """Collapse repeated token IDs before removing blanks."""
    previous = None
    tokens = []
    for token in path:
        if token != previous and token != vocabulary.blank_id:
            tokens.append(token)
        previous = token
    return vocabulary.decode(tokens)


def edit_distance(hypothesis, reference):
    """Return Levenshtein edits for either character or word sequences."""
    previous = list(range(len(reference) + 1))
    for index, token in enumerate(hypothesis, start=1):
        current = [index]
        for offset, expected in enumerate(reference, start=1):
            current.append(min(
                previous[offset] + 1,
                current[offset - 1] + 1,
                previous[offset - 1] + (token != expected),
            ))
        previous = current
    return previous[-1]

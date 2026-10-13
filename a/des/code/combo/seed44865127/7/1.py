from __future__ import annotations

import numpy as np


GREY = 5
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
RULE_RETRIES = 200
GEN_RETRIES = 400


def _compatible(partial: np.ndarray, full: np.ndarray) -> bool:
    mask = partial != GREY
    return bool(np.all(partial[mask] == full[mask]))


def _unique_complete_lines(lines: np.ndarray) -> list[np.ndarray]:
    seen: set[tuple[int, ...]] = set()
    patterns: list[np.ndarray] = []
    for line in lines:
        if np.any(line == GREY):
            continue
        key = tuple(map(int, line))
        if key in seen:
            continue
        seen.add(key)
        patterns.append(np.array(line, dtype=int))
    return patterns


def _assign_lines(lines: np.ndarray, patterns: list[np.ndarray]) -> np.ndarray:
    ids = np.full(lines.shape[0], -1, dtype=int)
    for idx, line in enumerate(lines):
        matches = [pid for pid, pattern in enumerate(patterns) if _compatible(line, pattern)]
        if len(matches) == 1:
            ids[int(idx)] = int(matches[0])
    return ids


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair grey corruption in a row-color/column-color intersection table.

    Input structure:
    - Rows belong to a small set of repeated row types.
    - Columns belong to a small set of repeated column types.
    - Every cell color is determined by the combination of its row type and column type.
    - Grey cells are corrupted intersections that should be restored.

    Rule:
    - Use intact cells to recover the repeated row patterns and repeated column patterns.
    - Infer the row-type × column-type color mapping from the intact intersections.
    - Replace each grey cell with the color implied by its row type and column type.
    - Leave every non-grey cell unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    if not np.any(g == GREY):
        return out

    row_patterns = _unique_complete_lines(g)
    col_patterns = _unique_complete_lines(g.T)
    assert row_patterns, "expected at least one intact representative row"
    assert col_patterns, "expected at least one intact representative column"

    row_ids = _assign_lines(g, row_patterns)
    col_ids = _assign_lines(g.T, col_patterns)
    assert np.all(row_ids >= 0), "every row must match exactly one recovered row pattern"
    assert np.all(col_ids >= 0), "every column must match exactly one recovered column pattern"

    row_reps = {rid: int(np.flatnonzero(row_ids == rid)[0]) for rid in range(len(row_patterns))}
    col_reps = {cid: int(np.flatnonzero(col_ids == cid)[0]) for cid in range(len(col_patterns))}

    pair_value: dict[tuple[int, int], int] = {}
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            value = int(g[r, c])
            if value == GREY:
                continue
            key = (int(row_ids[r]), int(col_ids[c]))
            if key in pair_value:
                assert pair_value[key] == value, "intersection mapping must be consistent"
            else:
                pair_value[key] = value

    for rid in range(len(row_patterns)):
        for cid in range(len(col_patterns)):
            if (rid, cid) in pair_value:
                continue
            rr = row_reps[rid]
            cc = col_reps[cid]
            row_value = int(row_patterns[rid][cc])
            col_value = int(col_patterns[cid][rr])
            assert row_value == col_value, "row and column prototypes disagree"
            pair_value[(rid, cid)] = row_value

    for r, c in np.argwhere(g == GREY):
        out[int(r), int(c)] = int(pair_value[(int(row_ids[int(r)]), int(col_ids[int(c)]))])

    return out


def _sample_rule_matrix(row_type_count: int, col_type_count: int, rng: np.random.Generator) -> np.ndarray:
    target_imbalance = (row_type_count * col_type_count) % 2
    for _ in range(RULE_RETRIES):
        mask = rng.integers(0, 2, size=(row_type_count, col_type_count), dtype=int).astype(bool)
        if not np.all(mask.any(axis=1)):
            continue
        if not np.all((~mask).any(axis=1)):
            continue
        if not np.all(mask.any(axis=0)):
            continue
        if not np.all((~mask).any(axis=0)):
            continue
        if abs(int(mask.sum()) * 2 - row_type_count * col_type_count) > target_imbalance:
            continue
        return mask
    raise RuntimeError("failed to build a nondegenerate row/column rule matrix")


def _spread_labels(counts: list[int], rng: np.random.Generator) -> list[int]:
    remaining = counts[:]
    labels: list[int] = []
    while sum(remaining) > 0:
        choices = [idx for idx, left in enumerate(remaining) if left > 0 and (not labels or idx != labels[-1])]
        if not choices:
            choices = [idx for idx, left in enumerate(remaining) if left > 0]
        best = max(remaining[idx] for idx in choices)
        tied = [idx for idx in choices if remaining[idx] == best]
        pick = int(tied[int(rng.integers(0, len(tied)))])
        labels.append(pick)
        remaining[pick] -= 1
    return labels


def _render_grid(
    row_labels: list[int],
    col_labels: list[int],
    row_colors: np.ndarray,
    col_colors: np.ndarray,
    row_wins: np.ndarray,
) -> np.ndarray:
    height = len(row_labels)
    width = len(col_labels)
    grid = np.empty((height, width), dtype=int)
    for r, row_type in enumerate(row_labels):
        for c, col_type in enumerate(col_labels):
            if bool(row_wins[int(row_type), int(col_type)]):
                grid[r, c] = int(row_colors[int(row_type)])
            else:
                grid[r, c] = int(col_colors[int(col_type)])
    return grid


def _representative_indices(labels: list[int], class_count: int, rng: np.random.Generator) -> list[int]:
    reps = []
    for label in range(class_count):
        positions = [idx for idx, value in enumerate(labels) if value == label]
        reps.append(int(positions[int(rng.integers(0, len(positions)))]))
    return reps


def _pollute_grid(
    clean: np.ndarray,
    protected_rows: list[int],
    protected_cols: list[int],
    rng: np.random.Generator,
) -> np.ndarray:
    inp = clean.copy()
    candidate_rows = [r for r in range(clean.shape[0]) if r not in protected_rows]
    candidate_cols = [c for c in range(clean.shape[1]) if c not in protected_cols]
    rng.shuffle(candidate_rows)
    rng.shuffle(candidate_cols)

    max_corrupt = min(len(candidate_rows), len(candidate_cols))
    assert max_corrupt >= 1, "need at least one unprotected row and column"
    corrupt_count = int(rng.integers(1, max_corrupt + 1))
    for r, c in zip(candidate_rows[:corrupt_count], candidate_cols[:corrupt_count]):
        inp[int(r), int(c)] = GREY
    return inp


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    row_type_count = int(rng.integers(2, 4))
    col_type_count = int(rng.integers(2, 4))
    row_counts = [int(rng.integers(2, 5)) for _ in range(row_type_count)]
    col_counts = [int(rng.integers(2, 5)) for _ in range(col_type_count)]

    colors = rng.choice(PALETTE, size=row_type_count + col_type_count, replace=False)
    row_colors = np.array(colors[:row_type_count], dtype=int)
    col_colors = np.array(colors[row_type_count:], dtype=int)
    row_wins = _sample_rule_matrix(row_type_count, col_type_count, rng)

    row_labels = _spread_labels(row_counts, rng)
    col_labels = _spread_labels(col_counts, rng)
    clean = _render_grid(row_labels, col_labels, row_colors, col_colors, row_wins)

    if len({tuple(map(int, row)) for row in clean}) != row_type_count:
        return None
    if len({tuple(map(int, col)) for col in clean.T}) != col_type_count:
        return None

    protected_rows = _representative_indices(row_labels, row_type_count, rng)
    protected_cols = _representative_indices(col_labels, col_type_count, rng)
    inp = _pollute_grid(clean, protected_rows, protected_cols, rng)

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, clean):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

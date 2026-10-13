from __future__ import annotations

import random

import numpy as np


MOVING_GROUPS = ((1, 8), (4, 7))
MOVING_SETS = {frozenset(group) for group in MOVING_GROUPS}
DISTRACTOR_COLORS = (2, 6)
MAX_ATTEMPTS = 400
_RNG = np.random.default_rng()


def _find_pair_columns(row: np.ndarray) -> tuple[int, int] | None:
    for group in MOVING_GROUPS:
        cols = []
        for color in group:
            hits = np.flatnonzero(row == color)
            if len(hits) != 1:
                cols = []
                break
            cols.append(int(hits[0]))
        if len(cols) == 2:
            return tuple(sorted(cols))
    return None


def _slide_pairs(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for r in range(out.shape[0]):
        pair_cols = _find_pair_columns(out[r])
        if pair_cols is None:
            continue
        c_left, c_right = pair_cols
        dst = c_left + 1
        if c_right != dst:
            out[r, dst] = out[r, c_right]
            out[r, c_right] = 0
    return out


def _is_vertically_symmetric(compacted: np.ndarray) -> bool:
    nonempty_rows = np.any(compacted != 0, axis=1)
    if not np.any(nonempty_rows):
        return True
    mask = compacted[nonempty_rows] != 0
    return np.array_equal(mask, np.fliplr(mask))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide same-family row pairs together, then branch on whether the compacted silhouette is vertically symmetric.

    Input structure:
    - The grid has a black background with mostly empty rows.
    - A row may contain one moving pair from the blue/teal family or yellow/orange family.
    - Some rows may also contain one red or magenta distractor pixel that never moves.

    Rule:
    - In each moving-pair row, keep the left family pixel fixed and slide the right family pixel left until it becomes adjacent.
    - Leave distractor pixels unchanged.
    - Check the compacted grid for left-right mirror symmetry using the non-black occupancy pattern.
    - If symmetric, return the left half of the compacted grid; otherwise append a left-right mirror copy on the right.

    Key insight:
    - The slide is only an intermediate step: it changes the geometry that decides whether the puzzle crops or mirrors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    compacted = _slide_pairs(g)
    if _is_vertically_symmetric(compacted):
        return compacted[:, : compacted.shape[1] // 2].copy()
    return np.concatenate([compacted, np.fliplr(compacted)], axis=1)


def _random_pair() -> tuple[int, int]:
    pair = MOVING_GROUPS[int(_RNG.integers(0, len(MOVING_GROUPS)))]
    if bool(_RNG.integers(0, 2)):
        return pair
    return pair[::-1]


def _build_symmetric_post(rows: int, cols: int) -> np.ndarray:
    post = np.zeros((rows, cols), dtype=int)
    count = int(_RNG.integers(2, min(rows, 6) + 1))
    chosen_rows = _RNG.choice(rows, size=count, replace=False)
    left_col = cols // 2 - 1
    for r in chosen_rows:
        left_color, right_color = _random_pair()
        post[int(r), left_col] = left_color
        post[int(r), left_col + 1] = right_color
    return post


def _free_columns(post: np.ndarray, row: int) -> list[int]:
    return [c for c in range(post.shape[1]) if post[row, c] == 0]


def _add_asymmetric_distractors(post: np.ndarray, used_rows: set[int]) -> None:
    available_rows = list(range(post.shape[0]))
    _RNG.shuffle(available_rows)
    count = int(_RNG.integers(0, min(3, post.shape[0]) + 1))
    added = 0
    for row in available_rows:
        if added >= count:
            break
        if np.count_nonzero(post[row]) >= 3:
            continue
        free = _free_columns(post, row)
        if not free:
            continue
        c = int(free[int(_RNG.integers(0, len(free)))])
        post[row, c] = DISTRACTOR_COLORS[int(_RNG.integers(0, len(DISTRACTOR_COLORS)))]
        used_rows.add(row)
        added += 1


def _build_asymmetric_post(rows: int, cols: int) -> np.ndarray:
    post = np.zeros((rows, cols), dtype=int)
    count = int(_RNG.integers(2, min(rows, 6) + 1))
    chosen_rows = list(map(int, _RNG.choice(rows, size=count, replace=False)))
    center_left = cols // 2 - 1
    off_center_rows = 0

    for row in chosen_rows:
        candidates = [c for c in range(cols - 1)]
        left_col = int(candidates[int(_RNG.integers(0, len(candidates)))])
        if left_col == center_left and bool(_RNG.integers(0, 2)):
            left_col = center_left - 1 if center_left > 0 else center_left + 1
        if left_col != center_left:
            off_center_rows += 1
        left_color, right_color = _random_pair()
        post[row, left_col] = left_color
        post[row, left_col + 1] = right_color

    if off_center_rows == 0:
        row = chosen_rows[0]
        left_color = int(post[row, center_left])
        right_color = int(post[row, center_left + 1])
        post[row, center_left:center_left + 2] = 0
        new_left = center_left - 1 if center_left > 0 else center_left + 1
        post[row, new_left] = left_color
        post[row, new_left + 1] = right_color

    _add_asymmetric_distractors(post, set(chosen_rows))
    return post


def _expand_to_input(post: np.ndarray) -> np.ndarray | None:
    inp = post.copy()
    for r in range(inp.shape[0]):
        pair_cols = _find_pair_columns(post[r])
        if pair_cols is None:
            continue
        c_left, c_right = pair_cols
        far_choices = [c for c in range(c_right + 1, inp.shape[1]) if inp[r, c] == 0]
        if not far_choices:
            return None
        dst = int(far_choices[int(_RNG.integers(0, len(far_choices)))])
        inp[r, dst] = inp[r, c_right]
        inp[r, c_right] = 0
    return inp


def _gen_attempt() -> np.ndarray | None:
    rows = int(_RNG.integers(7, 13))
    cols = int(_RNG.integers(4, 8)) * 2
    symmetric_branch = bool(_RNG.integers(0, 2))

    post = _build_symmetric_post(rows, cols) if symmetric_branch else _build_asymmetric_post(rows, cols)
    inp = _expand_to_input(post)
    if inp is None:
        return None

    if symmetric_branch != _is_vertically_symmetric(post):
        return None
    if not np.array_equal(_slide_pairs(inp), post):
        return None

    out = main(inp)
    if symmetric_branch:
        expected = post[:, : cols // 2]
    else:
        expected = np.concatenate([post, np.fliplr(post)], axis=1)
    if not np.array_equal(out, expected):
        return None

    if max(inp.shape) > 30 or max(out.shape) > 30:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid sample")
